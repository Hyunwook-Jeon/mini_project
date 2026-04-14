#!/usr/bin/env python3
"""
LabelMe JSON(폴리곤) → YOLO 형식(images/train|val, labels/train|val, data.yaml).

각 .json 옆(또는 imagePath가 가리키는 상대 경로)에 원본 이미지가 있어야 합니다.
이미지가 없으면 해당 샘플은 건너뜁니다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any


def collect_class_names(raw_root: Path) -> list[str]:
    names: set[str] = set()
    for p in raw_root.rglob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for s in data.get("shapes") or []:
            lab = (s.get("label") or "").strip()
            if lab:
                names.add(lab)
    return sorted(names)


def resolve_image_path(json_path: Path, data: dict[str, Any]) -> Path | None:
    rel = data.get("imagePath") or ""
    if not rel:
        return None
    cand = (json_path.parent / rel).resolve()
    if cand.is_file():
        return cand
    stem = json_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        c = json_path.with_suffix(ext)
        if c.is_file():
            return c.resolve()
    return None


def shape_to_yolo_line(
    shape: dict[str, Any], w: int, h: int, class_to_id: dict[str, int]
) -> str | None:
    label = (shape.get("label") or "").strip()
    if label not in class_to_id:
        return None
    cid = class_to_id[label]
    st = shape.get("shape_type") or "polygon"
    pts = shape.get("points") or []
    if not pts:
        return None

    if st == "rectangle" and len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[0], pts[1]
        xmin, xmax = min(float(x1), float(x2)), max(float(x1), float(x2))
        ymin, ymax = min(float(y1), float(y2)), max(float(y1), float(y2))
    else:
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

    xmin = max(0.0, min(xmin, float(w)))
    xmax = max(0.0, min(xmax, float(w)))
    ymin = max(0.0, min(ymin, float(h)))
    ymax = max(0.0, min(ymax, float(h)))
    bw, bh = xmax - xmin, ymax - ymin
    if bw < 1.0 or bh < 1.0:
        return None

    xc = ((xmin + xmax) / 2.0) / float(w)
    yc = ((ymin + ymax) / 2.0) / float(h)
    nw = bw / float(w)
    nh = bh / float(h)
    xc = min(1.0, max(0.0, xc))
    yc = min(1.0, max(0.0, yc))
    nw = min(1.0, max(0.0, nw))
    nh = min(1.0, max(0.0, nh))
    return f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="LabelMe .json 이 있는 루트(하위 폴더 포함)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "yolo_final",
        help="YOLO 데이터셋 출력 디렉터리",
    )
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_root: Path = args.raw_root.resolve()
    out_root: Path = args.out.resolve()
    random.seed(args.seed)

    class_names = collect_class_names(raw_root)
    if not class_names:
        raise SystemExit("클래스를 찾을 수 없습니다. .json 안에 shapes.label 이 있는지 확인하세요.")

    class_to_id = {n: i for i, n in enumerate(class_names)}

    pairs: list[tuple[Path, Path]] = []
    missing_img = 0
    for jp in sorted(raw_root.rglob("*.json")):
        if out_root in jp.parents:
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"skip (read error): {jp} ({e})")
            continue
        ip = resolve_image_path(jp, data)
        if ip is None:
            missing_img += 1
            print(f"skip (no image): {jp}")
            continue
        pairs.append((jp, ip))

    if not pairs:
        raise SystemExit(
            "이미지가 있는 샘플이 0개입니다. 각 .json과 같은 폴더에 "
            "imagePath 파일(예: .jpg)을 두거나 파일명을 맞춰 주세요."
        )

    random.shuffle(pairs)
    n = len(pairs)
    if n == 1:
        train_pairs = list(pairs)
        val_pairs = list(pairs)
    elif n == 2:
        train_pairs, val_pairs = [pairs[0]], [pairs[1]]
    else:
        n_val = max(1, int(round(n * args.val_ratio)))
        if n_val >= n:
            n_val = max(1, n // 5)
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    def slug(json_path: Path) -> str:
        rel = json_path.relative_to(raw_root)
        return str(rel).replace(os.sep, "__").replace(".json", "")

    def process(split: str, items: list[tuple[Path, Path]]) -> None:
        for jp, src_img in items:
            data = json.loads(jp.read_text(encoding="utf-8"))
            w = int(data.get("imageWidth") or 0)
            h = int(data.get("imageHeight") or 0)
            if w <= 0 or h <= 0:
                print(f"skip (bad size): {jp}")
                continue

            base = slug(jp)
            ext = src_img.suffix.lower() or ".jpg"
            dst_img = out_root / "images" / split / f"{base}{ext}"
            shutil.copy2(src_img, dst_img)

            lines: list[str] = []
            for sh in data.get("shapes") or []:
                line = shape_to_yolo_line(sh, w, h, class_to_id)
                if line:
                    lines.append(line)

            label_path = out_root / "labels" / split / f"{base}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    process("train", train_pairs)
    process("val", val_pairs)

    yaml_path = out_root / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml_path.write_text(
        f"path: {out_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )

    print(f"classes ({len(class_names)}): {class_names}")
    print(f"train: {len(train_pairs)}  val: {len(val_pairs)}  skipped_no_image: {missing_img}")
    print(f"wrote: {yaml_path}")


if __name__ == "__main__":
    main()
