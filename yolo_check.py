#!/usr/bin/env python3
"""
학습 결과 확인 + Ultralytics YOLO로 검증(val) / 추론 시각화(predict).

  python yolo_check.py summary
  python yolo_check.py val
  python yolo_check.py predict --source datasets/yolo_final/images/val
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from ultralytics import YOLO


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def find_best_pt(search_under: Path) -> Path | None:
    cands = list(search_under.rglob("best.pt"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def default_data_yaml() -> Path:
    return repo_root() / "datasets" / "yolo_final" / "data.yaml"


def cmd_summary(args: argparse.Namespace) -> None:
    root = repo_root()
    proj = Path(args.project).resolve()
    run_dir = proj / args.name if args.name else proj

    print(f"학습 루트(탐색): {run_dir}\n")

    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        with results_csv.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            ep = last.get("epoch", "?")
            print(f"마지막 epoch (results.csv): {ep}")
            keys = [
                ("정밀도 P", "metrics/precision(B)"),
                ("재현율 R", "metrics/recall(B)"),
                ("mAP50", "metrics/mAP50(B)"),
                ("mAP50-95", "metrics/mAP50-95(B)"),
            ]
            for label, k in keys:
                if k in last:
                    print(f"  {label}: {last[k]}")
            print(f"\n전체 로그: {results_csv}")
    else:
        print(f"results.csv 없음: {results_csv}")

    args_yaml = run_dir / "args.yaml"
    if args_yaml.is_file():
        print(f"\n학습 인자: {args_yaml}")

    plots = [
        run_dir / "results.png",
        run_dir / "confusion_matrix.png",
        run_dir / "confusion_matrix_normalized.png",
        run_dir / "BoxPR_curve.png",
        run_dir / "BoxF1_curve.png",
    ]
    found = [p for p in plots if p.is_file()]
    if found:
        print("\n생성된 그래프(파일 탐색기로 열어보면 됨):")
        for p in found:
            print(f"  {p}")
    else:
        print("\n(그래프 png는 이 경로에 없을 수 있습니다. 학습 시 plots=True 였는지 확인.)")

    w = find_best_pt(proj)
    if w:
        print(f"\n가중치(best.pt, 최신): {w}")
    else:
        print(
            f"\n[주의] best.pt 를 {proj} 아래에서 찾지 못했습니다. "
            "학습이 끝난 폴더의 weights/best.pt 경로를 --weights 로 지정하세요."
        )


def cmd_val(args: argparse.Namespace) -> None:
    data = Path(args.data).resolve()
    if not data.is_file():
        raise SystemExit(f"data.yaml 없음: {data}")

    w = Path(args.weights).resolve() if args.weights else find_best_pt(Path(args.project).resolve())
    if w is None or not w.is_file():
        raise SystemExit(
            "가중치(best.pt)를 찾을 수 없습니다. 예:\n"
            "  python yolo_check.py val --weights runs/weights/weights/best.pt"
        )

    device = args.device or (0 if torch.cuda.is_available() else "cpu")
    model = YOLO(str(w))
    model.val(data=str(data), imgsz=args.imgsz, batch=args.batch, device=device, plots=True, save_json=args.save_json)


def cmd_predict(args: argparse.Namespace) -> None:
    src = Path(args.source).resolve()
    if not src.exists():
        raise SystemExit(f"소스 없음: {src}")

    w = Path(args.weights).resolve() if args.weights else find_best_pt(Path(args.project).resolve())
    if w is None or not w.is_file():
        raise SystemExit(
            "가중치(best.pt)를 찾을 수 없습니다. 예:\n"
            "  python yolo_check.py predict --source ... --weights runs/weights/weights/best.pt"
        )

    device = args.device or (0 if torch.cuda.is_available() else "cpu")
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(w))
    model.predict(
        source=str(src),
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        save=True,
        project=str(out.parent),
        name=out.name,
        exist_ok=True,
    )
    print(f"\n박스 그려진 결과 저장 위치: {out}")


def main() -> None:
    root = repo_root()
    ap = argparse.ArgumentParser(description="YOLO 학습 결과 확인 / val / predict")
    ap.add_argument(
        "--project",
        type=Path,
        default=root / "runs",
        help="train 시 project 로 준 상위 폴더 (기본: ./runs)",
    )

    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summary", help="results.csv·그래프·best.pt 위치 출력")
    p_sum.add_argument(
        "--name",
        type=str,
        default="weights",
        help="train 시 name= (기본: weights → runs/weights/)",
    )
    p_sum.set_defaults(func=cmd_summary)

    p_val = sub.add_parser("val", help="검증 세트로 메트릭 재계산 (plots 저장)")
    p_val.add_argument("--data", type=Path, default=default_data_yaml())
    p_val.add_argument("--weights", type=str, default="", help="비우면 runs 아래 최신 best.pt")
    p_val.add_argument("--imgsz", type=int, default=640)
    p_val.add_argument("--batch", type=int, default=8)
    p_val.add_argument("--device", type=str, default="")
    p_val.add_argument("--save-json", action="store_true", help="COCO 형식 JSON 저장")
    p_val.set_defaults(func=cmd_val)

    p_pred = sub.add_parser("predict", help="이미지/폴더 추론 + 박스 그림 저장")
    p_pred.add_argument(
        "--source",
        type=Path,
        default=root / "datasets" / "yolo_final" / "images" / "val",
        help="이미지 파일 또는 폴더",
    )
    p_pred.add_argument("--weights", type=str, default="", help="비우면 runs 아래 최신 best.pt")
    p_pred.add_argument("--imgsz", type=int, default=640)
    p_pred.add_argument("--conf", type=float, default=0.25)
    p_pred.add_argument("--device", type=str, default="")
    p_pred.add_argument(
        "--out",
        type=Path,
        default=root / "runs" / "predict_preview",
        help="결과 저장 프로젝트 하위 폴더명까지 (Ultralytics project/name)",
    )
    p_pred.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
