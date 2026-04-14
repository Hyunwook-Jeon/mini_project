#!/usr/bin/env python3
"""
학습한 YOLO 가중치로 웹캠 실시간 추론 (OpenCV 창).

  source .venv/bin/activate
  python yolo_live_cam.py

USB 웹캠만 쓰는 경우(내장 0, USB 1인 경우가 많음) 기본이 --camera 1.
어느 번호인지 모를 때: python yolo_live_cam.py --list-cameras

종료: 창이 포커스일 때 q 키
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def find_best_pt(search_under: Path) -> Path | None:
    cands = list(search_under.rglob("best.pt"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def open_capture(index: int) -> cv2.VideoCapture:
    """Linux에서는 V4L2로 USB 캠 인식이 더 안정적인 경우가 많음."""
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def list_cameras(max_index: int = 8) -> None:
    print("사용 가능한 카메라 인덱스 (프레임을 한 장 읽을 수 있는 것만 표시):\n")
    any_ok = False
    for i in range(max_index + 1):
        cap = open_capture(i)
        if not cap.isOpened():
            continue
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"  [{i}] 열리지만 프레임 없음 (스킵)")
            continue
        h, w = frame.shape[:2]
        print(f"  [{i}] OK  해상도 약 {w}x{h}")
        any_ok = True
    if not any_ok:
        print("  (없음) — 권한·케이블·다른 앱 점유 여부를 확인하세요.")
    else:
        print("\n실행 예: python yolo_live_cam.py --camera <위 번호>")


def main() -> None:
    root = repo_root()
    ap = argparse.ArgumentParser(description="웹캠 + YOLO 실시간 검출")
    ap.add_argument(
        "--weights",
        type=str,
        default="",
        help="비우면 runs/ 아래 최신 best.pt",
    )
    ap.add_argument(
        "--project",
        type=Path,
        default=root / "runs",
        help="best.pt 검색 루트",
    )
    ap.add_argument(
        "--camera",
        type=int,
        default=1,
        help="VideoCapture 인덱스. 내장+USB 같이 쓰면 USB가 보통 1 (단독 USB면 0일 수 있음)",
    )
    ap.add_argument(
        "--list-cameras",
        action="store_true",
        help="0~8번 카메라를 열어보고 쓸 수 있는 인덱스만 출력 후 종료",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument(
        "--device",
        type=str,
        default="",
        help="예: 0 또는 cpu. 비우면 CUDA 있으면 0",
    )
    ap.add_argument(
        "--no-half",
        action="store_true",
        help="GPU에서도 FP16 끄기 (호환성 문제 시)",
    )
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    wpath: Path
    if args.weights:
        wpath = Path(args.weights).resolve()
    else:
        found = find_best_pt(Path(args.project).resolve())
        if found is None:
            raise SystemExit(
                f"best.pt 를 찾을 수 없습니다. 예:\n"
                f"  python yolo_live_cam.py --weights {root}/runs/weights/weights/best.pt"
            )
        wpath = found

    if not wpath.is_file():
        raise SystemExit(f"가중치 파일 없음: {wpath}")

    dev = args.device or (0 if torch.cuda.is_available() else "cpu")
    use_half = (not args.no_half) and dev != "cpu" and torch.cuda.is_available()

    print(f"가중치: {wpath}")
    print(f"device={dev}, half={use_half}, 카메라 index={args.camera}")
    print("종료: 창 선택 후 q 키")

    model = YOLO(str(wpath))

    cap = open_capture(args.camera)
    if not cap.isOpened():
        raise SystemExit(
            f"카메라를 열 수 없습니다 (index={args.camera}).\n"
            "  python yolo_live_cam.py --list-cameras\n"
            "로 번호를 확인한 뒤 --camera N 으로 다시 실행해 보세요."
        )

    # 가능하면 해상도 올리기 (실패해도 무시)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = "YOLO — q 로 종료"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임 읽기 실패, 종료합니다.")
            break

        t0 = time.perf_counter()
        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=dev,
            half=use_half,
            verbose=False,
        )
        t1 = time.perf_counter()
        out = results[0].plot()

        fps = 1.0 / max(t1 - t0, 1e-6)
        cv2.putText(
            out,
            f"FPS ~{fps:.1f}  conf={args.conf}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window, out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
