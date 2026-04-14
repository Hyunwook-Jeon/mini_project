#!/usr/bin/env python3
"""
RTX 2050 노트북 등 단일 GPU 환경용 Ultralytics YOLO 학습.

사전 실행: python prepare_yolo_dataset.py
VRAM 부족(OOM) 시 --batch 8 또는 --model yolov8n.pt 로 낮추세요.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO


def pick_workers() -> int:
    n = os.cpu_count() or 8
    # 데이터 로더: 너무 크면 노트북에서 디스크/CPU 병목·발열
    return max(2, min(8, n - 1))


def main() -> None:
    root = Path(__file__).resolve().parent
    default_yaml = root / "datasets" / "yolo_final" / "data.yaml"

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=default_yaml, help="data.yaml 경로")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="yolov8n/s/m 등 사전학습 가중치")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--device",
        type=str,
        default="",
        help="비우면 CUDA 있으면 0, 없으면 cpu",
    )
    ap.add_argument(
        "--project",
        type=Path,
        default=root / "runs",
        help="학습 결과 상위 폴더",
    )
    args = ap.parse_args()

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise SystemExit(
            f"data.yaml 없음: {data_path}\n"
            "먼저 이 폴더에서: python prepare_yolo_dataset.py"
        )

    if args.device:
        device_type: str | int = args.device
    else:
        device_type = 0 if torch.cuda.is_available() else "cpu"

    base_save_path = str(args.project.resolve())
    os.makedirs(base_save_path, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=os.path.abspath(str(data_path)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device_type,
        workers=pick_workers(),
        project=base_save_path,
        name="weights",
        exist_ok=True,
        plots=True,
        save=True,
        amp=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
        degrees=15,
        perspective=0.0005,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
    )


if __name__ == "__main__":
    main()
