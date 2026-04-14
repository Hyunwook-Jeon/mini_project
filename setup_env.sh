#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -U pip
# RTX 2050: CUDA 12.x 드라이버가 있으면 아래 한 줄로 torch GPU 설치 (권장)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

echo "완료. 활성화: source .venv/bin/activate"
echo "데이터 준비: python prepare_yolo_dataset.py"
echo "학습:        python train_yolo.py"
