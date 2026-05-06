# Contributing

이 저장소는 팀원들이 함께 개발하고 실험할 수 있도록 만든 ROS 2 패키지입니다.

## 기본 원칙

- 공통 설정은 저장소에 남기고, 개인 장비 전용 값은 launch 인수나 별도 메모로 관리합니다.
- 큰 모델 파일, 로그, 개인 실험 결과물은 Git에 올리지 않습니다.
- 실제 로봇에 영향을 주는 변경은 가상 모드에서 먼저 확인합니다.

## 작업 흐름

### 1. 저장소 받기

```bash
git clone <REPOSITORY_URL>
cd dsr_realsense_pick_place
```

### 2. 브랜치 생성

```bash
git checkout -b feature/<short-name>
```

### 3. 변경 후 확인

최소한 아래 항목은 확인하는 것을 권장합니다.

```bash
python3 -m py_compile \
  setup.py \
  dsr_realsense_pick_place/object_detector.py \
  dsr_realsense_pick_place/pick_place_node.py \
  launch/pick_place.launch.py

python3 scripts/check_gui_integrity.py
```

워크스페이스 루트에서는 아래도 함께 확인합니다.

```bash
cd ~/ros2_ws
colcon build --packages-select dsr_realsense_pick_place
```

## 커밋 규칙

- 한 커밋에는 하나의 목적만 담습니다.
- 커밋 메시지는 짧고 분명하게 씁니다.

예시:

- `add README and requirements`
- `fix service wait logic in pick place node`
- `update launch defaults for e0509`

## Pull Request 권장 내용

- 무엇을 바꿨는지
- 왜 바꿨는지
- 어떻게 테스트했는지
- 실제 장비 영향이 있는지

## 주의할 점

- `pick_place.launch.py`의 TF 값은 예시값일 수 있으니 함부로 공통 기본값을 바꾸지 않습니다.
- 실제 로봇 IP, 카메라 시리얼, hand-eye calibration 값은 팀 합의 없이 하드코딩하지 않습니다.
- 그리퍼 IO 번호 변경 시 README 또는 설정 파일 설명도 같이 수정합니다.
