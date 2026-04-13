# dsr_realsense_pick_place

Doosan `E0509` 로봇과 Intel RealSense 카메라를 이용해 객체를 인식하고 Pick & Place를 수행하는 ROS 2 패키지입니다.

이 패키지는 아래 두 노드로 구성됩니다.

- `object_detector`: RGB-D 이미지에서 물체를 검출하고 로봇 베이스 좌표계 기준 위치를 발행
- `pick_place_node`: 검출된 물체 좌표를 받아 Pick & Place 상태머신 수행
- `gui_node`: 카메라 영상과 검출 물체 버튼을 보여 주고, 사용자가 집을 물체를 선택

## 주요 구성

- `launch/pick_place.launch.py`
  - Doosan bringup
  - RealSense 카메라
  - 카메라-로봇 정적 TF
  - 객체 검출 노드
  - Pick & Place 노드
- `config/pick_place_params.yaml`
  - 카메라 토픽, YOLO 설정, 작업영역, 그리퍼 IO, 로봇 속도 설정
- `dsr_realsense_pick_place/object_detector.py`
  - RGB + Depth 기반 객체 위치 계산
  - TF를 이용한 카메라 좌표계 → 로봇 베이스 좌표계 변환
- `dsr_realsense_pick_place/pick_place_node.py`
  - IDLE → DETECTING → PRE_PICK → PICK → LIFT → MOVE_TO_PLACE → PLACE → HOME
- `dsr_realsense_pick_place/gui_node.py`
  - PyQt5 GUI
  - 물체 선택 버튼
  - 현재 상태 표시

## 요구 환경

- Ubuntu 22.04
- ROS 2 Humble
- Doosan ROS 2 패키지
- `realsense2_camera`
- Python 패키지
  - `numpy`
  - `opencv-python` 또는 `python3-opencv`
  - `ultralytics` 선택 사항

## 설치해야 하는 것

팀원이 처음 세팅할 때는 아래 항목을 먼저 준비하면 됩니다.

### 1. ROS 2 기본 패키지

```bash
sudo apt update
sudo apt install -y \
  ros-humble-cv-bridge \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  ros-humble-vision-msgs
```

### 2. RealSense ROS 패키지

환경에 따라 패키지명이 다를 수 있지만, 보통 아래 패키지를 준비합니다.

```bash
sudo apt update
sudo apt install -y ros-humble-realsense2-camera
```

### 3. Python 패키지

```bash
sudo apt update
sudo apt install -y python3-numpy python3-opencv python3-pyqt5
pip install ultralytics
```

`ultralytics`는 선택 사항이지만, 설치하지 않으면 YOLO 대신 단순 색상 기반 fallback 검출을 사용합니다.

또는 저장소 안의 `requirements.txt`를 사용할 수 있습니다.

```bash
cd ~/ros2_ws/src/dsr_realsense_pick_place
pip install -r requirements.txt
```

### 4. Doosan 패키지

이 패키지는 Doosan ROS 2 패키지들이 같이 있어야 정상 동작합니다.

필수로 필요한 패키지 예:

- `dsr_bringup2`
- `dsr_msgs2`
- `dsr_controller2`
- `dsr_description2`
- `dsr_moveit2`

현재 프로젝트에서는 `~/ros2_ws/src/doosan-robot2` 아래에 같이 두는 방식을 사용하고 있습니다.

### 5. YOLO weight 파일

기본 설정은 아래 weight 이름을 사용합니다.

```yaml
yolo_model: "yolov8n.pt"
```

처음 실행 시 네트워크가 가능하면 자동 다운로드될 수 있지만, 오프라인 환경이라면 팀원 각자가 미리 준비해 두는 것이 안전합니다.

### 6. 권장 설치 순서

```bash
cd ~/ros2_ws/src
git clone <DOOSAN_REPOSITORY_URL> doosan-robot2
git clone <THIS_REPOSITORY_URL> dsr_realsense_pick_place

cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select dsr_realsense_pick_place
source install/setup.bash
```

## 워크스페이스 예시

이 패키지는 아래와 같은 ROS 2 워크스페이스에서 사용한다고 가정합니다.

```bash
ros2_ws/
├── src/
│   ├── doosan-robot2/
│   └── dsr_realsense_pick_place/
├── build/
├── install/
└── log/
```

## 빌드 방법

워크스페이스 루트에서 실행합니다.

```bash
cd ~/ros2_ws
colcon build --packages-select dsr_realsense_pick_place
source install/setup.bash
```

## 실행 방법

기본 launch 에서는 GUI 가 함께 실행됩니다.
RViz 는 `dsr_bringup2_moveit.launch.py` 내부에서 이미 실행되므로 이 launch 에서는 별도로 한 번 더 띄우지 않습니다.

Wayland 환경에서 RViz 또는 GUI 가 안 뜨면 아래처럼 `xcb`를 먼저 지정하고 실행하는 것을 권장합니다.

```bash
export QT_QPA_PLATFORM=xcb
```

### 1. 가상 모드

```bash
cd ~/ros2_ws
source install/setup.bash
export QT_QPA_PLATFORM=xcb
ros2 launch dsr_realsense_pick_place pick_place.launch.py mode:=virtual
```

### 2. 실제 로봇

```bash
cd ~/ros2_ws
source install/setup.bash
export QT_QPA_PLATFORM=xcb
ros2 launch dsr_realsense_pick_place pick_place.launch.py mode:=real host:=192.168.1.100
```

### 3. RealSense 없이 테스트

```bash
cd ~/ros2_ws
source install/setup.bash
export QT_QPA_PLATFORM=xcb
ros2 launch dsr_realsense_pick_place pick_place.launch.py use_realsense:=false
```

### 4. GUI 없이 실행

```bash
cd ~/ros2_ws
source install/setup.bash
export QT_QPA_PLATFORM=xcb
ros2 launch dsr_realsense_pick_place pick_place.launch.py gui:=false
```

## 설정 포인트

### 1. 로봇 모델

- launch 기본값은 `e0509`
- 필요 시 launch 인수 `model:=...` 로 변경 가능

### 2. 카메라 TF

`pick_place.launch.py`의 아래 인수는 예시값입니다.

- `cam_tf_x`
- `cam_tf_y`
- `cam_tf_z`
- `cam_tf_qx`
- `cam_tf_qy`
- `cam_tf_qz`
- `cam_tf_qw`

실제 환경에서는 hand-eye calibration 결과로 교체해야 합니다.

### 3. 카메라 토픽

기본 설정은 `config/pick_place_params.yaml`에 있습니다.

- `color_topic`
- `depth_topic`
- `camera_info_topic`

RealSense launch 설정과 토픽 이름이 다르면 반드시 맞춰야 합니다.

### 4. 그리퍼 IO

그리퍼 제어는 현재 Doosan 컨트롤 박스의 디지털 출력 기준입니다.

- `gripper_open_io`
- `gripper_close_io`
- `gripper_wait_sec`

실제 배선과 IO 번호에 맞게 수정해야 합니다.

### 5. GUI 선택 토픽

- GUI 는 `/selected_object_label` 토픽으로 선택한 물체 라벨을 보냅니다.
- `object_detector`는 그 라벨에 맞는 물체 좌표를 `/selected_object_pose`로 발행합니다.
- `pick_place_node`는 그 좌표만 받아서 pick 동작을 수행합니다.

### 6. 선택 흐름 확인 방법

아래 토픽을 보면 GUI 선택이 실제 pick 동작까지 연결되는지 확인할 수 있습니다.

```bash
ros2 topic echo /selected_object_label
ros2 topic echo /selected_object_pose
ros2 topic echo /pick_place_state
```

확인 순서:

1. GUI 버튼을 누르면 `/selected_object_label` 값이 바뀌는지 확인
2. 선택한 물체가 화면에 보이면 `/selected_object_pose`가 발행되는지 확인
3. `pick_place_state`가 `DETECTING -> PRE_PICK -> PICK ...` 순서로 바뀌는지 확인

## YOLO 사용 관련

- `ultralytics`가 설치되어 있고 모델 파일이 준비되어 있으면 YOLO 검출 사용
- YOLO 모델 로드에 실패하면 빨간 물체를 찾는 간단한 색상 기반 fallback으로 동작

예시:

```yaml
object_detector:
  ros__parameters:
    use_yolo: true
    yolo_model: "yolov8n.pt"
```

로컬에 weight 파일이 없다면 네트워크 환경에 따라 자동 다운로드가 실패할 수 있으니, 팀 저장소에는 weight 파일을 포함하지 않고 각자 준비하는 방식을 권장합니다.

## 팀원 공유 절차

### 1. 저장소 클론

```bash
cd ~/ros2_ws/src
git clone <REPOSITORY_URL>
```

### 2. 의존 패키지 준비

- `doosan-robot2`
- `realsense2_camera`
- 필요한 Python 패키지 설치

### 3. 빌드

```bash
cd ~/ros2_ws
colcon build --packages-select dsr_realsense_pick_place
source install/setup.bash
```

## 권장 공유 규칙

- 실제 장비 IP, 시리얼 번호, 캘리브레이션 값은 README 또는 별도 문서로 관리
- 개인 PC 전용 설정은 코드에 하드코딩하지 않기
- 대용량 weight 파일과 로그 파일은 Git에 올리지 않기
- launch 기본값은 팀 공통 기준으로 유지하고, 개인 테스트 값은 launch 인수로 덮어쓰기

## 협업 문서

- [CONTRIBUTING.md](./CONTRIBUTING.md)
- [LICENSE](./LICENSE)

## 추후 보강하면 좋은 항목

- 캘리브레이션 절차 문서
- 실제 토픽/TF 예시 스크린샷
- 그리퍼 배선 및 IO 맵
- 자주 발생하는 오류와 해결 방법
