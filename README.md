# dsr_realsense_pick_place

Doosan E0509 협동 로봇과 Intel RealSense RGB-D 카메라를 이용해
YOLOv8 객체 인식 기반 Pick & Place를 수행하는 ROS 2 패키지.

---

## 시스템 구성

```
Intel RealSense D4xx
    │  RGB + Depth (aligned)
    ▼
[object_detector]  ─────────────────────────┐
  YOLOv8 검출                                 │
  MAD Depth 필터링                             │  /detected_objects (JSON)
  RealSense deproject                          │  /detection_debug_image
  TF 좌표 변환 (camera → base_link)            │
    │                                          ▼
    │  /selected_object_pose           [gui_node]
    │                                    카메라 영상 표시
    ▼                                    물체 선택 버튼
[pick_place_node]                        상태 표시
  상태머신                                     │
  Doosan 서비스 호출                           │ /selected_object_label
  그리퍼 제어 (IO / RH-P12-Rn)  ◄────────────┘
    │
    ▼
Doosan E0509 + 그리퍼
```

### 노드 역할 요약

| 노드 | 역할 |
|------|------|
| `object_detector` | RealSense에서 동기화된 RGB+Depth 수신 → YOLO 검출 → 3D 좌표 변환 → 발행 |
| `pick_place_node` | 타겟 좌표를 받아 상태머신으로 Pick & Place 수행 |
| `gui_node` | 검출 영상 표시, 물체 선택 버튼, 상태 모니터링 |

---

## 패키지 구조

```
dsr_realsense_pick_place/
├── dsr_realsense_pick_place/
│   ├── object_detector.py    # RGB-D 객체 검출 노드
│   ├── pick_place_node.py    # Pick & Place 상태머신 노드
│   └── gui_node.py           # PyQt5 GUI 노드
├── launch/
│   └── pick_place.launch.py  # 전체 시스템 런치 파일
├── config/
│   └── pick_place_params.yaml # 노드 파라미터 설정
├── requirements.txt
├── package.xml
└── setup.py
```

---

## 상태머신 흐름

`pick_place_node`는 아래 순서로 동작한다.

```
IDLE
 │ 홈 포지션으로 이동
 ▼
DETECTING
 │ /selected_object_pose 수신 + 작업 영역 검증
 ▼
PRE_PICK
 │ 물체 위 안전 높이(pre_pick_z_offset=0.12m)까지 이동 + 그리퍼 열기
 ▼
PICK
 │ 저속(50mm/s)으로 파지 높이(pick_z_offset=0.005m)까지 하강 + 그리퍼 닫기
 ▼
LIFT
 │ 파지 후 PRE_PICK 높이로 상승
 ▼
MOVE_TO_PLACE
 │ Place 위치 상단(place_position + pre_place_z_offset=0.15m)으로 이동
 ▼
PLACE
 │ 저속으로 place_position까지 하강 + 그리퍼 열기
 ▼
POST_PLACE
 │ Place 위 안전 높이로 복귀
 ▼
HOME
 │ 홈 포지션으로 복귀
 ▼
IDLE (다음 사이클)

예외 발생 시 → ERROR (수동 복구 필요)
```

---

## 요구 환경

| 항목 | 버전 / 사양 |
|------|------------|
| OS | Ubuntu 22.04 LTS |
| ROS | ROS 2 Humble |
| Python | 3.10 이상 |
| 로봇 | Doosan E0509 (또는 가상 모드) |
| 카메라 | Intel RealSense D400 시리즈 (선택) |
| 그리퍼 | 디지털 IO, 툴 플랜지 IO, ROBOTIS RH-P12-Rn 중 택1 |

---

## 설치

### 1. ROS 2 기본 패키지

```bash
sudo apt update
sudo apt install -y \
  ros-humble-cv-bridge \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  ros-humble-vision-msgs \
  python3-numpy \
  python3-opencv \
  python3-pyqt5
```

### 2. RealSense ROS 2 패키지

```bash
sudo apt install -y ros-humble-realsense2-camera
```

### 3. Doosan ROS 2 패키지

```bash
cd ~/ros2_ws/src
git clone <DOOSAN_REPOSITORY_URL> doosan-robot2
```

필요한 패키지: `dsr_bringup2`, `dsr_msgs2`, `dsr_controller2`, `dsr_description2`, `dsr_moveit2`

### 4. Python 패키지

```bash
# YOLO (선택 — 없으면 색상 기반 fallback으로 동작)
pip install ultralytics

# 또는 requirements.txt 일괄 설치
pip install -r ~/ros2_ws/src/dsr_realsense_pick_place/requirements.txt
```

### 5. YOLO 가중치 파일

기본 설정은 `yolov8n.pt`를 사용한다. 첫 실행 시 네트워크가 연결되어 있으면 자동으로 다운로드된다.
오프라인 환경에서는 미리 다운로드해 두어야 한다.

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

> **팀 공유 주의**: YOLO 가중치 파일(`.pt`)은 `.gitignore`에 포함되어 있으므로 Git에 올리지 않는다. 각자 준비할 것.

### 6. 빌드

```bash
cd ~/ros2_ws
colcon build --packages-select dsr_realsense_pick_place
source install/setup.bash
```

---

## 실행

Wayland 환경에서 Qt GUI 또는 RViz가 실행되지 않으면 먼저 아래를 설정한다.

```bash
export QT_QPA_PLATFORM=xcb
```

### 가상 모드 (에뮬레이터)

실제 로봇 없이 동작을 확인할 때 사용한다.

```bash
source ~/ros2_ws/install/setup.bash
export QT_QPA_PLATFORM=xcb
ros2 launch dsr_realsense_pick_place pick_place.launch.py mode:=virtual
```

### 실제 로봇

```bash
source ~/ros2_ws/install/setup.bash
export QT_QPA_PLATFORM=xcb
ros2 launch dsr_realsense_pick_place pick_place.launch.py \
  mode:=real \
  host:=192.168.1.100
```

### RealSense 없이 테스트

카메라 없이 노드 연결만 확인할 때 사용한다.

```bash
ros2 launch dsr_realsense_pick_place pick_place.launch.py use_realsense:=false
```

### GUI 없이 실행

헤드리스 환경 또는 자동화 테스트 시 사용한다.

```bash
ros2 launch dsr_realsense_pick_place pick_place.launch.py gui:=false
```

---

## 런치 인수 전체 목록

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `mode` | `virtual` | `virtual` 또는 `real` |
| `host` | `127.0.0.1` | 로봇 IP (real 모드) |
| `port` | `12345` | 로봇 통신 포트 |
| `model` | `e0509` | Doosan 모델명 |
| `color` | `white` | 로봇 색상 |
| `use_realsense` | `true` | RealSense 카메라 노드 실행 여부 |
| `camera_serial` | `` | RealSense 시리얼 번호 (비어 있으면 자동) |
| `cam_tf_x/y/z` | `0.5/0.0/0.6` | 카메라→베이스 TF 위치 (m) |
| `cam_tf_qx/qy/qz/qw` | `0/0.707/0/0.707` | 카메라→베이스 TF 회전 (quaternion) |
| `gui` | `true` | PyQt5 GUI 실행 여부 |

---

## 주요 설정 포인트

### 1. 카메라 TF 캘리브레이션

카메라와 로봇 베이스의 상대 위치를 `pick_place.launch.py`의 `cam_tf_*` 인수로 지정한다.
이 값이 부정확하면 Pick 좌표가 실제 물체 위치와 달라진다.

```bash
# launch 인수로 직접 지정
ros2 launch dsr_realsense_pick_place pick_place.launch.py \
  cam_tf_x:=0.450 cam_tf_y:=0.010 cam_tf_z:=0.620 \
  cam_tf_qx:=0.0 cam_tf_qy:=0.707 cam_tf_qz:=0.0 cam_tf_qw:=0.707
```

정밀 캘리브레이션이 필요하다면 `easy_handeye2` 패키지 사용을 권장한다.

### 2. YOLO 설정

`config/pick_place_params.yaml`에서 조정한다.

```yaml
object_detector:
  ros__parameters:
    use_yolo: true
    yolo_model: "yolov8n.pt"       # n < s < m < l < x (속도↔정확도)
    confidence_threshold: 0.5       # 낮출수록 더 많이 검출 (오검출 증가)
    target_classes: ["bottle", "cup", "bowl", "sports ball", "orange", "apple"]
```

`target_classes`를 빈 리스트(`[]`)로 설정하면 COCO 전체 클래스를 검출한다.

### 3. 그리퍼 타입 선택

```yaml
pick_place_node:
  ros__parameters:
    # 세 가지 중 하나 선택
    gripper_type: "robotis_rh_p12_rn"  # ROBOTIS RH-P12-Rn (기본)
    # gripper_type: "digital_io"        # 컨트롤 박스 디지털 출력
    # gripper_type: "tool_digital"      # 툴 플랜지 디지털 출력
```

**RH-P12-Rn** (Modbus RTU):
```yaml
    rh12_open_stroke: 700    # 0~700 범위, 700=완전 개방
    rh12_close_stroke: 0     # 0=완전 폐쇄
    rh12_goal_current: 400   # 파지력 (너무 높으면 과전류, 너무 낮으면 파지 실패)
```

**digital_io** / **tool_digital**:
```yaml
    gripper_open_io: 1       # Open 신호 포트 번호
    gripper_close_io: 2      # Close 신호 포트 번호
    gripper_wait_sec: 0.8    # 동작 완료 대기 시간 (s)
```

### 4. 작업 공간 설정

로봇 베이스 좌표계 기준으로 유효 작업 영역을 지정한다.
이 범위 밖의 검출 결과는 자동으로 무시된다.

```yaml
    workspace_x_min: 0.15    # 전방 최소 거리 (m)
    workspace_x_max: 0.80    # 전방 최대 거리
    workspace_y_min: -0.60   # 좌우 범위
    workspace_y_max: 0.60
    workspace_z_min: 0.0     # 높이 범위
    workspace_z_max: 0.60
```

### 5. Place 위치 설정

```yaml
    place_position: [0.4, -0.3, 0.1]  # [x, y, z] (m), 로봇 베이스 기준
    pre_place_z_offset: 0.15           # Place 위 접근 높이 (m)
    place_rpy: [0.0, 180.0, 0.0]       # 툴 방향 (deg) - [0, 180, 0] = 수직 하강
```

### 6. 카메라 토픽

RealSense 드라이버 버전에 따라 토픽 이름이 다를 수 있다.

```yaml
object_detector:
  ros__parameters:
    color_topic: "/camera/camera/color/image_raw"
    depth_topic: "/camera/camera/aligned_depth_to_color/image_raw"
    camera_info_topic: "/camera/camera/color/camera_info"
```

실제 발행 중인 토픽을 확인하려면:

```bash
ros2 topic list | grep camera
```

---

## GUI 사용법

1. 런치 후 GUI 창이 열리면 왼쪽 카메라 영상에서 검출된 물체에 초록색 bbox가 표시된다.
2. 오른쪽 버튼 패널에서 집을 물체를 클릭하면 해당 라벨로 Pick 동작이 수행된다.
3. **자동 선택 사용** 버튼을 누르면 가장 가까운 물체를 자동으로 선택한다.
4. 상태 패널에서 현재 Pick & Place 진행 상황을 확인할 수 있다.

---

## 토픽/서비스 목록

### 발행 토픽

| 토픽 | 타입 | 발행 노드 | 설명 |
|------|------|----------|------|
| `/detected_object_pose` | `geometry_msgs/PoseStamped` | object_detector | 최종 선택 물체 좌표 |
| `/selected_object_pose` | `geometry_msgs/PoseStamped` | object_detector | pick_place_node 타겟 좌표 |
| `/detected_objects` | `std_msgs/String` | object_detector | 검출 물체 JSON 목록 |
| `/detection_debug_image` | `sensor_msgs/Image` | object_detector | bbox 오버레이 디버그 이미지 |
| `/pick_place_state` | `std_msgs/String` | pick_place_node | 현재 상태머신 상태 |
| `/selected_object_label` | `std_msgs/String` | gui_node | GUI 선택 라벨 |

### 구독 토픽

| 토픽 | 발행 소스 | 구독 노드 |
|------|----------|----------|
| `/camera/camera/color/image_raw` | RealSense | object_detector |
| `/camera/camera/aligned_depth_to_color/image_raw` | RealSense | object_detector |
| `/camera/camera/color/camera_info` | RealSense | object_detector |
| `/selected_object_label` | gui_node | object_detector |
| `/selected_object_pose` | object_detector | pick_place_node |
| `/detection_debug_image` | object_detector | gui_node |
| `/detected_objects` | object_detector | gui_node |
| `/pick_place_state` | pick_place_node | gui_node |

---

## 상태 확인 및 디버그

### 동작 흐름 모니터링

```bash
# GUI 선택 확인
ros2 topic echo /selected_object_label

# Pick 타겟 좌표 확인 (로봇 베이스 기준, m 단위)
ros2 topic echo /selected_object_pose

# 상태머신 진행 확인
ros2 topic echo /pick_place_state

# 검출 물체 전체 목록 확인 (JSON)
ros2 topic echo /detected_objects
```

### 정상 동작 시 상태 순서

```
/pick_place_state 값이 아래 순서로 변해야 한다:
IDLE → DETECTING → PRE_PICK → PICK → LIFT → MOVE_TO_PLACE → PLACE → POST_PLACE → HOME → IDLE
```

### 디버그 이미지 확인

```bash
ros2 run rqt_image_view rqt_image_view /detection_debug_image
```

### TF 확인

```bash
# TF 트리 전체 보기
ros2 run tf2_tools view_frames

# 카메라 → 베이스 변환 실시간 확인
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
```

---

## 자주 발생하는 문제

### GUI / RViz가 열리지 않는다

Wayland 환경에서 발생한다.

```bash
export QT_QPA_PLATFORM=xcb
```

### "Cannot load platform plugin 'xcb'" 오류

OpenCV와 PyQt5의 Qt 플러그인 경로 충돌. `gui_node.py`에서 환경변수를 강제 설정하므로
보통 자동으로 해결된다. 문제가 지속되면:

```bash
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins
```

### TF 변환 실패 (pick 좌표가 엉뚱함)

```
TF 변환 실패: ...
```

`cam_tf_*` 값이 실제 카메라 위치와 다를 때 발생한다. TF를 확인하고 launch 인수를 수정한다.

```bash
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
```

### YOLO 모델을 찾을 수 없다

오프라인 환경에서 자동 다운로드가 실패한 경우다.

```bash
# 온라인 환경에서 미리 다운로드
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### RH-P12-Rn 그리퍼가 움직이지 않는다

```yaml
# pick_place_params.yaml에서 확인
rh12_allow_missing_service: true  # 서비스 없을 때 토픽으로 대체 발행
rh12_slave_id: 1                  # 그리퍼 Modbus slave ID 확인
rh12_open_stroke: 700             # 실제 그리퍼 최대 스트로크 확인
```

`/gripper/rh12_stroke_cmd` 토픽으로 명령이 발행되는지 확인:

```bash
ros2 topic echo /gripper/rh12_stroke_cmd
```

---

## 협업 규칙

- 실제 장비 IP, 시리얼 번호, 캘리브레이션 값은 launch 인수로 전달하고 코드에 하드코딩하지 않는다.
- YOLO 가중치(`.pt`)와 로그 파일은 Git에 올리지 않는다 (`.gitignore` 적용됨).
- 개인 PC 전용 설정은 launch 인수 오버라이드로 적용하고 `pick_place_params.yaml`은 팀 공통 기준으로 유지한다.

자세한 협업 가이드는 [CONTRIBUTING.md](./CONTRIBUTING.md)를 참고한다.

---

## 추후 보강 항목

- 실제 hand-eye 캘리브레이션 절차 문서
- 그리퍼 배선 및 IO 맵 다이어그램
- 실제 환경 토픽/TF 예시 스크린샷
- 자주 발생하는 오류 사례 추가
