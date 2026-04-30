# dsr_realsense_pick_place

Doosan E0509 협동로봇과 Intel RealSense RGB-D 카메라를 이용해  
YOLOv8 객체 인식 기반 Pick & Place를 수행하는 ROS 2 패키지.

---

## 시스템 구성

```
Intel RealSense D4xx
    │  RGB + Depth (aligned)
    ▼
[object_detector]  ─────────────────────────┐
  YOLOv8 검출                                │
  MAD Depth 필터링                            │  /detected_objects (JSON)
  RealSense deproject                         │  /detection_debug_image
  수동 원점 좌표 변환 (camera → base_link)      │
    │                                         ▼
    │  /selected_object_pose           [gui_node]
    │                                    카메라 영상 표시
    ▼                                    물체 선택 버튼
[pick_place_node]                        상태 표시 / 시스템 상태 바
  상태머신 (INITIALIZING → IDLE → DETECTING → ...)  │
  Doosan 서비스 호출                                 │ /selected_object_label
  gripper_node 경유 그리퍼 제어              ◄────────┘
    │
    ▼
[gripper_node]
  DRL 스크립트 기반 Modbus RTU
    │
    ▼
Doosan E0509 + RH-P12-RN(A) 그리퍼
```

### 노드 역할

| 노드 | 역할 |
|---|---|
| `object_detector` | RGB+Depth 동기화 수신 → YOLO 검출 → 3D 좌표 변환 → 발행 |
| `pick_place_node` | 타겟 좌표 수신 → 상태머신으로 Pick & Place 수행 |
| `gui_node` | 검출 영상 표시, 물체 선택 버튼, 상태 모니터링, 시스템 상태 바 |
| `gripper_node` | DRL 스크립트로 RS-485 Modbus RTU 그리퍼 제어 |

---

## 패키지 구조

```
mini_project/
├── dsr_realsense_pick_place/
│   ├── pick_place_node.py    # 상태머신 핵심 노드
│   ├── object_detector.py    # RGB-D 객체 검출 노드
│   ├── gui_node.py           # PyQt5 GUI 노드
│   ├── gripper_node.py       # RH-P12-RN 그리퍼 드라이버
│   └── __init__.py
├── launch/
│   └── pick_place.launch.py  # 전체 시스템 런치
├── config/
│   └── pick_place_params.yaml
├── package.xml
└── setup.py
```

---

## 상태머신 흐름

```
INITIALIZING  (robot_mode AUTO + 서보 ON 자동 시도)
     ↓
IDLE  (대기)
     ↓  run_once
HOME  (홈 이동)
     ↓
DETECTING  (/selected_object_pose 대기, 30초 타임아웃)
     ↓
PRE_PICK  (물체 위 안전 높이 이동 + 그리퍼 열기)
     ↓
PICK  (저속 하강 + 그리퍼 닫기)
     ↓
LIFT  (안전 높이 상승)
     ↓
MOVE_TO_PLACE  (place 위치 상단 이동)
     ↓
PLACE  (저속 하강 + 그리퍼 열기)
     ↓
POST_PLACE  (안전 높이 복귀)
     ↓
HOME  →  IDLE  (다음 사이클)

예외 → ERROR  (수동 복구 필요)
E-STOP → EMERGENCY_STOP  (e_stop_reset 서비스로만 해제)
BACKDRIVE  (중력보상 역구동 — safety_backdrive 서비스)
```

---

## 요구 환경

| 항목 | 버전 / 사양 |
|---|---|
| OS | Ubuntu 22.04 LTS |
| ROS | ROS 2 Humble |
| Python | 3.10 이상 |
| 로봇 | Doosan E0509 (또는 가상 모드) |
| 카메라 | Intel RealSense D400 시리즈 (선택) |
| 그리퍼 | ROBOTIS RH-P12-RN(A) (Modbus RTU over RS-485) |

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

필요 패키지: `dsr_bringup2`, `dsr_msgs2`, `dsr_controller2`, `dsr_description2`

### 4. Python 패키지

```bash
# YOLO (선택 — 없으면 색상 기반 fallback으로 동작)
pip install ultralytics
```

### 5. YOLO 가중치 파일

프로젝트 학습 가중치(`best.pt`)를 아래 경로에 배치한다.

```
src/mini_project/runs/weights/weights/best.pt
```

또는 `config/pick_place_params.yaml`의 `yolo_model` 값을 실제 경로로 수정한다.

> **주의**: `.pt` 파일은 `.gitignore`에 포함 — Git에 올리지 않는다.

### 6. 빌드

```bash
cd ~/kairos_ws
colcon build --symlink-install
source install/setup.bash
```

---

## 실행

Wayland 환경에서 Qt GUI가 실행되지 않으면 먼저 설정한다.

```bash
export QT_QPA_PLATFORM=xcb
```

### 가상 모드 (에뮬레이터)

```bash
source ~/kairos_ws/install/setup.bash
export QT_QPA_PLATFORM=xcb
ros2 launch dsr_realsense_pick_place pick_place.launch.py mode:=virtual
```

### 실제 로봇

```bash
source ~/kairos_ws/install/setup.bash
export QT_QPA_PLATFORM=xcb
ros2 launch dsr_realsense_pick_place pick_place.launch.py \
  mode:=real \
  host:=192.168.1.100 \
  use_realsense:=true
```

### RealSense 없이 테스트

```bash
ros2 launch dsr_realsense_pick_place pick_place.launch.py use_realsense:=false
```

### launch TimerAction으로 robot_mode 강제 설정 (디버그용)

```bash
ros2 launch dsr_realsense_pick_place pick_place.launch.py \
  use_launch_set_robot_mode:=true
```

> 기본값은 `false`. `pick_place_node` 내부 INITIALIZING 상태에서 자동 처리.

---

## 런치 인수 전체 목록

| 인수 | 기본값 | 설명 |
|---|---|---|
| `mode` | `virtual` | `virtual` 또는 `real` |
| `host` | `127.0.0.1` | 로봇 IP (real 모드) |
| `port` | `12345` | 로봇 통신 포트 |
| `model` | `e0509` | Doosan 모델명 |
| `use_realsense` | `true` | RealSense 카메라 노드 실행 여부 |
| `camera_serial` | `` | RealSense 시리얼 번호 (비어 있으면 자동) |
| `cam_tf_x/y/z` | `0.5/0.0/0.6` | 카메라→베이스 TF 위치 (m) |
| `cam_tf_qx/qy/qz/qw` | `0/0.707/0/0.707` | 카메라→베이스 TF 회전 |
| `gui` | `true` | PyQt5 GUI 실행 여부 |
| `use_launch_set_robot_mode` | `false` | launch에서 robot_mode 서비스 직접 호출 여부 |

---

## 주요 설정

### YOLO 설정 (`config/pick_place_params.yaml`)

```yaml
object_detector:
  ros__parameters:
    use_yolo: true
    yolo_model: "runs/weights/weights/best.pt"   # 실제 경로로 수정
    confidence_threshold: 0.5
    target_classes: ["doll", "pack", "pencil", "tape", "cup"]
```

### 작업 공간 설정

```yaml
pick_place_node:
  ros__parameters:
    workspace_x_min: 0.15
    workspace_x_max: 0.80
    workspace_y_min: -0.60
    workspace_y_max:  0.60
    workspace_z_min:  0.0
    workspace_z_max:  0.60
```

### Place 위치 설정

```yaml
    place_position: [0.4, -0.3, 0.1]   # [x, y, z] (m), 로봇 베이스 기준
    pre_place_z_offset: 0.15
    place_rpy: [0.0, 180.0, 0.0]        # 수직 하강
```

---

## GUI 사용법

1. 런치 후 GUI 창이 열리면 상단 상태 바(`CAM/DET/PICK/GRIP/HW/SPD`)로 시스템 상태를 확인한다.
2. 왼쪽 카메라 영상에서 검출된 물체에 초록색 bbox가 표시된다.
3. 오른쪽 버튼 패널에서 집을 물체를 클릭하면 해당 라벨로 Pick 동작이 수행된다.
4. **긴급 정지(E-STOP)** 버튼: 즉시 모션 중단 + 선택 라벨 해제.
5. **태스크 중단** 버튼: 현재 모션 후 그리퍼 열고 HOME 복귀 + 선택 라벨 해제.
6. **긴급정지 해제** 버튼: "⏳ 리셋 중..." 표시 후 하드웨어 알람 리셋 → IDLE 복귀.

---

## 토픽 / 서비스 목록

### 발행 토픽

| 토픽 | 타입 | 발행 노드 |
|---|---|---|
| `/selected_object_pose` | `PoseStamped` | object_detector |
| `/detected_objects` | `String` (JSON) | object_detector |
| `/detection_debug_image` | `Image` | object_detector |
| `/pick_place_state` | `String` | pick_place_node |
| `/robot_hw_state` | `Int32` | pick_place_node |
| `/robot_speed_mode` | `Int32` | pick_place_node |
| `/system/heartbeat` | `String` | pick_place_node |
| `/selected_object_label` | `String` | gui_node |

### pick_place_node 서비스 (수신)

| 서비스 | 설명 |
|---|---|
| `/pick_place/run_once` | 1사이클 Pick & Place 실행 |
| `/pick_place/go_home` | 홈 이동 |
| `/pick_place/e_stop` | 긴급정지 |
| `/pick_place/cancel` | 태스크 취소 (그리퍼 열고 홈 복귀) |
| `/pick_place/e_stop_reset` | 긴급정지 해제 + 알람 리셋 |
| `/pick_place/speed_normal` | 정상 속도 |
| `/pick_place/speed_reduced` | 감속 모드 |
| `/pick_place/servo_off` | 서보 OFF |
| `/pick_place/servo_on` | 서보 ON |
| `/pick_place/safety_normal` | 정상 운전 복귀 |
| `/pick_place/safety_backdrive` | 역구동(중력보상) 모드 진입 |

---

## 상태 확인 및 디버그

```bash
# 상태머신 진행 확인
ros2 topic echo /pick_place_state

# Pick 타겟 좌표 확인
ros2 topic echo /selected_object_pose

# 검출 물체 전체 목록 (JSON)
ros2 topic echo /detected_objects

# 디버그 이미지 확인
ros2 run rqt_image_view rqt_image_view /detection_debug_image
```

---

## 캘리브레이션 순서

1. 알려진 위치(base 기준 `x=0.4, y=0.0`)에 물체를 놓는다.
2. `/detected_objects` 토픽에서 검출 좌표를 확인한다.
3. `오차 = 실제 위치 - 검출 위치` → `absolute_calib_*_mm`에 반영한다.
4. `pick_z_offset` 조정으로 집는 높이를 미세 조정한다.
5. `place_position` 조정으로 내려놓을 위치를 설정한다.

---

## 자주 발생하는 문제

### GUI가 열리지 않는다 (Wayland)
```bash
export QT_QPA_PLATFORM=xcb
```

### YOLO 모델을 찾을 수 없다
`config/pick_place_params.yaml`의 `yolo_model` 경로를 실제 `best.pt` 위치로 수정한다.

### gripper_node가 늦게 떠서 pick_place_node가 경고를 출력한다
정상 동작이다. `_wait_for_services()`가 bounded wait 후 계속 진행하고, 실제 gripper 호출 시점에 연결 여부를 재확인한다.

### 로봇이 MANUAL 모드로 남아 있다
`pick_place_node`의 INITIALIZING 상태에서 robot_mode 설정이 실패한 경우다.
수동으로 설정하려면:
```bash
ros2 launch dsr_realsense_pick_place pick_place.launch.py use_launch_set_robot_mode:=true
```

---

## 협업 규칙

- 실제 장비 IP, 시리얼 번호, 캘리브레이션 값은 launch 인수로 전달, 코드에 하드코딩 금지.
- YOLO 가중치(`.pt`)와 로그 파일은 Git에 올리지 않는다 (`.gitignore` 적용됨).
- 개인 PC 전용 설정은 launch 인수 오버라이드로 적용하고 `pick_place_params.yaml`은 팀 공통 기준으로 유지한다.

자세한 협업 가이드는 [CONTRIBUTING.md](./CONTRIBUTING.md)를 참고한다.

---

## 관련 문서

| 문서 | 위치 | 내용 |
|---|---|---|
| 코드베이스 상세 분석 | `kairos_ws/analysis.md` | 노드별 설계, 패치 이력, 미해결 항목 |
| 수정 계획 | `kairos_ws/plan.md` | 우선순위별 문제점 및 수정 방향 |
| 확장성 개선 계획 | `kairos_ws/robot_arm_refactor.md` | RobotArm 레이어 분리 단계별 계획 |
