"""
gui_node.py
-----------
PyQt5 기반 Pick & Place GUI 노드.

화면 구성:
  좌측: 카메라 디버그 영상 (640×480, bbox + depth 정보 오버레이)
  우측 상단: 상태 패널 (Pick & Place 상태, 선택 물체, 선택 모드)
  우측 하단: 물체 선택 패널 (자동 선택 버튼 + 검출 물체 버튼 그리드 + 요약)

동작 흐름:
  1. /detection_debug_image  → 카메라 영상 표시
  2. /detected_objects       → JSON 파싱 후 물체 버튼 갱신
  3. /pick_place_state       → 현재 Pick & Place 상태 표시
  4. 사용자가 버튼 클릭      → /selected_object_label 발행
  5. object_detector가 라벨에 맞는 물체 선택 후 /selected_object_pose 발행
  6. pick_place_node가 pick 동작 수행

Qt-ROS 이벤트 루프 통합:
  QApplication.exec_()이 Qt 이벤트를 처리하는 메인 루프를 실행한다.
  ROS 콜백은 QTimer(10ms 간격)가 rclpy.spin_once()를 호출해 처리한다.
  UI 갱신은 별도 QTimer(100ms 간격)가 _update_ui()를 호출해 수행한다.
  이 방식으로 Qt 이벤트와 ROS 메시지가 단일 스레드에서 안전하게 공존한다.

구독:
  /detection_debug_image  (sensor_msgs/Image)  - bbox가 그려진 디버그 영상
  /detected_objects       (std_msgs/String)    - 검출 물체 JSON 목록
  /pick_place_state       (std_msgs/String)    - 현재 상태머신 상태

발행:
  /selected_object_label  (std_msgs/String)    - 사용자가 선택한 물체 라벨
                                                 빈 문자열 = 자동 선택 모드
"""

import os
import json
import re
import yaml
import subprocess
import sys
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
try:
    from cv_bridge import CvBridge
    _CV_BRIDGE_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - runtime env dependent
    CvBridge = None
    _CV_BRIDGE_IMPORT_ERROR = e

import cv2
# OpenCV 패키지가 cv2/qt/plugins 경로를 잡아 버리면
# PyQt5와 Qt 런타임 버전이 엇갈려 xcb 플러그인 로딩이 깨질 수 있다.
for key in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_PLUGIN_PATH'):
    value = os.environ.get(key, '')
    if 'cv2/qt/plugins' in value:
        os.environ.pop(key, None)

# GNOME Wayland 환경에서는 Qt가 xcb/wayland 사이에서 흔들릴 수 있어
# 별도 설정이 없으면 XWayland(xcb)로 고정한다.
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QSlider,
    QSpinBox,
    QFrame,
)
from PyQt5.QtGui import QPalette, QFont
from PyQt5.QtCore import QLibraryInfo
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from rcl_interfaces.msg import Parameter as RclParameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from std_msgs.msg import Int32, String

from std_srvs.srv import SetBool, Trigger
from dsr_gripper_tcp_interfaces.msg import GripperState

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = QLibraryInfo.location(QLibraryInfo.PluginsPath)


class RealTimeGraphWidget(QWidget):
    """실시간 그리퍼 전류를 롤링 플롯 형태로 시각화하는 커스텀 위젯."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = []
        self.max_len = 100
        self.max_value = 100.0
        self.setMinimumHeight(100)

    def add_data(self, value):
        self.data.append(float(value))
        if len(self.data) > self.max_len:
            self.data.pop(0)
        # 0 중심 양방향 표시 — |max| 기준 대칭 스케일
        abs_max = max(abs(v) for v in self.data) if self.data else 0.0
        self.max_value = max(100.0, abs_max * 1.2)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        mid_y = height / 2.0

        # 다크 테마 배경 칠하기
        painter.fillRect(0, 0, width, height, QColor(30, 30, 30))

        # 회색 격자선 그리기 (0 중심 기준 ±25%, ±50%, ±75%)
        grid_pen = QPen(QColor(60, 60, 60), 1, Qt.DashLine)
        painter.setPen(grid_pen)
        for frac in (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75):
            y = int(mid_y - frac * mid_y)
            painter.drawLine(0, y, width, y)
        for i in range(1, 10):
            x = int(width * i / 10)
            painter.drawLine(x, 0, x, height)

        # 0 기준선 — 격자보다 진하게
        zero_pen = QPen(QColor(120, 120, 120), 1, Qt.SolidLine)
        painter.setPen(zero_pen)
        painter.drawLine(0, int(mid_y), width, int(mid_y))

        # 실시간 플롯 라인 (값 부호+크기 기반: +는 붉은 계열, -는 파란 계열, 0 근처는 회색)
        if len(self.data) >= 2:
            x_step = width / (self.max_len - 1)
            for i in range(len(self.data) - 1):
                val = self.data[i+1]
                t = val / self.max_value
                t = max(-1.0, min(1.0, t))
                mag = abs(t)
                if t >= 0:
                    # 회색(180,180,180) → 붉은색(255,60,60)
                    r = int(180 + (255 - 180) * mag)
                    g = int(180 + (60  - 180) * mag)
                    b = int(180 + (60  - 180) * mag)
                else:
                    # 회색(180,180,180) → 파란색(60,120,255)
                    r = int(180 + (60  - 180) * mag)
                    g = int(180 + (120 - 180) * mag)
                    b = int(180 + (255 - 180) * mag)

                color = QColor(r, g, b)
                line_pen = QPen(color, 2, Qt.SolidLine)
                painter.setPen(line_pen)

                x1 = i * x_step
                y1 = mid_y - (self.data[i]   / self.max_value) * mid_y
                x2 = (i + 1) * x_step
                y2 = mid_y - (self.data[i+1] / self.max_value) * mid_y

                y1 = max(0.0, min(float(height), y1))
                y2 = max(0.0, min(float(height), y2))

                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # 우상단에 텍스트 표시 (부호 같이 표시)
        text_pen = QPen(QColor(255, 255, 255))
        painter.setPen(text_pen)
        painter.setFont(QFont('Arial', 9, QFont.Bold))
        if self.data:
            curr_val = self.data[-1]
            painter.drawText(10, 20, f'Current: {curr_val:+.0f} mA')


class PickPlaceGuiNode(Node):
    def __init__(self):
        super().__init__('pick_place_gui')

        # ROS 디버그 토픽 구독 모드 / 로컬 YOLO 모드 선택
        # 기본값은 false: object_detector가 이미 RealSense를 사용하므로
        # GUI가 카메라를 다시 열지 않게 한다.
        self.declare_parameter('use_local_yolo', False)
        self.declare_parameter('weights_path', '')
        self.declare_parameter('require_best_pt', True)
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('fov_h_deg', 60.0)
        self.declare_parameter('default_object_height_m', 0.12)
        self.declare_parameter('use_realsense', True)
        self.declare_parameter('rs_serial', '')
        self.declare_parameter('rs_width', 640)
        self.declare_parameter('rs_height', 480)
        self.declare_parameter('rs_fps', 30)
        self.declare_parameter('origin_x', -0.80)
        self.declare_parameter('origin_y', 0.0)
        self.declare_parameter('origin_z', -0.96)
        self.declare_parameter('calib_dx_mm', -20.0)
        self.declare_parameter('calib_dy_mm', -20.0)
        self.declare_parameter('calib_dz_mm', 140.0)

        # 도달 가능 영역 필터 — 이 범위 밖 검출 물체는 GUI에서 버튼·요약 자체를 숨긴다.
        # 사용자가 NOT REACHABLE 좌표를 클릭해 ERROR로 가는 것을 원천 차단.
        # 기본값은 pick_place_node의 workspace_* 와 동일하게 두고, yaml로 좁힐 수 있음.
        # reach_radius_max는 박스 한계와 별개로 sqrt(x^2+y^2)에 적용 — E0509 실제 운동학 한계 반영.
        self.declare_parameter('workspace_x_min', 0.15)
        self.declare_parameter('workspace_x_max', 0.80)
        self.declare_parameter('workspace_y_min', -0.60)
        self.declare_parameter('workspace_y_max', 0.60)
        self.declare_parameter('workspace_z_min', 0.0)
        self.declare_parameter('workspace_z_max', 0.60)
        self.declare_parameter('reach_radius_max', 0.65)
        self.workspace_x_min = float(self.get_parameter('workspace_x_min').value)
        self.workspace_x_max = float(self.get_parameter('workspace_x_max').value)
        self.workspace_y_min = float(self.get_parameter('workspace_y_min').value)
        self.workspace_y_max = float(self.get_parameter('workspace_y_max').value)
        self.workspace_z_min = float(self.get_parameter('workspace_z_min').value)
        self.workspace_z_max = float(self.get_parameter('workspace_z_max').value)
        self.reach_radius_max = float(self.get_parameter('reach_radius_max').value)

        # ROS 토픽으로 받은 영상/검출 결과를 Qt 위젯에서 바로 쓸 수 있게
        # 화면 표시용 상태를 멤버 변수로 유지한다.
        self.use_local_yolo = bool(self.get_parameter('use_local_yolo').value)
        self.bridge = CvBridge() if CvBridge is not None else None
        self.latest_qimage = None
        self.detected_objects = []
        self._last_nonempty_objects = []
        self._last_nonempty_objects_time = 0.0
        self.selected_label = ''
        self.pick_place_state = 'IDLE'
        self.last_error_text = ''
        self._latest_raw_detections = []
        self.last_image_time = 0.0
        self.last_objects_time = 0.0
        self.last_state_time = 0.0
        self.last_hw_state_time = 0.0
        self.last_speed_mode_time = 0.0
        self.system_status_items = []
        self._last_system_status_check = 0.0

        # 실시간 그리퍼 상태 캐시
        self.gripper_present_position = 0.0
        self.gripper_present_current = 0.0

        # GUI는 직접 로봇을 움직이지 않고 "어떤 물체를 집을지"만 알린다.
        self.pub_selected = self.create_publisher(String, '/selected_object_label', 10)

        self.cli_run_once      = self.create_client(Trigger, '/pick_place/run_once')
        self.cli_go_home       = self.create_client(Trigger, '/pick_place/go_home')
        self.cli_gripper_open  = self.create_client(Trigger, '/gripper/open')
        self.cli_gripper_close = self.create_client(Trigger, '/gripper/close')
        # 그리퍼 런타임 리셋(재초기화) 및 토크 on/off
        self.cli_gripper_reinit = self.create_client(Trigger, '/gripper_service/reinitialize')
        self.cli_gripper_enable = self.create_client(SetBool, '/gripper/enable')
        self.cli_recover_to_home = self.create_client(Trigger, '/pick_place/recover_to_home')
        # 로봇 이동 없이 ERROR 상태만 해제 (알람 리셋 + 그리퍼 reinit) — recover_to_home의 가벼운 변형.
        self.cli_clear_error     = self.create_client(Trigger, '/pick_place/clear_error')
        self.cli_e_stop        = self.create_client(Trigger, '/pick_place/e_stop')
        self.cli_cancel        = self.create_client(Trigger, '/pick_place/cancel')
        self.cli_e_stop_reset  = self.create_client(Trigger, '/pick_place/e_stop_reset')
        self.cli_speed_normal     = self.create_client(Trigger, '/pick_place/speed_normal')
        self.cli_speed_reduced    = self.create_client(Trigger, '/pick_place/speed_reduced')
        self.cli_servo_off        = self.create_client(Trigger, '/pick_place/servo_off')
        self.cli_servo_on         = self.create_client(Trigger, '/pick_place/servo_on')
        self.cli_safety_normal    = self.create_client(Trigger, '/pick_place/safety_normal')
        self.cli_safety_backdrive = self.create_client(Trigger, '/pick_place/safety_backdrive')
        
        self.cli_object_get_parameters = self.create_client(GetParameters, '/object_detector/get_parameters')
        self.cli_object_set_parameters = self.create_client(SetParameters, '/object_detector/set_parameters')
        
        self.cli_gripper_get_parameters = self.create_client(GetParameters, '/rh_p12_rna_gripper/get_parameters')
        self.cli_gripper_set_parameters = self.create_client(SetParameters, '/rh_p12_rna_gripper/set_parameters')

        # 물체별 파지 강도(grip_*)는 pick_place_node 파라미터로 라이브 적용한다.
        self.cli_pickplace_set_parameters = self.create_client(SetParameters, '/pick_place_node/set_parameters')
        # RealSense 카메라(노출 등) 파라미터 라이브 변경 — 운전 탭의 노출 슬라이더가 호출.
        self.cli_camera_set_parameters = self.create_client(SetParameters, '/camera/camera/set_parameters')

        # 로봇 하드웨어 상태 / 속도 모드 (pick_place_node 폴링 결과 수신)
        self.hw_state   = -1   # -1 = unknown
        self.speed_mode = 0    # 0 = NORMAL
        self.create_subscription(Int32, '/robot_hw_state',  self._cb_hw_state, 10)
        self.create_subscription(Int32, '/robot_speed_mode', self._cb_speed_mode, 10)

        # 그리퍼 서비스 ready 상태 (INITIALIZE 완료 여부)
        self.gripper_hw_ready = False
        self.create_subscription(
            GripperState, '/gripper_service/state', self._cb_gripper_service_state, 10)

        # 실시간 그리퍼 상태 수신 — /gripper/state(raw 0~1000)만 구독한다.
        # /gripper_service/joint_state는 정규화 스케일(예: 0.327)이라 둘 다 받으면 위치/전류가
        # raw↔정규화로 깜빡인다(0↔값). raw 토픽 하나만 쓴다.
        self.create_subscription(JointState, '/gripper/state', self._cb_gripper_joint_state, 10)
        # 그리퍼 INIT/REINIT 진행 상황 — 시간이 카운트되면 진행 중, 멈춰있으면 막힘.
        self.gripper_init_progress = ''       # 최근 메시지
        self.gripper_init_progress_t = 0.0    # 마지막 수신 시각 (GUI에서 stale 판정)
        self.create_subscription(String, '/gripper_service/init_progress',
                                  self._cb_gripper_init_progress, 10)

        if self.use_local_yolo:
            self._init_local_yolo()
        else:
            if self.bridge is None:
                raise RuntimeError(
                    f'cv_bridge import 실패: {_CV_BRIDGE_IMPORT_ERROR}. '
                    'use_local_yolo=true 로 실행하거나 ROS python 환경을 정리하세요.'
                )
            self.create_subscription(Image, '/detection_debug_image', self._cb_image, qos_profile_sensor_data)
            self.create_subscription(String, '/detected_objects', self._cb_objects, 10)
        self.create_subscription(String, '/pick_place_state', self._cb_state, 10)
        self.create_subscription(String, '/pick_place_error', self._cb_error, 10)

    def _cb_gripper_joint_state(self, msg: JointState):
        target_name = None
        for name in msg.name:
            if 'gripper_joint' in name or 'rh_p12_rn' in name:
                target_name = name
                break
        if target_name is not None:
            idx = msg.name.index(target_name)
            self.gripper_present_position = msg.position[idx]
            self.gripper_present_current = msg.effort[idx]

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def _candidate_search_roots(self) -> list[Path]:
        roots: list[Path] = []
        seen: set[Path] = set()

        def _add(path: Path):
            resolved = path.resolve()
            if resolved not in seen and resolved.exists():
                seen.add(resolved)
                roots.append(resolved)

        _add(self._repo_root())
        _add(Path.cwd())
        _add(Path.cwd() / 'mini_project')

        for parent in Path(__file__).resolve().parents:
            _add(parent)
            _add(parent / 'src')
            _add(parent / 'src' / 'mini_project')

        return roots

    def _resolve_weights_path(self, weights: str) -> Path:
        configured = Path(weights).expanduser()
        if configured.is_absolute():
            return configured.resolve()

        candidates = [root / configured for root in self._candidate_search_roots()]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        matches: list[Path] = []
        for root in self._candidate_search_roots():
            matches.extend(p for p in root.rglob(configured.name) if p.is_file())

        if matches:
            suffix = configured.as_posix()
            for match in matches:
                if match.as_posix().endswith(suffix):
                    return match.resolve()
            return matches[0].resolve()

        return (self._repo_root() / configured).resolve()

    def _find_best_pt(self, search_under: Path) -> Path | None:
        cands = list(search_under.rglob('best.pt'))
        if not cands:
            return None
        return max(cands, key=lambda p: p.stat().st_mtime)

    def _init_local_yolo(self):
        from ultralytics import YOLO

        weights = str(self.get_parameter('weights_path').value).strip()
        require_best_pt = bool(self.get_parameter('require_best_pt').value)
        if weights:
            self.weights_path = self._resolve_weights_path(weights)
        else:
            # yolo_live_cam_3d_metrics.py 와 동일하게 runs 아래 최신 best.pt를 기본 사용
            found = None
            for root in self._candidate_search_roots():
                found = self._find_best_pt(root / 'runs')
                if found is not None:
                    break
            if found is None and require_best_pt:
                raise RuntimeError(
                    'runs 아래에서 best.pt를 찾지 못했습니다. '
                    'weights_path 파라미터에 best.pt 경로를 지정하세요.'
                )
            self.weights_path = found if found is not None else Path('yolov8n.pt')

        if require_best_pt and self.weights_path.name != 'best.pt':
            raise RuntimeError(
                f'require_best_pt=true 인데 모델이 best.pt가 아닙니다: {self.weights_path}'
            )
        if not self.weights_path.is_file() and self.weights_path.name == 'best.pt':
            raise RuntimeError(f'best.pt 파일이 없습니다: {self.weights_path}')

        self.model = YOLO(str(self.weights_path))
        self.model_names = (
            self.model.names
            if isinstance(self.model.names, dict)
            else dict(enumerate(self.model.names))
        )
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.fov_h_deg = float(self.get_parameter('fov_h_deg').value)
        self.default_object_height_m = float(self.get_parameter('default_object_height_m').value)
        self.use_realsense = bool(self.get_parameter('use_realsense').value)
        self.rs_serial = str(self.get_parameter('rs_serial').value).strip()
        self.rs_width = int(self.get_parameter('rs_width').value)
        self.rs_height = int(self.get_parameter('rs_height').value)
        self.rs_fps = int(self.get_parameter('rs_fps').value)
        self.origin_x = float(self.get_parameter('origin_x').value)
        self.origin_y = float(self.get_parameter('origin_y').value)
        self.origin_z = float(self.get_parameter('origin_z').value)
        self.calib_dx_mm = float(self.get_parameter('calib_dx_mm').value)
        self.calib_dy_mm = float(self.get_parameter('calib_dy_mm').value)
        self.calib_dz_mm = float(self.get_parameter('calib_dz_mm').value)

        self.pipeline = None
        self.align = None
        self.depth_scale = 0.0
        self.rs_fx = None
        self.rs_fy = None
        self.rs_cx = None
        self.rs_cy = None
        self.cap = None

        if self.use_realsense:
            import pyrealsense2 as rs
            self.rs = rs
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            if self.rs_serial:
                cfg.enable_device(self.rs_serial)
            cfg.enable_stream(rs.stream.depth, self.rs_width, self.rs_height, rs.format.z16, self.rs_fps)
            cfg.enable_stream(rs.stream.color, self.rs_width, self.rs_height, rs.format.bgr8, self.rs_fps)
            profile = self.pipeline.start(cfg)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())
            self.align = rs.align(rs.stream.color)
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_profile.get_intrinsics()
            self.rs_fx = float(intr.fx)
            self.rs_fy = float(intr.fy)
            self.rs_cx = float(intr.ppx)
            self.rs_cy = float(intr.ppy)
        else:
            self.cap = cv2.VideoCapture(int(self.get_parameter('camera_index').value))
            if not self.cap.isOpened():
                raise RuntimeError('카메라를 열 수 없습니다. camera_index 파라미터를 확인하세요.')

        self.local_timer = self.create_timer(0.033, self._tick_local_yolo)
        mode = 'RealSense depth' if self.use_realsense else 'pinhole approx'
        self.get_logger().info(f'로컬 YOLO 모드 시작: weights={self.weights_path} | mode={mode}')

    def cleanup_hardware(self):
        """종료 시 RealSense 파이프라인·웹캠 캡처를 닫는다."""
        if not getattr(self, 'use_local_yolo', False):
            return
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _intrinsics_from_fov(self, w: int, h: int, fov_h_deg: float):
        fh = math.radians(fov_h_deg)
        fx = (0.5 * w) / math.tan(0.5 * fh)
        fy = fx
        cx = 0.5 * w
        cy = 0.5 * h
        return fx, fy, cx, cy

    def _estimate_depth_m(self, bbox_h_px: float, fy: float, object_height_m: float) -> float:
        if bbox_h_px < 1.0:
            return float('nan')
        return float(fy * object_height_m / bbox_h_px)

    def _camera_to_project_camera_coords(self, x_optical: float, y_optical: float, z_optical: float):
        return -x_optical, y_optical, -z_optical

    def _to_absolute_coords(self, x_cam: float, y_cam: float, z_cam: float):
        return (
            x_cam - self.origin_x,
            y_cam - self.origin_y,
            z_cam - self.origin_z,
        )

    def _apply_calibration_offset_mm(self, x_abs: float, y_abs: float, z_abs: float):
        return (
            x_abs + (self.calib_dx_mm / 1000.0),
            y_abs + (self.calib_dy_mm / 1000.0),
            z_abs + (self.calib_dz_mm / 1000.0),
        )

    def _clip_box_to_image(self, x1: float, y1: float, x2: float, y2: float, w: int, h: int):
        xi1 = int(max(0, min(w - 1, round(x1))))
        yi1 = int(max(0, min(h - 1, round(y1))))
        xi2 = int(max(0, min(w - 1, round(x2))))
        yi2 = int(max(0, min(h - 1, round(y2))))
        if xi2 < xi1:
            xi1, xi2 = xi2, xi1
        if yi2 < yi1:
            yi1, yi2 = yi2, yi1
        return xi1, yi1, xi2, yi2

    def _median_depth_in_roi(self, depth_m: np.ndarray, x1: float, y1: float, x2: float, y2: float, w: int, h: int):
        bw = x2 - x1
        bh = y2 - y1
        if bw < 4 or bh < 4:
            return float('nan')
        dx = bw * 0.08 * 0.5
        dy = bh * 0.08 * 0.5
        xa, ya = x1 + dx, y1 + dy
        xb, yb = x2 - dx, y2 - dy
        if xb <= xa or yb <= ya:
            xa, ya, xb, yb = x1, y1, x2, y2
        xi1, yi1, xi2, yi2 = self._clip_box_to_image(xa, ya, xb, yb, w, h)
        roi = depth_m[yi1: yi2 + 1, xi1: xi2 + 1]
        valid = roi[np.isfinite(roi) & (roi > 0.05) & (roi < 10.0)]
        if valid.size < 3:
            return float('nan')
        return float(np.median(valid))

    def _tick_local_yolo(self):
        depth_m = None
        if self.use_realsense:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                return
            frame = np.asanyarray(color_frame.get_data())
            raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth_m = raw * float(self.depth_scale)
            fx, fy, cx, cy = self.rs_fx, self.rs_fy, self.rs_cx, self.rs_cy
        else:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return
            h, w = frame.shape[:2]
            fx, fy, cx, cy = self._intrinsics_from_fov(w, h, self.fov_h_deg)

        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False,
        )
        r0 = results[0]
        out = r0.plot()
        raw_dets = []
        objects = []

        if r0.boxes is not None and len(r0.boxes) > 0:
            boxes = r0.boxes.xyxy.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)
            confs = r0.boxes.conf.cpu().numpy().astype(float)
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                cid = int(clss[i]) if i < len(clss) else 0
                conf = float(confs[i]) if i < len(confs) else 0.0
                label = self.model_names.get(cid, str(cid))
                cx_box = 0.5 * (x1 + x2)
                cy_box = 0.5 * (y1 + y2)
                bh = max(y2 - y1, 1.0)
                if self.use_realsense and depth_m is not None:
                    z_m = self._median_depth_in_roi(depth_m, x1, y1, x2, y2, w, h)
                else:
                    z_m = self._estimate_depth_m(bh, fy, self.default_object_height_m)
                if math.isnan(z_m):
                    continue

                x_opt = ((cx_box - cx) / fx) * z_m
                y_opt = ((cy_box - cy) / fy) * z_m
                x_cam, y_cam, z_cam = self._camera_to_project_camera_coords(x_opt, y_opt, z_m)
                x_abs, y_abs, z_abs = self._to_absolute_coords(x_cam, y_cam, z_cam)
                x_abs, y_abs, z_abs = self._apply_calibration_offset_mm(x_abs, y_abs, z_abs)

                pt = (int(round(cx_box)), int(round(cy_box)))
                cv2.circle(out, pt, 6, (0, 255, 255), -1, cv2.LINE_AA)
                overlay = (
                    f'{label} c=({cx_box:.0f},{cy_box:.0f})px '
                    f'ABS=[{x_abs:+.3f},{y_abs:+.3f},{z_abs:+.3f}]m'
                )
                cv2.putText(
                    out,
                    overlay,
                    (10, 28 + (i * 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

                raw_dets.append((int(round(cx_box)), int(round(cy_box)), int(max(x2 - x1, 1.0)),
                                 int(max(y2 - y1, 1.0)), label, conf))
                objects.append({
                    'label': label,
                    'confidence': conf,
                    'depth_m': z_m,
                    'pixel_u': int(round(cx_box)),
                    'pixel_v': int(round(cy_box)),
                    'pose': {'x': x_abs, 'y': y_abs, 'z': z_abs},
                })

        rgb = np.ascontiguousarray(out[:, :, ::-1])
        hh, ww, channel = rgb.shape
        bytes_per_line = channel * ww
        self.latest_qimage = QImage(
            rgb.data, ww, hh, bytes_per_line, QImage.Format_RGB888
        ).copy()
        self.last_image_time = time.monotonic()

        self._latest_raw_detections = raw_dets
        self.detected_objects = objects
        self.last_objects_time = time.monotonic()
        self._update_selected_label_from_local_detections()

    def _update_selected_label_from_local_detections(self):
        if not self._latest_raw_detections:
            return
        if not self.selected_label:
            # 자동 선택 모드에서는 selected_label을 비워 둔다.
            return
        labels = [obj.get('label', '') for obj in self.detected_objects]
        if self.selected_label not in labels:
            self.selected_label = ''

    def _cb_image(self, msg: Image):
        """ROS Image 메시지를 QImage로 변환해 멤버 변수에 보관한다.

        변환 흐름:
          sensor_msgs/Image (BGR8, ROS)
            → OpenCV ndarray (BGR, uint8)   via CvBridge
            → OpenCV ndarray (RGB, uint8)   채널 역순 ([:, :, ::-1])
            → QImage (RGB888)               Qt 표시용

        OpenCV는 BGR, Qt는 RGB 채널 순서를 사용하므로 반드시 채널을 반전해야 한다.
        np.ascontiguousarray()로 메모리 연속성을 보장해야 QImage가 데이터를 안전하게 읽는다.
        .copy()는 QImage가 ndarray의 data 포인터를 공유하지 않고 독립 복사본을 갖도록 한다.
        (ndarray가 가비지 컬렉션되면 QImage 데이터가 깨지는 문제 방지)
        """
        if self.bridge is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # OpenCV BGR → Qt RGB: 채널 순서를 뒤집어 [:, :, ::-1]
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        height, width, channel = rgb.shape
        bytes_per_line = channel * width   # 행당 바이트 수 (stride)
        self.latest_qimage = QImage(
            rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).copy()   # ndarray 수명 독립을 위해 QImage 복사본 보관
        self.last_image_time = time.monotonic()

    def _cb_objects(self, msg: String):
        """object_detector가 발행한 검출 물체 목록(JSON)을 파싱해 멤버 변수를 갱신한다.

        JSON 형식 (object_detector._publish_detected_objects 참조):
          {
            "selected_label": "bottle",   // 현재 선택된 라벨 (빈 문자열이면 자동 선택)
            "objects": [
              {
                "label": "bottle",
                "confidence": 0.87,
                "depth_m": 0.53,
                "pixel_u": 320,
                "pixel_v": 240,
                "pose": {"x": 0.3, "y": 0.1, "z": 0.05}
              },
              ...
            ]
          }

        GUI는 selected_label과 objects 두 가지를 함께 보관해야
        버튼 강조 색상과 요약 문구를 올바르게 표시할 수 있다.
        """
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('detected_objects JSON 파싱 실패')
            return
        objects = payload.get('objects', [])
        now = time.monotonic()
        if objects:
            self._last_nonempty_objects = objects
            self._last_nonempty_objects_time = now
            self.detected_objects = objects
        elif now - self._last_nonempty_objects_time < 1.0:
            self.detected_objects = list(self._last_nonempty_objects)
        else:
            self.detected_objects = []
        self.selected_label = payload.get('selected_label', '')
        self.last_objects_time = now

    def _cb_state(self, msg: String):
        # 상태 문자열은 pick_place_node가 발행하는 값을 그대로 사용한다.
        self.pick_place_state = msg.data
        self.last_state_time = time.monotonic()

    def _cb_error(self, msg: String):
        # ERROR 진입 시 분류된 에러 사유 — _update_ui에서 좌상단 배너에 표시(GUI 스레드 안전).
        self.last_error_text = msg.data

    def _cb_gripper_service_state(self, msg: GripperState):
        self.gripper_hw_ready = msg.ready

    def _cb_gripper_init_progress(self, msg: String):
        # 그리퍼 INIT/REINIT 진행 — 메시지 수신 시각도 같이 저장(GUI에서 "stale" 판정용)
        self.gripper_init_progress = msg.data.strip()
        self.gripper_init_progress_t = time.monotonic()

    def _cb_hw_state(self, msg: Int32):
        self.hw_state = msg.data
        self.last_hw_state_time = time.monotonic()

    def _cb_speed_mode(self, msg: Int32):
        self.speed_mode = msg.data
        self.last_speed_mode_time = time.monotonic()

    def publish_selected_label(self, label: str):
        # 빈 문자열은 "자동 선택" 모드로 해석된다.
        self.selected_label = label
        msg = String()
        msg.data = label
        self.pub_selected.publish(msg)

    def call_trigger_service(self, client, label: str):
        if not client.service_is_ready():
            self.get_logger().warn(f'서비스 미연결: {label}')
            return

        future = client.call_async(Trigger.Request())

        def _done(done_future):
            try:
                res = done_future.result()
            except Exception as e:
                self.get_logger().error(f'{label} 호출 실패: {e}')
                return
            status = '성공' if res.success else '거절'
            self.get_logger().info(f'{label}: {status} - {res.message}')

        future.add_done_callback(_done)

    def refresh_system_status(self):
        now = time.monotonic()
        if now - self._last_system_status_check < 1.0:
            return
        self._last_system_status_check = now

        def fresh(stamp: float, max_age: float = 3.0) -> bool:
            return stamp > 0.0 and now - stamp <= max_age

        def ready(client) -> bool:
            return client.service_is_ready()

        # 기동 순서대로 표시 (HW=로봇 먼저 → 그리퍼 → 카메라/검출 → pick).
        # GRIP은 3단 구분: ok(준비완료) / warn(초기화 중 — init_progress 수신) / bad(무응답·죽음).
        # 콜드부팅 INITIALIZE 동안엔 죽은(빨강) 게 아니라 깨우는 중(노랑)으로 보이게 한다.
        grip_state = (
            'ok' if self.gripper_hw_ready
            else ('warn' if fresh(self.gripper_init_progress_t, 30.0) else 'bad')
        )
        self.system_status_items = [
            ('HW', 'ok' if fresh(self.last_hw_state_time) else 'warn'),
            ('GRIP', grip_state),
            ('CAM', 'ok' if fresh(self.last_image_time) else 'bad'),
            ('DET', 'ok' if fresh(self.last_objects_time) else 'bad'),
            ('PICK', 'ok' if ready(self.cli_run_once) and fresh(self.last_state_time) else 'bad'),
            ('SPD', 'ok' if fresh(self.last_speed_mode_time) else 'warn'),
        ]


class PickPlaceGui(QWidget):
    def __init__(self, ros_node: PickPlaceGuiNode):
        super().__init__()
        self.ros_node = ros_node
        self._reset_in_progress = False
        self._reset_deadline = 0.0
        self._manual_command = None
        self._manual_command_seen_active = False
        self._manual_command_deadline = 0.0
        self._manual_feedback = ''
        self._manual_feedback_until = 0.0
        self._manual_command_token = 0
        self._gripper_feedback_hold_sec = 2.2
        self.object_buttons = {}
        self._stable_labels = []
        self._candidate_labels = []
        self._candidate_label_hits = 0
        self._label_stable_frames = 3
        self._settings_path = Path.home() / '.config' / 'dsr_realsense_pick_place' / 'gui_settings.json'
        self._settings = self._load_gui_settings()
        self._calib_current_mm = [None, None, None]
        self._object_settings_loaded = False
        self._object_settings_loading = False
        self._saved_model_applied = False
        self._last_object_settings_attempt = 0.0
        self._system_reset_proc = None   # shutdown_nodes.sh 프로세스
        self._system_restart_proc = None # 재시작 launch 프로세스
        self._system_reset_phase = ''    # '' | 'shutting_down' | 'waiting' | 'restarting'
        self._system_reset_phase_until = 0.0
        self._gripper_bridge_restart_proc = None   # restart_gripper_bridge.sh 프로세스
        self._gripper_bridge_restart_until = 0.0   # 이 시각까지 버튼 비활성(재기동 진행 표시)

        # 좌측은 카메라 영상, 우측은 상태/선택 패널로 나누어 배치한다.
        self.setWindowTitle('DSR RealSense Pick & Place GUI')
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.resize(1100, 720)
        self.move(40, 40)

        # 최상위: 상태바(항상 표시) + 탭 위젯
        root = QVBoxLayout(self)

        self.tabs = QTabWidget()
        # 가장 넓은 탭(모델·캘리브)이 전체 창 폭을 끌어올리는 것 차단.
        # 각 탭 안 위젯은 QSizePolicy에 따라 이 폭에 맞춰 압축됨(스크롤 영역이 흡수).
        self.tabs.setMaximumWidth(440)

        # 탭1 "운전": 카메라(본문 좌측 고정) + 우측 단일 컬럼(긴급제어/상태/물체선택/실시간전류)
        _tab_op = QWidget()
        op_right = QVBoxLayout(_tab_op)

        # 탭2 "gripper": 그리퍼 동작/정밀 전류/물체별 강도 (스크롤)
        _tab_grip = QWidget()
        _grip_outer = QVBoxLayout(_tab_grip)
        _grip_scroll = QScrollArea()
        _grip_scroll.setWidgetResizable(True)
        _grip_content = QWidget()
        grip_col = QVBoxLayout(_grip_content)
        _grip_scroll.setWidget(_grip_content)
        _grip_outer.addWidget(_grip_scroll)

        # 탭3 "수동·설정": 로봇 수동 제어/안전/모델·캘리브 (스크롤)
        _tab_set = QWidget()
        _set_outer = QVBoxLayout(_tab_set)
        _set_scroll = QScrollArea()
        _set_scroll.setWidgetResizable(True)
        _set_content = QWidget()
        set_col = QVBoxLayout(_set_content)
        _set_scroll.setWidget(_set_content)
        _set_outer.addWidget(_set_scroll)

        # 각 그룹은 op_right(운전)/grip_col(gripper)/set_col(수동·설정)에 직접 추가한다
        self.system_status_labels = {}
        self.system_status_bar = QWidget()
        self.system_status_bar.setFixedSize(276, 24)
        status_bar_layout = QHBoxLayout(self.system_status_bar)
        status_bar_layout.setContentsMargins(0, 0, 0, 4)
        status_bar_layout.setSpacing(4)
        for key in ('HW', 'GRIP', 'CAM', 'DET', 'PICK', 'SPD'):  # 기동 순서대로 표시
            label = QLabel(key)
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(42, 20)
            label.setStyleSheet(
                'background-color: #666; color: white; border-radius: 3px;'
                'font-size: 11px; font-weight: bold;'
            )
            self.system_status_labels[key] = label
            status_bar_layout.addWidget(label)

        # status 점들(좌) + 현재 동작/단계 텍스트(우)를 한 줄에 배치.
        self.activity_label = QLabel('⚙️ 시스템 시작 중...')
        self.activity_label.setStyleSheet(
            'background-color: #1a2a3a; color: #cfe8ff; border-radius: 3px;'
            ' padding: 2px 8px; font-size: 12px; font-weight: bold;'
        )
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 4)
        top_row.setSpacing(8)
        top_row.addWidget(self.system_status_bar)
        top_row.addWidget(self.activity_label, 1)
        root.addLayout(top_row)

        # 🔴 에러 배너 — ERROR 시 '어떤 에러인지' 좌상단 status 바 바로 아래에 명확히 표기
        self.error_banner_label = QLabel('')
        self.error_banner_label.setStyleSheet(
            'color: #ff5555; font-weight: bold; font-size: 12px;'
            'background-color: #2a1010; border-radius: 3px; padding: 2px 6px;'
        )
        self.error_banner_label.setWordWrap(True)
        self.error_banner_label.setVisible(False)
        root.addWidget(self.error_banner_label, 0, Qt.AlignLeft)

        # 상태 그룹박스 — 카메라와 그래프 사이(cam_col)에 배치. 카메라 폭에 맞춰 클램프.
        status_group = QGroupBox('상태')
        status_group.setMaximumHeight(54)
        # 내부 라벨들이 길어 cam_col 폭을 키우는 것을 방지(전체 창이 같이 넓어짐).
        status_group.setMaximumWidth(640)
        status_layout = QHBoxLayout(status_group)
        status_layout.setContentsMargins(8, 2, 8, 2)
        status_layout.setSpacing(12)

        self.state_label = QLabel('Pick & Place 상태: IDLE')
        self.selection_label = QLabel('선택 물체: 자동 선택')
        self.selection_status_label = QLabel('선택 상태: 자동으로 가장 가까운 물체를 사용')
        self.command_status_label = QLabel('')
        self.command_status_label.setStyleSheet('color: #b0b0b0; font-weight: bold;')
        # 그리퍼 INIT/REINIT 진행 — 막힘과 진행 중 구분용. /gripper_service/init_progress 구독.
        self.gripper_init_label = QLabel('그리퍼: 대기')
        self.gripper_init_label.setStyleSheet(
            'color: #aaa; font-weight: bold; padding: 2px 6px; border-radius: 4px;'
            'background-color: #2a2a2a;'
        )

        status_layout.addWidget(self.state_label)
        status_layout.addWidget(self.selection_label)
        status_layout.addWidget(self.selection_status_label)
        status_layout.addWidget(self.command_status_label)
        status_layout.addWidget(self.gripper_init_label)
        status_layout.addStretch(1)
        # status_group은 운전 탭 우측 컬럼(op_right)에서 긴급 제어 아래에 배치(아래 조립부)

        compact_settings_group = QGroupBox('모델 설정 / 수동 캘리브레이션')
        compact_settings_group.setMaximumHeight(108)
        compact_settings_layout = QVBoxLayout(compact_settings_group)
        compact_settings_layout.setContentsMargins(8, 5, 8, 5)
        compact_settings_layout.setSpacing(3)

        model_row = QHBoxLayout()
        model_row.setSpacing(4)
        model_label = QLabel('모델')
        model_label.setFixedWidth(54)
        model_row.addWidget(model_label)
        self.model_path_edit = QLineEdit(str(self._settings.get('yolo_model_path', '')))
        self.model_path_edit.setPlaceholderText('YOLO .pt 파일 경로')
        self.model_path_edit.setFixedHeight(24)
        self.model_path_edit.editingFinished.connect(self._model_path_edited)
        self.model_browse_button = QPushButton('찾기')
        self.model_browse_button.setFixedSize(48, 24)
        self.model_browse_button.clicked.connect(self._model_browse)
        self.model_apply_button = QPushButton('적용')
        self.model_apply_button.setFixedSize(48, 24)
        self.model_apply_button.clicked.connect(lambda: self._model_apply(save=True))
        model_row.addWidget(self.model_path_edit, 1)
        model_row.addWidget(self.model_browse_button)
        model_row.addWidget(self.model_apply_button)
        compact_settings_layout.addLayout(model_row)

        calib_edit_row = QHBoxLayout()
        calib_edit_row.setSpacing(4)
        edit_label = QLabel('수정값')
        edit_label.setFixedWidth(54)
        calib_edit_row.addWidget(edit_label)
        self._calib_offset_spins = []
        for axis in ('X', 'Y', 'Z'):
            axis_label = QLabel(f'{axis}축')
            axis_label.setFixedWidth(24)
            calib_edit_row.addWidget(axis_label)
            spin = QDoubleSpinBox()
            spin.setRange(-300.0, 300.0)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setAlignment(Qt.AlignRight)
            spin.setFixedSize(78, 24)
            self._calib_offset_spins.append(spin)
            calib_edit_row.addWidget(spin)
        self.calib_load_button = QPushButton('불러오기')
        self.calib_load_button.setFixedSize(62, 24)
        self.calib_load_button.clicked.connect(self._calib_load)
        self.calib_apply_button = QPushButton('적용')
        self.calib_apply_button.setFixedSize(48, 24)
        self.calib_apply_button.clicked.connect(self._calib_apply)
        calib_edit_row.addStretch(1)
        calib_edit_row.addWidget(self.calib_load_button)
        calib_edit_row.addWidget(self.calib_apply_button)
        compact_settings_layout.addLayout(calib_edit_row)

        calib_current_row = QHBoxLayout()
        calib_current_row.setSpacing(4)
        current_label = QLabel('현재값')
        current_label.setFixedWidth(54)
        calib_current_row.addWidget(current_label)
        self.calib_current_labels = {}
        for axis in ('X', 'Y', 'Z'):
            axis_label = QLabel(f'{axis}축')
            axis_label.setFixedWidth(24)
            value_label = QLabel('--.- mm')
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_label.setFixedWidth(78)
            value_label.setStyleSheet('color: #b0b0b0; font-family: monospace; font-size: 11px;')
            self.calib_current_labels[axis] = value_label
            calib_current_row.addWidget(axis_label)
            calib_current_row.addWidget(value_label)
        calib_current_row.addStretch(1)
        compact_settings_layout.addLayout(calib_current_row)
        set_col.addWidget(compact_settings_group)

        self.image_label = QLabel('카메라 영상 대기 중...')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet(
            'background-color: #1e1e1e; color: white; border-radius: 10px;'
        )
        # image_label(카메라)은 탭과 무관하게 항상 보이도록 본문 좌측(cam_col)에 배치 — 아래 조립부 참고

        # ── 긴급 제어 패널 (탭1 "운전" 우측 상단) ──────────────────

        emergency_group = QGroupBox('긴급 제어')
        emergency_layout = QVBoxLayout(emergency_group)

        self.e_stop_button = QPushButton('⛔  긴급 정지  (E-STOP)')
        self.e_stop_button.setMinimumHeight(54)
        self.e_stop_button.setStyleSheet(
            'QPushButton {'
            '  background-color: #cc0000; color: white;'
            '  font-size: 16px; font-weight: bold; border-radius: 8px;'
            '}'
            'QPushButton:hover { background-color: #ff1a1a; }'
            'QPushButton:pressed { background-color: #990000; }'
            'QPushButton:disabled { background-color: #555; color: #999; }'
        )
        self.e_stop_button.clicked.connect(self._e_stop)

        self.cancel_button = QPushButton('🚫  태스크 중단')
        self.cancel_button.setMinimumHeight(38)
        self.cancel_button.setStyleSheet(
            'QPushButton {'
            '  background-color: #e65c00; color: white;'
            '  font-size: 13px; font-weight: bold; border-radius: 6px;'
            '}'
            'QPushButton:hover { background-color: #ff6600; }'
            'QPushButton:pressed { background-color: #b34700; }'
            'QPushButton:disabled { background-color: #555; color: #999; }'
        )
        self.cancel_button.clicked.connect(self._cancel_task)

        self.e_stop_reset_button = QPushButton('✅  긴급정지 해제')
        self.e_stop_reset_button.setMinimumHeight(38)
        self.e_stop_reset_button.setStyleSheet(
            'QPushButton {'
            '  background-color: #1a7a1a; color: white;'
            '  font-size: 13px; font-weight: bold; border-radius: 6px;'
            '}'
            'QPushButton:hover { background-color: #22aa22; }'
            'QPushButton:pressed { background-color: #115511; }'
            'QPushButton:disabled { background-color: #555; color: #999; }'
        )
        self.e_stop_reset_button.clicked.connect(self._e_stop_reset)
        self.e_stop_reset_button.setEnabled(False)

        # 에러 해제 — ERROR 상태에서만 활성. 로봇 이동 없이 알람 리셋 + 그리퍼 reinit만.
        self.clear_error_button = QPushButton('에러 해제')
        self.clear_error_button.setMinimumHeight(38)
        self.clear_error_button.setStyleSheet(
            'QPushButton {'
            '  background-color: #b58900; color: white;'
            '  font-size: 13px; font-weight: bold; border-radius: 6px;'
            '}'
            'QPushButton:hover { background-color: #d6a000; }'
            'QPushButton:pressed { background-color: #806000; }'
            'QPushButton:disabled { background-color: #555; color: #999; }'
        )
        self.clear_error_button.clicked.connect(self._clear_error)
        self.clear_error_button.setEnabled(False)

        emergency_layout.addWidget(self.e_stop_button)
        emergency_layout.addWidget(self.cancel_button)
        emergency_layout.addWidget(self.clear_error_button)
        emergency_layout.addWidget(self.e_stop_reset_button)



        # 로봇 수동 제어(HOME/복구) — 수동·설정 탭. 그리퍼 동작은 gripper 탭으로 분리.
        robot_ctrl_group = QGroupBox('로봇 수동 제어')
        robot_ctrl_grid = QGridLayout(robot_ctrl_group)
        robot_ctrl_grid.setSpacing(6)
        robot_ctrl_grid.setContentsMargins(8, 6, 8, 6)

        self.home_button = QPushButton('HOME 이동')
        self.home_button.setMinimumHeight(32)
        self.home_button.clicked.connect(self._go_home)

        self.recover_home_button = QPushButton('에러 복구 & HOME 복귀')
        self.recover_home_button.setMinimumHeight(32)
        self.recover_home_button.clicked.connect(self._recover_to_home)

        robot_ctrl_grid.addWidget(self.home_button, 0, 0)
        robot_ctrl_grid.addWidget(self.recover_home_button, 0, 1)

        # 그리퍼 동작 버튼 — gripper 탭. open/close/리셋/토크.
        gripper_action_group = QGroupBox('그리퍼 동작')
        gripper_action_grid = QGridLayout(gripper_action_group)
        gripper_action_grid.setSpacing(6)
        gripper_action_grid.setContentsMargins(8, 6, 8, 6)

        self.gripper_open_button = QPushButton('그리퍼 OPEN')
        self.gripper_open_button.setMinimumHeight(32)
        self.gripper_open_button.clicked.connect(self._gripper_open)

        self.gripper_close_button = QPushButton('그리퍼 CLOSE')
        self.gripper_close_button.setMinimumHeight(32)
        self.gripper_close_button.clicked.connect(self._gripper_close)

        # [2026-06-09] 그리퍼 리셋(재초기화) 버튼 제거 — in-process reinit이 DrlStart 실패로
        # 오히려 status3를 유발(복구 불가)해 무용+위험. status3는 start_clean 전체 재시작이 답.
        self.gripper_torque_on_button = QPushButton('토크 ON')
        self.gripper_torque_on_button.setMinimumHeight(32)
        self.gripper_torque_on_button.clicked.connect(self._gripper_torque_on)

        self.gripper_torque_off_button = QPushButton('토크 OFF')
        self.gripper_torque_off_button.setMinimumHeight(32)
        self.gripper_torque_off_button.clicked.connect(self._gripper_torque_off)

        gripper_action_grid.addWidget(self.gripper_open_button, 0, 0)
        gripper_action_grid.addWidget(self.gripper_close_button, 0, 1)
        gripper_action_grid.addWidget(self.gripper_torque_on_button, 1, 0)
        gripper_action_grid.addWidget(self.gripper_torque_off_button, 1, 1)

        # ── 그리퍼 정밀 전류 및 속도/가속도 제어 패널 ───────────────────────
        self.gripper_ctrl_group = QGroupBox('그리퍼 정밀 전류 제어')
        gripper_ctrl_layout = QVBoxLayout(self.gripper_ctrl_group)
        gripper_ctrl_layout.setContentsMargins(8, 6, 8, 6)
        gripper_ctrl_layout.setSpacing(4)

        # 닫기 전류
        close_curr_row = QHBoxLayout()
        close_curr_label = QLabel('닫기 전류:')
        close_curr_label.setFixedWidth(64)
        self.close_curr_slider = QSlider(Qt.Horizontal)
        self.close_curr_slider.setRange(50, 800)
        self.close_curr_slider.setValue(300)
        self.close_curr_spin = QSpinBox()
        self.close_curr_spin.setRange(50, 800)
        self.close_curr_spin.setValue(300)
        self.close_curr_spin.setFixedWidth(54)
        self.close_curr_slider.valueChanged.connect(self.close_curr_spin.setValue)
        self.close_curr_spin.valueChanged.connect(self.close_curr_slider.setValue)
        close_curr_row.addWidget(close_curr_label)
        close_curr_row.addWidget(self.close_curr_slider)
        close_curr_row.addWidget(self.close_curr_spin)
        gripper_ctrl_layout.addLayout(close_curr_row)

        # 열기 전류
        open_curr_row = QHBoxLayout()
        open_curr_label = QLabel('열기 전류:')
        open_curr_label.setFixedWidth(64)
        self.open_curr_slider = QSlider(Qt.Horizontal)
        self.open_curr_slider.setRange(50, 800)
        self.open_curr_slider.setValue(400)
        self.open_curr_spin = QSpinBox()
        self.open_curr_spin.setRange(50, 800)
        self.open_curr_spin.setValue(400)
        self.open_curr_spin.setFixedWidth(54)
        self.open_curr_slider.valueChanged.connect(self.open_curr_spin.setValue)
        self.open_curr_spin.valueChanged.connect(self.open_curr_slider.setValue)
        open_curr_row.addWidget(open_curr_label)
        open_curr_row.addWidget(self.open_curr_slider)
        open_curr_row.addWidget(self.open_curr_spin)
        gripper_ctrl_layout.addLayout(open_curr_row)

        # 이송 전류 (파지 후 들고 이동 시 — 낮으면 발열↓, 너무 낮으면 이송 중 낙하)
        transport_curr_row = QHBoxLayout()
        transport_curr_label = QLabel('이송 전류:')
        transport_curr_label.setFixedWidth(64)
        self.transport_curr_slider = QSlider(Qt.Horizontal)
        self.transport_curr_slider.setRange(50, 800)
        self.transport_curr_slider.setValue(150)
        self.transport_curr_spin = QSpinBox()
        self.transport_curr_spin.setRange(50, 800)
        self.transport_curr_spin.setValue(150)
        self.transport_curr_spin.setFixedWidth(54)
        self.transport_curr_slider.valueChanged.connect(self.transport_curr_spin.setValue)
        self.transport_curr_spin.valueChanged.connect(self.transport_curr_slider.setValue)
        transport_curr_row.addWidget(transport_curr_label)
        transport_curr_row.addWidget(self.transport_curr_slider)
        transport_curr_row.addWidget(self.transport_curr_spin)
        gripper_ctrl_layout.addLayout(transport_curr_row)

        # 속도
        vel_row = QHBoxLayout()
        vel_label = QLabel('구동 속도:')
        vel_label.setFixedWidth(64)
        self.vel_slider = QSlider(Qt.Horizontal)
        self.vel_slider.setRange(100, 1500)
        self.vel_slider.setValue(1500)
        self.vel_spin = QSpinBox()
        self.vel_spin.setRange(100, 1500)
        self.vel_spin.setValue(1500)
        self.vel_spin.setFixedWidth(54)
        self.vel_slider.valueChanged.connect(self.vel_spin.setValue)
        self.vel_spin.valueChanged.connect(self.vel_slider.setValue)
        vel_row.addWidget(vel_label)
        vel_row.addWidget(self.vel_slider)
        vel_row.addWidget(self.vel_spin)
        gripper_ctrl_layout.addLayout(vel_row)

        # 가속도
        acc_row = QHBoxLayout()
        acc_label = QLabel('구동 가속:')
        acc_label.setFixedWidth(64)
        self.acc_slider = QSlider(Qt.Horizontal)
        self.acc_slider.setRange(100, 1000)
        self.acc_slider.setValue(1000)
        self.acc_spin = QSpinBox()
        self.acc_spin.setRange(100, 1000)
        self.acc_spin.setValue(1000)
        self.acc_spin.setFixedWidth(54)
        self.acc_slider.valueChanged.connect(self.acc_spin.setValue)
        self.acc_spin.valueChanged.connect(self.acc_slider.setValue)
        acc_row.addWidget(acc_label)
        acc_row.addWidget(self.acc_slider)
        acc_row.addWidget(self.acc_spin)
        gripper_ctrl_layout.addLayout(acc_row)

        # 모니터링 레이블
        self.gripper_status_label = QLabel('실시간 - 전류: -- mA | 위치: ----')
        self.gripper_status_label.setStyleSheet(
            'color: #33ff33; font-weight: bold; background-color: #1e1e1e; padding: 1px 4px; border-radius: 4px; font-family: monospace;'
        )
        self.gripper_status_label.setAlignment(Qt.AlignCenter)
        self.gripper_status_label.setMaximumHeight(22)  # 폰트 한 줄 높이만큼만 — 두껍지 않게

        # 실시간 전류 모니터링 그래프 추가
        self.realtime_graph = RealTimeGraphWidget(self)
        # 카메라 아래(cam_col)에 모든 탭 상시 배치. 가로로 길게(더 긴 시간), 높이는 낮게.
        self.realtime_graph.max_len = 300   # ≈30초 history (_update_ui 10Hz 기준)
        self.realtime_graph.setMinimumHeight(90)
        self.realtime_graph.setMaximumHeight(150)

        # 적용 버튼
        self.gripper_apply_button = QPushButton('설정 적용')
        self.gripper_apply_button.setMinimumHeight(28)
        self.gripper_apply_button.clicked.connect(self._gripper_apply)
        self.gripper_apply_button.setStyleSheet(
            'QPushButton { background-color: #2a2a5a; color: white;'
            '  font-weight: bold; border-radius: 5px; }'
            'QPushButton:hover { background-color: #3a3a80; }'
            'QPushButton:disabled { background-color: #444; color: #888; }'
        )
        gripper_ctrl_layout.addWidget(self.gripper_apply_button)

        # 📂 불러오기 — yaml의 현재 값을 슬라이더로 로드 (적용/저장 전 기준값 복원)
        self.gripper_load_button = QPushButton('📂 불러오기')
        self.gripper_load_button.setMinimumHeight(28)
        self.gripper_load_button.clicked.connect(self._gripper_ctrl_load)
        self.gripper_load_button.setStyleSheet(
            'QPushButton { background-color: #3a3a3a; color: white;'
            '  font-weight: bold; border-radius: 5px; }'
            'QPushButton:hover { background-color: #505050; }'
        )
        gripper_ctrl_layout.addWidget(self.gripper_load_button)

        # 💾 yaml 저장 — GUI 튜닝을 파일에 영구 반영(yaml 공유로 다른 PC에서 동일 튜닝 재현)
        self.gripper_save_button = QPushButton('💾 yaml 저장')
        self.gripper_save_button.setMinimumHeight(28)
        self.gripper_save_button.clicked.connect(self._gripper_ctrl_save)
        self.gripper_save_button.setStyleSheet(
            'QPushButton { background-color: #2a5a2a; color: white;'
            '  font-weight: bold; border-radius: 5px; }'
            'QPushButton:hover { background-color: #3a803a; }'
        )
        gripper_ctrl_layout.addWidget(self.gripper_save_button)

        # ── 물체별 파지 강도 (gripper 탭) ───────────────────────────────
        # config(pick_place_params.yaml)에서 클래스↔전류를 읽어 슬라이더 행을 동적 생성.
        # [적용] → pick_place_node 파라미터 라이브 설정, [저장] → yaml 파일 갱신.
        names, currents, default, cmin, cmax = self._load_grip_strength_config()
        self._grip_cmin, self._grip_cmax = cmin, cmax
        self.grip_strength_group = QGroupBox('물체별 파지 강도 (mA)')
        grip_strength_layout = QVBoxLayout(self.grip_strength_group)
        grip_strength_layout.setContentsMargins(8, 6, 8, 6)
        grip_strength_layout.setSpacing(4)

        info_label = QLabel('낮을수록 약하게 파지. 미인식/맵에 없는 물체는 "기본값" 사용.')
        info_label.setStyleSheet('color: #aaa; font-size: 11px;')
        info_label.setWordWrap(True)
        grip_strength_layout.addWidget(info_label)

        # class명 → (slider, spin). 기본값 행은 '__default__' 키로 보관.
        self._grip_strength_rows = {}

        def _make_curr_row(caption: str, value: int):
            row = QHBoxLayout()
            lab = QLabel(caption)
            lab.setFixedWidth(72)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(cmin, cmax)
            slider.setValue(int(value))
            spin = QSpinBox()
            spin.setRange(cmin, cmax)
            spin.setValue(int(value))
            spin.setFixedWidth(56)
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            row.addWidget(lab)
            row.addWidget(slider)
            row.addWidget(spin)
            grip_strength_layout.addLayout(row)
            return slider, spin

        for _n, _c in zip(names, currents):
            self._grip_strength_rows[_n] = _make_curr_row(_n, _c)
        # 미인식 기본값 행
        self._grip_strength_rows['__default__'] = _make_curr_row('미인식 기본', default)

        grip_btn_row = QHBoxLayout()
        self.grip_strength_apply_button = QPushButton('물체별 강도 적용')
        self.grip_strength_apply_button.setMinimumHeight(30)
        self.grip_strength_apply_button.clicked.connect(self._grip_strength_apply)
        self.grip_strength_apply_button.setStyleSheet(
            'QPushButton { background-color: #2a2a5a; color: white;'
            '  font-weight: bold; border-radius: 5px; }'
            'QPushButton:hover { background-color: #3a3a80; }'
            'QPushButton:disabled { background-color: #444; color: #888; }'
        )
        self.grip_strength_save_button = QPushButton('💾 파일 저장')
        self.grip_strength_save_button.setMinimumHeight(30)
        self.grip_strength_save_button.clicked.connect(self._grip_strength_save)
        self.grip_strength_save_button.setStyleSheet(
            'QPushButton { background-color: #2a4a2a; color: white;'
            '  font-weight: bold; border-radius: 5px; }'
            'QPushButton:hover { background-color: #3a6a3a; }'
            'QPushButton:disabled { background-color: #444; color: #888; }'
        )
        grip_btn_row.addWidget(self.grip_strength_apply_button)
        grip_btn_row.addWidget(self.grip_strength_save_button)
        grip_strength_layout.addLayout(grip_btn_row)

        self.grip_strength_status_label = QLabel('')
        self.grip_strength_status_label.setStyleSheet('color: #aaa; font-size: 11px;')
        self.grip_strength_status_label.setWordWrap(True)
        grip_strength_layout.addWidget(self.grip_strength_status_label)


        # ── 시스템 안전 및 동작 모드 (기존 안전 모드 및 Doosan 안전 모드 통합) ──
        safety_group = QGroupBox('시스템 안전 및 동작 모드')
        safety_layout = QVBoxLayout(safety_group)
        safety_layout.setContentsMargins(8, 6, 8, 6)
        safety_layout.setSpacing(4)

        # 하드웨어 상태 / 속도 모드 표시 행
        hw_row = QHBoxLayout()
        self.hw_state_label    = QLabel('HW: --')
        self.speed_mode_label  = QLabel('속도: --')
        self.hw_state_label.setStyleSheet(
            'font-weight: bold; padding: 4px 8px; border-radius: 4px;'
            'background-color: #2a2a2a;'
        )
        self.speed_mode_label.setStyleSheet(
            'font-weight: bold; padding: 4px 8px; border-radius: 4px;'
            'background-color: #2a2a2a;'
        )
        hw_row.addWidget(self.hw_state_label)
        hw_row.addStretch(1)
        hw_row.addWidget(self.speed_mode_label)
        safety_layout.addLayout(hw_row)

        # 속도 모드 전환 행
        speed_row = QHBoxLayout()
        self.speed_normal_button = QPushButton('🟢 정상 속도')
        self.speed_normal_button.setMinimumHeight(32)
        self.speed_normal_button.setStyleSheet(
            'QPushButton { background-color: #1a5c1a; color: white;'
            '  font-weight: bold; border-radius: 5px; }'
            'QPushButton:hover { background-color: #22881a; }'
            'QPushButton:disabled { background-color: #444; color: #888; }'
        )
        self.speed_normal_button.clicked.connect(self._speed_normal)

        self.speed_reduced_button = QPushButton('🟡 감속 모드')
        self.speed_reduced_button.setMinimumHeight(32)
        self.speed_reduced_button.setStyleSheet(
            'QPushButton { background-color: #7a6000; color: white;'
            '  font-weight: bold; border-radius: 5px; }'
            'QPushButton:hover { background-color: #aa8800; }'
            'QPushButton:disabled { background-color: #444; color: #888; }'
        )
        self.speed_reduced_button.clicked.connect(self._speed_reduced)

        speed_row.addWidget(self.speed_normal_button)
        speed_row.addWidget(self.speed_reduced_button)
        safety_layout.addLayout(speed_row)

        # 서보 OFF / ON 행
        servo_row = QHBoxLayout()
        self.servo_off_button = QPushButton('⚡ 서보 OFF')
        self.servo_off_button.setMinimumHeight(32)
        self.servo_off_button.setStyleSheet(
            'QPushButton { background-color: #5a0050; color: white;'
            '  font-weight: bold; border-radius: 5px; }'
            'QPushButton:hover { background-color: #880077; }'
            'QPushButton:disabled { background-color: #444; color: #888; }'
        )
        self.servo_off_button.clicked.connect(self._servo_off)

        self.servo_on_button = QPushButton('🟢 서보 ON')
        self.servo_on_button.setMinimumHeight(32)
        self.servo_on_button.setEnabled(False)
        self.servo_on_button.setStyleSheet(
            'QPushButton { background-color: #006600; color: white;'
            '  font-weight: bold; border-radius: 5px; }'
            'QPushButton:hover { background-color: #009900; }'
            'QPushButton:disabled { background-color: #444; color: #888; }'
        )
        self.servo_on_button.clicked.connect(self._servo_on)

        servo_row.addWidget(self.servo_off_button)
        servo_row.addWidget(self.servo_on_button)
        safety_layout.addLayout(servo_row)

        # 안전 구분선
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        divider.setStyleSheet('background-color: #444;')
        safety_layout.addWidget(divider)

        # Doosan 안전 모드 통합 영역
        self.safety_mode_label = QLabel('현재 안전 모드: 알 수 없음')
        self.safety_mode_label.setStyleSheet(
            'font-weight: bold; padding: 3px 6px; border-radius: 4px;'
            'background-color: #2a2a2a; color: white;'
        )
        safety_layout.addWidget(self.safety_mode_label)

        safety_mode_row = QHBoxLayout()
        self.safety_auto_button = QPushButton('🤖  정상 운전')
        self.safety_auto_button.setMinimumHeight(40)
        self.safety_auto_button.setToolTip('AUTONOMOUS — 정상 Pick & Place 자율 운전')
        self.safety_auto_button.setStyleSheet(
            'QPushButton { background-color: #003a70; color: white;'
            '  font-size: 13px; font-weight: bold; border-radius: 6px; }'
            'QPushButton:hover { background-color: #0055a0; }'
            'QPushButton:disabled { background-color: #444; color: #888; }'
        )
        self.safety_auto_button.clicked.connect(self._safety_normal)

        self.safety_backdrive_button = QPushButton('✋  역구동')
        self.safety_backdrive_button.setMinimumHeight(40)
        self.safety_backdrive_button.setToolTip('BACKDRIVE — 외력으로 로봇 수동 이동 가능')
        self.safety_backdrive_button.setStyleSheet(
            'QPushButton { background-color: #2a2a5a; color: white;'
            '  font-size: 13px; font-weight: bold; border-radius: 6px; }'
            'QPushButton:hover { background-color: #3a3a80; }'
            'QPushButton:disabled { background-color: #444; color: #888; }'
        )
        self.safety_backdrive_button.clicked.connect(self._safety_backdrive)

        safety_mode_row.addWidget(self.safety_auto_button)
        safety_mode_row.addWidget(self.safety_backdrive_button)
        safety_layout.addLayout(safety_mode_row)

        # 시스템 리셋 버튼 (GUI 제외 전체 노드 재시작)
        reset_divider = QFrame()
        reset_divider.setFrameShape(QFrame.HLine)
        reset_divider.setFrameShadow(QFrame.Sunken)
        reset_divider.setStyleSheet('background-color: #444;')
        safety_layout.addWidget(reset_divider)

        self.system_reset_button = QPushButton('🔄  시스템 리셋 (GUI 유지)')
        self.system_reset_button.setMinimumHeight(36)
        self.system_reset_button.setToolTip(
            'GUI를 제외한 모든 노드를 정상 종료 후 재시작합니다.\n'
            'DRCF 연결이 정상 해제되어 joint가 즉시 활성화됩니다.'
        )
        self.system_reset_button.setStyleSheet(
            'QPushButton { background-color: #4a3000; color: white;'
            '  font-weight: bold; border-radius: 5px; }'
            'QPushButton:hover { background-color: #7a5000; }'
            'QPushButton:disabled { background-color: #444; color: #888; }'
        )
        self.system_reset_button.clicked.connect(self._system_reset)
        safety_layout.addWidget(self.system_reset_button)

        self.system_reset_label = QLabel('')
        self.system_reset_label.setStyleSheet('color: #aaa; font-size: 11px;')
        self.system_reset_label.setAlignment(Qt.AlignCenter)
        safety_layout.addWidget(self.system_reset_label)

        # ── 그리퍼 브릿지 재시작 (status3 가벼운 복구) ─────────────────
        # 그리퍼 status3(Modbus 무응답)가 "그리퍼 리셋"(in-process reinit)으로 안 풀릴 때,
        # 브릿지 노드만 새 프로세스로 재기동해 ~5초 복구한다. 로봇/카메라는 안 건드림.
        self.gripper_bridge_restart_button = QPushButton('🔧  그리퍼 브릿지 재시작 (status3 복구)')
        self.gripper_bridge_restart_button.setMinimumHeight(34)
        self.gripper_bridge_restart_button.setToolTip(
            '그리퍼 브릿지(gripper_service_node + gripper_node)만 새 프로세스로 재기동합니다.\n'
            '"그리퍼 리셋"으로 안 풀리는 status3(Modbus 무응답) 복구용 — 전원 사이클 불필요.')
        self.gripper_bridge_restart_button.setStyleSheet(
            'QPushButton { background-color: #4a3000; color: white;'
            '  font-weight: bold; border-radius: 5px; }'
            'QPushButton:hover { background-color: #7a5000; }'
            'QPushButton:disabled { background-color: #444; color: #888; }'
        )
        self.gripper_bridge_restart_button.clicked.connect(self._gripper_bridge_restart)
        safety_layout.addWidget(self.gripper_bridge_restart_button)

        self.gripper_bridge_restart_label = QLabel('')
        self.gripper_bridge_restart_label.setStyleSheet('color: #aaa; font-size: 11px;')
        self.gripper_bridge_restart_label.setAlignment(Qt.AlignCenter)
        safety_layout.addWidget(self.gripper_bridge_restart_label)

        # ── TCP Z 절대 하한 (min_safe_z) ───────────────────────────────
        # pick_place_node가 모든 직교 이동에서 이 높이(base_link 기준, m) 아래로
        # 못 내려가게 클램프한다. [적용]=라이브 set_parameters, [저장]=yaml 영구 반영.
        zfloor_divider = QFrame()
        zfloor_divider.setFrameShape(QFrame.HLine)
        zfloor_divider.setFrameShadow(QFrame.Sunken)
        zfloor_divider.setStyleSheet('background-color: #444;')
        safety_layout.addWidget(zfloor_divider)

        zfloor_caption = QLabel('TCP Z 안전 하한 (m)')
        zfloor_caption.setStyleSheet('font-weight: bold; color: #ddd;')
        safety_layout.addWidget(zfloor_caption)

        zfloor_info = QLabel('이 높이 아래로는 로봇이 내려가지 않음. 테이블/지그 높이로 설정해 충돌 방지.')
        zfloor_info.setStyleSheet('color: #aaa; font-size: 11px;')
        zfloor_info.setWordWrap(True)
        safety_layout.addWidget(zfloor_info)

        zfloor_row = QHBoxLayout()
        zfloor_label = QLabel('하한 Z:')
        zfloor_label.setFixedWidth(56)
        self.min_safe_z_spin = QDoubleSpinBox()
        self.min_safe_z_spin.setRange(0.0, 0.60)
        self.min_safe_z_spin.setSingleStep(0.005)
        self.min_safe_z_spin.setDecimals(3)
        self.min_safe_z_spin.setValue(self._load_min_safe_z())
        self.min_safe_z_spin.setFixedWidth(80)
        self.min_safe_z_apply_button = QPushButton('적용')
        self.min_safe_z_apply_button.setFixedSize(48, 26)
        self.min_safe_z_apply_button.clicked.connect(self._min_safe_z_apply)
        self.min_safe_z_save_button = QPushButton('💾 저장')
        self.min_safe_z_save_button.setFixedSize(64, 26)
        self.min_safe_z_save_button.clicked.connect(self._min_safe_z_save)
        zfloor_row.addWidget(zfloor_label)
        zfloor_row.addWidget(self.min_safe_z_spin)
        zfloor_row.addStretch(1)
        zfloor_row.addWidget(self.min_safe_z_apply_button)
        zfloor_row.addWidget(self.min_safe_z_save_button)
        safety_layout.addLayout(zfloor_row)

        self.min_safe_z_status_label = QLabel('')
        self.min_safe_z_status_label.setStyleSheet('color: #aaa; font-size: 11px;')
        self.min_safe_z_status_label.setWordWrap(True)
        safety_layout.addWidget(self.min_safe_z_status_label)

        # ── 검출 임계 / 카메라 노출 조정 (운전 탭, 물체 선택 위) ─────────
        # confidence는 object_detector의 conf_thresh를 라이브 변경.
        # 노출은 RealSense rgb_camera 파라미터를 라이브 변경 (auto OFF 후 수동 값 설정).
        detect_tune_group = QGroupBox('검출/노출 조정')
        detect_tune_layout = QVBoxLayout(detect_tune_group)
        detect_tune_layout.setContentsMargins(8, 6, 8, 6)
        detect_tune_layout.setSpacing(4)

        # 신뢰도 임계 행
        conf_row = QHBoxLayout()
        conf_label = QLabel('신뢰도:')
        conf_label.setFixedWidth(56)
        # QDoubleSpinBox로 0.05~0.95 (0.01 step). 슬라이더는 정수만 가능해 100배 스케일.
        self.conf_thresh_slider = QSlider(Qt.Horizontal)
        self.conf_thresh_slider.setRange(5, 95)
        self.conf_thresh_slider.setValue(50)
        self.conf_thresh_spin = QDoubleSpinBox()
        self.conf_thresh_spin.setRange(0.05, 0.95)
        self.conf_thresh_spin.setSingleStep(0.05)
        self.conf_thresh_spin.setDecimals(2)
        self.conf_thresh_spin.setValue(0.50)
        self.conf_thresh_spin.setFixedWidth(64)
        # 슬라이더 ↔ 스핀 양방향 동기화 (스케일 변환)
        self.conf_thresh_slider.valueChanged.connect(
            lambda v: self.conf_thresh_spin.setValue(v / 100.0))
        self.conf_thresh_spin.valueChanged.connect(
            lambda v: self.conf_thresh_slider.setValue(int(round(v * 100))))
        self.conf_apply_button = QPushButton('적용')
        self.conf_apply_button.setFixedSize(48, 24)
        self.conf_apply_button.clicked.connect(self._confidence_apply)
        conf_row.addWidget(conf_label)
        conf_row.addWidget(self.conf_thresh_slider)
        conf_row.addWidget(self.conf_thresh_spin)
        conf_row.addWidget(self.conf_apply_button)
        detect_tune_layout.addLayout(conf_row)

        # 자동노출 토글
        auto_exp_row = QHBoxLayout()
        auto_exp_label = QLabel('자동노출:')
        auto_exp_label.setFixedWidth(56)
        self.auto_exposure_check = QCheckBox('ON')
        self.auto_exposure_check.setChecked(True)
        # 토글 시 즉시 적용 (set_parameters 호출). 매번 [적용] 버튼 없이 켜고 끄기 직관적.
        self.auto_exposure_check.stateChanged.connect(self._exposure_auto_toggle)
        auto_exp_row.addWidget(auto_exp_label)
        auto_exp_row.addWidget(self.auto_exposure_check)
        auto_exp_row.addStretch(1)
        detect_tune_layout.addLayout(auto_exp_row)

        # 수동 노출 행 (자동노출 OFF일 때만 활성)
        exp_row = QHBoxLayout()
        exp_label = QLabel('수동노출:')
        exp_label.setFixedWidth(56)
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(20, 5000)    # μs 단위. RealSense 일반 범위
        self.exposure_slider.setValue(80)
        self.exposure_spin = QSpinBox()
        self.exposure_spin.setRange(20, 5000)
        self.exposure_spin.setValue(80)
        self.exposure_spin.setSuffix(' μs')
        self.exposure_spin.setFixedWidth(80)
        self.exposure_slider.valueChanged.connect(self.exposure_spin.setValue)
        self.exposure_spin.valueChanged.connect(self.exposure_slider.setValue)
        self.exposure_apply_button = QPushButton('적용')
        self.exposure_apply_button.setFixedSize(48, 24)
        self.exposure_apply_button.clicked.connect(self._exposure_apply)
        exp_row.addWidget(exp_label)
        exp_row.addWidget(self.exposure_slider)
        exp_row.addWidget(self.exposure_spin)
        exp_row.addWidget(self.exposure_apply_button)
        detect_tune_layout.addLayout(exp_row)

        # 상태 라벨 (적용 결과 표시)
        self.detect_tune_status = QLabel('')
        self.detect_tune_status.setStyleSheet('color: #aaa; font-size: 11px;')
        self.detect_tune_status.setWordWrap(True)
        detect_tune_layout.addWidget(self.detect_tune_status)

        object_group = QGroupBox('검출된 물체 선택')
        object_layout = QVBoxLayout(object_group)
        self.auto_button = QPushButton('자동 선택 사용')
        self.auto_button.clicked.connect(lambda: self._select_label(''))
        object_layout.addWidget(self.auto_button)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_container = QWidget()
        self.button_grid = QGridLayout(scroll_container)
        scroll.setWidget(scroll_container)
        object_layout.addWidget(scroll)

        self.object_summary = QLabel('검출된 물체가 없습니다.')
        self.object_summary.setWordWrap(True)
        object_layout.addWidget(self.object_summary)

        # 탭1 "운전" 단일 컬럼: 긴급 제어 → 검출/노출 조정 → 물체 선택 → 실시간 전류
        # (상태창은 카메라와 그래프 사이의 cam_col로 이동 — 모든 탭에서 상시 노출)
        op_right.addWidget(emergency_group)
        op_right.addWidget(detect_tune_group)
        op_right.addWidget(object_group)
        op_right.addStretch(1)

        # 탭2 "gripper": 그리퍼 동작 + 정밀 전류 제어 + 물체별 강도
        grip_col.addWidget(gripper_action_group)
        grip_col.addWidget(self.gripper_ctrl_group)
        grip_col.addWidget(self.grip_strength_group)
        grip_col.addStretch(1)

        # 탭3 "수동·설정": 로봇 수동 제어 + 안전 모드 (+ 모델·캘리브은 위에서 추가됨)
        set_col.addWidget(robot_ctrl_group)
        set_col.addWidget(safety_group)
        set_col.addStretch(1)

        self.tabs.addTab(_tab_op, '운전')
        self.tabs.addTab(_tab_grip, 'gripper')
        self.tabs.addTab(_tab_set, '수동·설정')

        # 카메라 영상은 어느 탭에서나 항상 보이도록 본문 좌측에 고정, 탭은 우측에 배치
        body = QHBoxLayout()
        cam_col = QVBoxLayout()
        cam_col.addWidget(self.image_label)
        # status_group은 좌상단 activity 라벨과 중복 → 숨김(setVisible False)만 한다.
        # ★ 레이아웃에서 빼면(addWidget 안 하면) 자식 라벨(state_label 등)이 C++에서 삭제돼
        #   _update_ui의 setText가 RuntimeError로 죽고 UI가 통째로 멈춘다. 반드시 객체는 유지.
        status_group.setVisible(False)
        cam_col.addWidget(status_group)
        # 그 자리(카메라-그래프 사이)에 실시간 전류/위치 한 줄.
        cam_col.addWidget(self.gripper_status_label)
        # 실시간 전류 그래프: 전류값 바로 아래, 모든 탭 상시(가로 길게)
        cam_col.addWidget(self.realtime_graph)
        body.addLayout(cam_col, 2)
        body.addWidget(self.tabs, 3)
        root.addLayout(body)

        # 스크린샷 등으로 포커스된 버튼이 키 입력에 잘못 눌리는 것 방지(서보 OFF 사고 등).
        # 모든 버튼을 키보드 비포커스로 만들어 마우스 클릭으로만 동작하게 한다.
        for _btn in self.findChildren(QPushButton):
            _btn.setFocusPolicy(Qt.NoFocus)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_ui)
        self.timer.start(100)

    def _select_label(self, label: str):
        self.ros_node.publish_selected_label(label)
        if label:
            # 그리퍼 미준비면 사전 안내 후에도 서비스는 그대로 호출 — pick_place_node가 reject하면 사용자에게 응답 메시지로 통지됨.
            if not self.ros_node.gripper_hw_ready:
                self.ros_node.get_logger().warn('그리퍼 준비 미완료 — run_once 요청은 거절될 수 있습니다.')
            self.ros_node.call_trigger_service(self.ros_node.cli_run_once, 'pick_place/run_once')

    def _recover_to_home(self):
        self._call_manual_command(
            key='recover_to_home',
            client=self.ros_node.cli_recover_to_home,
            service_label='pick_place/recover_to_home',
            progress_text='에러 복구 및 HOME 복귀 중...',
            done_text='에러 복구 및 HOME 복귀 완료',
            timeout_sec=45.0,
            wait_for_state=True,
        )

    def _go_home(self):
        self._call_manual_command(
            key='home',
            client=self.ros_node.cli_go_home,
            service_label='pick_place/go_home',
            progress_text='HOME 이동 중...',
            done_text='HOME 이동 완료',
            timeout_sec=45.0,
            wait_for_state=True,
        )

    def _gripper_open(self):
        self._call_manual_command(
            key='gripper_open',
            client=self.ros_node.cli_gripper_open,
            service_label='gripper/open',
            progress_text='그리퍼 OPEN 중...',
            done_text='그리퍼 OPEN 완료',
            timeout_sec=25.0,
            wait_for_state=False,
            min_busy_sec=self._gripper_feedback_hold_sec,
        )

    def _gripper_close(self):
        self._call_manual_command(
            key='gripper_close',
            client=self.ros_node.cli_gripper_close,
            service_label='gripper/close',
            progress_text='그리퍼 CLOSE 중...',
            done_text='그리퍼 CLOSE 완료',
            timeout_sec=25.0,
            wait_for_state=False,
            min_busy_sec=self._gripper_feedback_hold_sec,
        )

    def _gripper_torque_on(self):
        self._call_gripper_torque(True)

    def _gripper_torque_off(self):
        self._call_gripper_torque(False)

    def _call_gripper_torque(self, enable: bool):
        """/gripper/enable(SetBool)로 토크 on/off. _call_manual_command는 Trigger
        전용이라 SetBool은 직접 호출한다."""
        label = '토크 ON' if enable else '토크 OFF'
        client = self.ros_node.cli_gripper_enable
        if not client.service_is_ready():
            self.ros_node.get_logger().warn('서비스 미연결: gripper/enable')
            self._set_manual_feedback(f'{label} 실패: 서비스 미연결')
            return
        self._set_manual_feedback(f'{label} 중...')
        req = SetBool.Request()
        req.data = bool(enable)
        future = client.call_async(req)

        def _done(f):
            try:
                res = f.result()
                if res.success:
                    self._set_manual_feedback(f'{label} 완료')
                else:
                    self._set_manual_feedback(f'{label} 거절: {res.message}')
            except Exception as e:
                self.ros_node.get_logger().error(f'gripper/enable 호출 실패: {e}')
                self._set_manual_feedback(f'{label} 실패')

        future.add_done_callback(_done)

    def _gripper_apply(self):
        # 중복 클릭 방지: 이전 요청이 완료되기 전에는 재진입 불가
        if getattr(self, '_gripper_apply_busy', False):
            return
        cli = self.ros_node.cli_gripper_set_parameters
        if not cli.service_is_ready():
            self.ros_node.get_logger().warn('그리퍼 set_parameters 서비스 미연결')
            return

        self._gripper_apply_busy = True
        self.gripper_apply_button.setEnabled(False)

        req = SetParameters.Request()

        params = [
            ('open_current', ParameterType.PARAMETER_INTEGER, int(self.open_curr_spin.value())),
            ('close_current', ParameterType.PARAMETER_INTEGER, int(self.close_curr_spin.value())),
            ('transport_current', ParameterType.PARAMETER_INTEGER, int(self.transport_curr_spin.value())),
            ('profile_velocity', ParameterType.PARAMETER_INTEGER, int(self.vel_spin.value())),
            ('profile_acceleration', ParameterType.PARAMETER_INTEGER, int(self.acc_spin.value()))
        ]

        for name, p_type, val in params:
            rp = RclParameter()
            rp.name = name
            rp.value = ParameterValue()
            rp.value.type = p_type
            rp.value.integer_value = val
            req.parameters.append(rp)

        future = cli.call_async(req)
        future.add_done_callback(self._on_gripper_applied)

    def _on_gripper_applied(self, future):
        # rclpy 콜백 스레드에서 호출됨. Qt GUI는 _update_ui에서 안전하게 복원한다.
        self._gripper_apply_busy = False
        try:
            results = future.result().results
            ok = bool(results) and all(result.successful for result in results)
        except Exception as e:
            self.ros_node.get_logger().error(f'그리퍼 파라미터 적용 실패: {e}')
            return
        if ok:
            self.ros_node.get_logger().info('그리퍼 정밀 제어 파라미터 적용 완료.')
        else:
            reason = next((r.reason for r in results if not r.successful), '')
            self.ros_node.get_logger().warn(f'그리퍼 파라미터 적용 거절: {reason}')

    # ── 물체별 파지 강도 ────────────────────────────────────────────────
    def _find_params_yaml(self) -> Path | None:
        """config/pick_place_params.yaml 경로를 후보 루트에서 찾는다."""
        rel = Path('config') / 'pick_place_params.yaml'
        for root in self.ros_node._candidate_search_roots():
            cand = root / rel
            if cand.is_file():
                return cand.resolve()
        return None

    def _load_grip_strength_config(self):
        """yaml에서 (names, currents, default, min, max)를 읽는다.
        실패 시 안전한 기본값으로 폴백한다."""
        names = ['doll', 'cup', 'pencil', 'tape', 'pack']
        currents = [250, 250, 200, 280, 350]
        default, cmin, cmax = 300, 100, 500
        path = self._find_params_yaml()
        if path is None:
            return names, currents, default, cmin, cmax
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            pp = data.get('pick_place_node', {}).get('ros__parameters', {})
            n = pp.get('grip_class_names', names)
            c = pp.get('grip_class_currents', currents)
            if isinstance(n, list) and isinstance(c, list) and len(n) == len(c) and n:
                names = [str(x) for x in n]
                currents = [int(x) for x in c]
            default = int(pp.get('grip_current_default', default))
            cmin = int(pp.get('grip_current_min', cmin))
            cmax = int(pp.get('grip_current_max', cmax))
        except Exception as e:
            self.ros_node.get_logger().warn(f'grip 강도 config 읽기 실패(기본값 사용): {e}')
        return names, currents, default, cmin, cmax

    def _collect_grip_strength(self):
        """현재 슬라이더 값을 (names, currents, default)로 모은다 (clamp 적용)."""
        names, currents = [], []
        for cls, (_slider, spin) in self._grip_strength_rows.items():
            val = max(self._grip_cmin, min(self._grip_cmax, int(spin.value())))
            if cls == '__default__':
                default = val
            else:
                names.append(cls)
                currents.append(val)
        return names, currents, default

    def _grip_strength_apply(self):
        """슬라이더 값을 pick_place_node 파라미터로 라이브 적용하고,
        detector의 known_classes도 같은 names로 동기화한다 (라벨↔강도 일관성)."""
        cli = self.ros_node.cli_pickplace_set_parameters
        if not cli.service_is_ready():
            self.grip_strength_status_label.setText('⚠ pick_place set_parameters 서비스 미연결')
            return
        names, currents, default = self._collect_grip_strength()

        req = SetParameters.Request()
        # grip_class_names (STRING_ARRAY)
        p_names = RclParameter()
        p_names.name = 'grip_class_names'
        p_names.value = ParameterValue(type=ParameterType.PARAMETER_STRING_ARRAY,
                                       string_array_value=names)
        # grip_class_currents (INTEGER_ARRAY)
        p_curr = RclParameter()
        p_curr.name = 'grip_class_currents'
        p_curr.value = ParameterValue(type=ParameterType.PARAMETER_INTEGER_ARRAY,
                                      integer_array_value=currents)
        # grip_current_default (INTEGER)
        p_def = RclParameter()
        p_def.name = 'grip_current_default'
        p_def.value = ParameterValue(type=ParameterType.PARAMETER_INTEGER,
                                     integer_value=int(default))
        req.parameters = [p_names, p_curr, p_def]

        # 동시에 object_detector의 known_classes도 동일 names로 push (라벨 일관성 보장)
        det_cli = self.ros_node.cli_object_set_parameters
        if det_cli.service_is_ready():
            det_req = SetParameters.Request()
            p_known = RclParameter()
            p_known.name = 'known_classes'
            p_known.value = ParameterValue(type=ParameterType.PARAMETER_STRING_ARRAY,
                                            string_array_value=names)
            det_req.parameters = [p_known]
            det_future = det_cli.call_async(det_req)
            det_future.add_done_callback(self._on_known_classes_synced)
        else:
            self.ros_node.get_logger().warn(
                '⚠ object_detector set_parameters 미연결 — known_classes 동기화 못 함. '
                'detector 재시작 후 다음 launch 때 yaml에서 읽힘.')

        future = cli.call_async(req)
        future.add_done_callback(self._on_grip_strength_applied)
        self.grip_strength_status_label.setText('적용 중...')

    def _on_grip_strength_applied(self, future):
        try:
            results = future.result().results
            ok = bool(results) and all(r.successful for r in results)
        except Exception as e:
            self.ros_node.get_logger().error(f'물체별 강도 적용 실패: {e}')
            self.grip_strength_status_label.setText(f'⚠ 적용 실패: {e}')
            return
        if ok:
            self.ros_node.get_logger().info('물체별 파지 강도 적용 완료.')
            self.grip_strength_status_label.setText('✅ 적용 완료 (저장하지 않으면 재시작 시 초기화)')
        else:
            reason = next((r.reason for r in results if not r.successful), '')
            self.ros_node.get_logger().warn(f'물체별 강도 적용 거절: {reason}')
            self.grip_strength_status_label.setText(f'⚠ 거절: {reason}')

    def _on_known_classes_synced(self, future):
        """detector의 known_classes set_parameters 응답 처리(보조)."""
        try:
            results = future.result().results
            ok = bool(results) and all(r.successful for r in results)
        except Exception as e:
            self.ros_node.get_logger().warn(f'⚠ detector known_classes 동기화 실패: {e}')
            return
        if ok:
            self.ros_node.get_logger().info('object_detector.known_classes 동기화 완료.')
        else:
            reason = next((r.reason for r in results if not r.successful), '')
            self.ros_node.get_logger().warn(f'⚠ detector known_classes 거절: {reason}')

    def _grip_strength_save(self):
        """현재 슬라이더 값을 yaml 파일의 해당 라인만 교체해 저장(주석 보존).
        라벨↔강도 일관성을 위해 object_detector 섹션의 known_classes도 동시에 갱신."""
        path = self._find_params_yaml()
        if path is None:
            self.grip_strength_status_label.setText('⚠ config yaml 파일을 찾지 못함')
            return
        names, currents, default = self._collect_grip_strength()
        names_str = '[' + ', '.join(f'"{n}"' for n in names) + ']'
        curr_str = '[' + ', '.join(str(c) for c in currents) + ']'
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            # 들여쓰기를 보존하며 키 라인만 값 교체 (인라인 주석은 제거됨)
            # known_classes도 같이 갱신 — 라벨↔강도 일관성 보장(단일 yaml 편집으로 둘 다 sync)
            patterns = {
                'grip_class_names': names_str,
                'grip_class_currents': curr_str,
                'grip_current_default': str(int(default)),
                'known_classes': names_str,
            }
            replaced = {k: False for k in patterns}
            for i, line in enumerate(lines):
                for key, new_val in patterns.items():
                    m = re.match(rf'^(\s*){key}\s*:', line)
                    if m and not replaced[key]:
                        lines[i] = f'{m.group(1)}{key}: {new_val}\n'
                        replaced[key] = True
            # known_classes는 옵셔널(이전 yaml에 없을 수 있음) → 누락 허용
            optional = {'known_classes'}
            missing = [k for k, v in replaced.items() if not v and k not in optional]
            if missing:
                self.grip_strength_status_label.setText(f'⚠ yaml에서 필수 키 누락: {missing}')
                return
            with open(path, 'w') as f:
                f.writelines(lines)
            extras = '' if replaced['known_classes'] else ' (known_classes 라인 없음 — 수동 추가 필요)'
            self.ros_node.get_logger().info(f'물체별 파지 강도 yaml 저장: {path}{extras}')
            self.grip_strength_status_label.setText(f'💾 저장 완료: {path.name}{extras}')
        except Exception as e:
            self.ros_node.get_logger().error(f'yaml 저장 실패: {e}')
            self.grip_strength_status_label.setText(f'⚠ 저장 실패: {e}')

    def _gripper_ctrl_save(self):
        """그리퍼 정밀 전류/속도 슬라이더 값을 yaml 해당 라인만 교체해 저장(값만 바꾸고 주석 보존).
        GUI 튜닝을 yaml에 영구 반영 → yaml 파일 공유로 다른 PC에서 동일 튜닝 재현."""
        path = self._find_params_yaml()
        if path is None:
            self.ros_node.get_logger().warn('⚠ config yaml 파일을 찾지 못함 (그리퍼 전류 저장)')
            return
        patterns = {
            'open_current': str(int(self.open_curr_spin.value())),
            'close_current': str(int(self.close_curr_spin.value())),
            'transport_current': str(int(self.transport_curr_spin.value())),
            'profile_velocity': str(int(self.vel_spin.value())),
            'profile_acceleration': str(int(self.acc_spin.value())),
        }
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            replaced = {k: False for k in patterns}
            for i, line in enumerate(lines):
                for key, new_val in patterns.items():
                    # 값만 교체하고 인라인 주석(#...)은 보존
                    m = re.match(rf'^(\s*){key}\s*:\s*[^#\n]*?(\s*#.*)?$', line)
                    if m and not replaced[key]:
                        comment = m.group(2) or ''
                        lines[i] = f'{m.group(1)}{key}: {new_val}{comment}\n'
                        replaced[key] = True
            missing = [k for k, v in replaced.items() if not v]
            if missing:
                self.ros_node.get_logger().warn(f'⚠ 그리퍼 전류 yaml 저장 — 키 누락: {missing}')
                return
            with open(path, 'w') as f:
                f.writelines(lines)
            self.ros_node.get_logger().info(f'💾 그리퍼 전류/프로파일 yaml 저장 완료: {path}')
        except Exception as e:
            self.ros_node.get_logger().error(f'그리퍼 전류 yaml 저장 실패: {e}')

    def _gripper_ctrl_load(self):
        """yaml의 그리퍼 전류/속도 값을 슬라이더로 불러온다(현재 파일값으로 복원)."""
        path = self._find_params_yaml()
        if path is None:
            self.ros_node.get_logger().warn('⚠ config yaml 파일을 찾지 못함 (그리퍼 전류 불러오기)')
            return
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
            p = (data or {}).get('rh_p12_rna_gripper', {}).get('ros__parameters', {})
            mapping = {
                'open_current': self.open_curr_spin,
                'close_current': self.close_curr_spin,
                'transport_current': self.transport_curr_spin,
                'profile_velocity': self.vel_spin,
                'profile_acceleration': self.acc_spin,
            }
            loaded = [k for k, spin in mapping.items() if k in p and (spin.setValue(int(p[k])) or True)]
            self.ros_node.get_logger().info(f'📂 그리퍼 전류 yaml 불러오기: {loaded}')
        except Exception as e:
            self.ros_node.get_logger().error(f'그리퍼 전류 yaml 불러오기 실패: {e}')

    # ── TCP Z 안전 하한 (min_safe_z) ────────────────────────────────────
    def _load_min_safe_z(self) -> float:
        """yaml에서 pick_place_node.min_safe_z를 읽는다. 실패 시 0.0."""
        path = self._find_params_yaml()
        if path is None:
            return 0.0
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            pp = data.get('pick_place_node', {}).get('ros__parameters', {})
            return float(pp.get('min_safe_z', 0.0))
        except Exception as e:
            self.ros_node.get_logger().warn(f'min_safe_z config 읽기 실패(0.0 사용): {e}')
            return 0.0

    def _min_safe_z_apply(self):
        """스핀박스 값을 pick_place_node.min_safe_z로 라이브 적용."""
        cli = self.ros_node.cli_pickplace_set_parameters
        if not cli.service_is_ready():
            self.min_safe_z_status_label.setText('⚠ pick_place set_parameters 서비스 미연결')
            return
        val = float(self.min_safe_z_spin.value())
        req = SetParameters.Request()
        p = RclParameter()
        p.name = 'min_safe_z'
        p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=val)
        req.parameters = [p]
        future = cli.call_async(req)
        future.add_done_callback(self._on_min_safe_z_applied)
        self.min_safe_z_status_label.setText('적용 중...')

    def _on_min_safe_z_applied(self, future):
        try:
            results = future.result().results
            ok = bool(results) and all(r.successful for r in results)
        except Exception as e:
            self.ros_node.get_logger().error(f'min_safe_z 적용 실패: {e}')
            self.min_safe_z_status_label.setText(f'⚠ 적용 실패: {e}')
            return
        if ok:
            self.ros_node.get_logger().info('TCP Z 안전 하한(min_safe_z) 적용 완료.')
            self.min_safe_z_status_label.setText('✅ 적용 완료 (저장하지 않으면 재시작 시 초기화)')
        else:
            reason = next((r.reason for r in results if not r.successful), '')
            self.ros_node.get_logger().warn(f'min_safe_z 적용 거절: {reason}')
            self.min_safe_z_status_label.setText(f'⚠ 거절: {reason}')

    def _min_safe_z_save(self):
        """현재 스핀박스 값을 yaml의 min_safe_z 라인만 교체해 저장(주석 보존)."""
        path = self._find_params_yaml()
        if path is None:
            self.min_safe_z_status_label.setText('⚠ config yaml 파일을 찾지 못함')
            return
        val = float(self.min_safe_z_spin.value())
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            replaced = False
            for i, line in enumerate(lines):
                m = re.match(r'^(\s*)min_safe_z\s*:', line)
                if m:
                    lines[i] = f'{m.group(1)}min_safe_z: {val:.3f}\n'
                    replaced = True
                    break
            if not replaced:
                self.min_safe_z_status_label.setText('⚠ yaml에서 min_safe_z 키를 찾지 못함')
                return
            with open(path, 'w') as f:
                f.writelines(lines)
            self.ros_node.get_logger().info(f'min_safe_z yaml 저장: {path} = {val:.3f}')
            self.min_safe_z_status_label.setText(f'💾 저장 완료: {val:.3f} m')
        except Exception as e:
            self.ros_node.get_logger().error(f'min_safe_z yaml 저장 실패: {e}')
            self.min_safe_z_status_label.setText(f'⚠ 저장 실패: {e}')

    # ── 검출 임계 / 노출 (운전 탭) ───────────────────────────────────
    def _confidence_apply(self):
        """object_detector의 confidence_threshold를 라이브 변경."""
        cli = self.ros_node.cli_object_set_parameters
        if not cli.service_is_ready():
            self.detect_tune_status.setText('⚠ object_detector set_parameters 서비스 미연결')
            return
        val = float(self.conf_thresh_spin.value())
        req = SetParameters.Request()
        p = RclParameter()
        p.name = 'confidence_threshold'
        p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=val)
        req.parameters = [p]
        future = cli.call_async(req)
        future.add_done_callback(
            lambda f: self._on_param_apply_done(f, f'신뢰도 임계 → {val:.2f}'))
        self.detect_tune_status.setText(f'신뢰도 적용 중 ({val:.2f})...')

    def _exposure_auto_toggle(self, state):
        """rgb_camera.enable_auto_exposure 라이브 토글. 켜진 동안엔 수동 슬라이더 비활성화."""
        enable = bool(state)
        cli = self.ros_node.cli_camera_set_parameters
        if not cli.service_is_ready():
            self.detect_tune_status.setText('⚠ camera set_parameters 서비스 미연결')
            return
        req = SetParameters.Request()
        p = RclParameter()
        p.name = 'rgb_camera.enable_auto_exposure'
        p.value = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=enable)
        req.parameters = [p]
        future = cli.call_async(req)
        future.add_done_callback(
            lambda f: self._on_param_apply_done(f, f'자동노출 → {"ON" if enable else "OFF"}'))
        self.detect_tune_status.setText(f'자동노출 {"ON" if enable else "OFF"} 적용 중...')

    def _exposure_apply(self):
        """수동 노출(rgb_camera.exposure, μs) 라이브 적용. 자동노출 OFF 상태일 때만 효과."""
        cli = self.ros_node.cli_camera_set_parameters
        if not cli.service_is_ready():
            self.detect_tune_status.setText('⚠ camera set_parameters 서비스 미연결')
            return
        val = int(self.exposure_spin.value())
        req = SetParameters.Request()
        p = RclParameter()
        p.name = 'rgb_camera.exposure'
        p.value = ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=val)
        req.parameters = [p]
        future = cli.call_async(req)
        future.add_done_callback(
            lambda f: self._on_param_apply_done(f, f'수동 노출 → {val} μs'))
        if self.auto_exposure_check.isChecked():
            self.detect_tune_status.setText(
                f'노출 {val} μs 적용 중... (※ 자동노출 ON 상태라 무시될 수 있음)')
        else:
            self.detect_tune_status.setText(f'노출 {val} μs 적용 중...')

    def _on_param_apply_done(self, future, label: str):
        """공통 응답 처리 — 적용 결과를 detect_tune_status에 표시."""
        try:
            results = future.result().results
            ok = bool(results) and all(r.successful for r in results)
        except Exception as e:
            self.ros_node.get_logger().error(f'{label} 실패: {e}')
            self.detect_tune_status.setText(f'⚠ {label} 실패: {e}')
            return
        if ok:
            self.ros_node.get_logger().info(f'{label} 적용 완료')
            self.detect_tune_status.setText(f'✅ {label} 적용 완료')
        else:
            reason = next((r.reason for r in results if not r.successful), '')
            self.ros_node.get_logger().warn(f'{label} 거절: {reason}')
            self.detect_tune_status.setText(f'⚠ {label} 거절: {reason}')

    def _call_manual_command(
        self,
        key: str,
        client,
        service_label: str,
        progress_text: str,
        done_text: str,
        timeout_sec: float,
        wait_for_state: bool,
        min_busy_sec: float = 0.0,
    ):
        if self._manual_command is not None:
            self._set_manual_feedback(f'{self._manual_command["progress_text"]} 이미 진행 중')
            return
        if not client.service_is_ready():
            self.ros_node.get_logger().warn(f'서비스 미연결: {service_label}')
            self._set_manual_feedback(f'{progress_text} 실패: 서비스 미연결')
            return

        self._manual_command = {
            'key': key,
            'service_label': service_label,
            'progress_text': progress_text,
            'done_text': done_text,
            'wait_for_state': wait_for_state,
            'accepted': False,
            'min_busy_until': time.monotonic() + float(min_busy_sec),
        }
        self._manual_command_seen_active = False
        self._manual_command_deadline = time.monotonic() + timeout_sec
        self._manual_command_token += 1
        token = self._manual_command_token

        future = client.call_async(Trigger.Request())

        def _on_done(done_future):
            if token != self._manual_command_token:
                return
            try:
                res = done_future.result()
                status = '성공' if res.success else '거절'
                self.ros_node.get_logger().info(f'{service_label}: {status} - {res.message}')
            except Exception as e:
                self.ros_node.get_logger().error(f'{service_label} 호출 실패: {e}')
                self._finish_manual_command(f'{progress_text} 실패')
                return

            if not res.success:
                self._finish_manual_command(f'{progress_text} 거절: {res.message}')
                return

            if wait_for_state or min_busy_sec > 0.0:
                if self._manual_command is not None and self._manual_command.get('key') == key:
                    self._manual_command['accepted'] = True
                return

            self._finish_manual_command(done_text)

        future.add_done_callback(_on_done)

    def _set_manual_feedback(self, text: str, duration: float = 2.0):
        self._manual_feedback = text
        self._manual_feedback_until = time.monotonic() + duration

    def _finish_manual_command(self, feedback: str = ''):
        self._manual_command_token += 1
        self._manual_command = None
        self._manual_command_seen_active = False
        self._manual_command_deadline = 0.0
        if feedback:
            self._set_manual_feedback(feedback)

    def _clear_manual_command_feedback(self):
        self._manual_command_token += 1
        self._manual_command = None
        self._manual_command_seen_active = False
        self._manual_command_deadline = 0.0
        self._manual_feedback = ''
        self._manual_feedback_until = 0.0


    def _e_stop(self):
        self._clear_manual_command_feedback()
        self.ros_node.publish_selected_label('')
        self.ros_node.call_trigger_service(self.ros_node.cli_e_stop, 'pick_place/e_stop')

    def _cancel_task(self):
        self._clear_manual_command_feedback()
        self.ros_node.publish_selected_label('')
        self.ros_node.call_trigger_service(self.ros_node.cli_cancel, 'pick_place/cancel')

    def _e_stop_reset(self):
        self._reset_in_progress = True
        self._reset_deadline = time.monotonic() + 20.0
        self.e_stop_reset_button.setEnabled(False)
        self.e_stop_reset_button.setText('리셋 중...')

        def _on_done(future):
            try:
                res = future.result()
                status = '성공' if res.success else '거절'
                self.ros_node.get_logger().info(f'pick_place/e_stop_reset: {status} - {res.message}')
                if not res.success:
                    self._reset_in_progress = False
            except Exception as e:
                self.ros_node.get_logger().error(f'e_stop_reset 호출 실패: {e}')
                self._reset_in_progress = False

        if not self.ros_node.cli_e_stop_reset.service_is_ready():
            self.ros_node.get_logger().warn('서비스 미연결: pick_place/e_stop_reset')
            self._reset_in_progress = False
            self.e_stop_reset_button.setText('긴급정지 해제')
            return

        future = self.ros_node.cli_e_stop_reset.call_async(Trigger.Request())
        future.add_done_callback(_on_done)

    def _clear_error(self):
        self.ros_node.call_trigger_service(self.ros_node.cli_clear_error, 'pick_place/clear_error')

    def _speed_normal(self):
        self.ros_node.call_trigger_service(self.ros_node.cli_speed_normal, 'pick_place/speed_normal')

    def _speed_reduced(self):
        self.ros_node.call_trigger_service(self.ros_node.cli_speed_reduced, 'pick_place/speed_reduced')

    def _servo_off(self):
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.warning(
            self, '서보 OFF 확인',
            '모든 관절 모터 전원을 차단합니다.\n로봇이 중력에 의해 움직일 수 있습니다.\n\n계속하시겠습니까?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.ros_node.call_trigger_service(self.ros_node.cli_servo_off, 'pick_place/servo_off')

    def _servo_on(self):
        self.ros_node.call_trigger_service(self.ros_node.cli_servo_on, 'pick_place/servo_on')

    def _safety_normal(self):
        self.ros_node.call_trigger_service(self.ros_node.cli_safety_normal, 'pick_place/safety_normal')

    def _safety_backdrive(self):
        self.ros_node.call_trigger_service(self.ros_node.cli_safety_backdrive, 'pick_place/safety_backdrive')

    def _gripper_bridge_restart(self):
        """그리퍼 브릿지(gripper_service_node + gripper_node)만 새 프로세스로 재기동한다.
        status3(Modbus 무응답)가 in-process reinit("그리퍼 리셋")으로 안 풀릴 때의 가벼운 복구.
        로봇/카메라는 안 건드리고 ~5-40초 안에 복구된다. 전원 사이클 불필요."""
        from ament_index_python.packages import get_package_share_directory

        proc = self._gripper_bridge_restart_proc
        if proc is not None and proc.poll() is None:
            return  # 이미 진행 중

        try:
            pkg_share = get_package_share_directory('dsr_realsense_pick_place')
            script = os.path.join(pkg_share, 'scripts', 'restart_gripper_bridge.sh')
            self._gripper_bridge_restart_proc = subprocess.Popen(
                ['bash', script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._gripper_bridge_restart_until = time.monotonic() + 45.0
            self.gripper_bridge_restart_button.setEnabled(False)
            self.gripper_bridge_restart_label.setText('⏳ 그리퍼 브릿지 재기동 중... (~5-40초, 그리퍼 ready 확인)')
            self.ros_node.get_logger().warn('🔧 그리퍼 브릿지 재기동 스크립트 실행 (status3 복구)')
        except Exception as e:
            self.ros_node.get_logger().error(f'그리퍼 브릿지 재기동 실패: {e}')
            self.gripper_bridge_restart_label.setText(f'❌ 실패: {e}')

    def _system_reset(self):
        """GUI를 제외한 모든 노드를 정상 종료 후 재시작한다.

        shutdown_nodes.sh → DRCF 해제 대기 → ros2 launch (gui:=false) 순서로 진행.
        subprocess 완료 여부는 _update_ui에서 QTimer로 폴링한다.
        """
        from ament_index_python.packages import get_package_share_directory

        if self._system_reset_phase:
            return  # 이미 진행 중

        self.system_reset_button.setEnabled(False)
        self.system_reset_label.setText('⏳ 노드 종료 중...')
        self._system_reset_phase = 'shutting_down'

        try:
            pkg_share = get_package_share_directory('dsr_realsense_pick_place')
            shutdown_script = os.path.join(pkg_share, 'scripts', 'shutdown_nodes.sh')
            self._system_reset_proc = subprocess.Popen(
                ['bash', shutdown_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self.ros_node.get_logger().error(f'시스템 리셋 실패: {e}')
            self.system_reset_label.setText(f'❌ 실패: {e}')
            self._system_reset_phase = ''
            self.system_reset_button.setEnabled(True)

    def _load_gui_settings(self) -> dict:
        try:
            if self._settings_path.is_file():
                with open(self._settings_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            self.ros_node.get_logger().warn(f'GUI 설정 불러오기 실패: {e}')
        return {}

    def _save_gui_settings(self):
        try:
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._settings_path, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.ros_node.get_logger().warn(f'GUI 설정 저장 실패: {e}')

    _CALIB_OFFSET_PARAMS = [
        'absolute_calib_x_mm',
        'absolute_calib_y_mm',
        'absolute_calib_z_mm',
    ]

    def _object_settings_param_names(self):
        return self._CALIB_OFFSET_PARAMS + ['yolo_model']

    def _maybe_load_object_settings(self):
        now = time.monotonic()
        if (
            self._object_settings_loaded
            or self._object_settings_loading
            or now - self._last_object_settings_attempt < 1.0
        ):
            return
        if not self.ros_node.cli_object_get_parameters.service_is_ready():
            return
        self._last_object_settings_attempt = now
        self._calib_load()

    def _maybe_apply_saved_model_path(self):
        if self._saved_model_applied:
            return
        model_path = str(self._settings.get('yolo_model_path', '')).strip()
        if not model_path or not self.ros_node.cli_object_set_parameters.service_is_ready():
            return
        self._saved_model_applied = True
        self._model_apply(save=False, silent=True)

    def _calib_load(self):
        cli = self.ros_node.cli_object_get_parameters
        if not cli.service_is_ready():
            self.ros_node.get_logger().warn('object_detector get_parameters 서비스 미연결')
            return
        self._object_settings_loading = True
        req = GetParameters.Request()
        req.names = self._object_settings_param_names()
        future = cli.call_async(req)
        future.add_done_callback(self._on_calib_loaded)

    def _on_calib_loaded(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.ros_node.get_logger().error(f'캘리브레이션 불러오기 실패: {e}')
            self._object_settings_loading = False
            return
        vals = [v.double_value for v in res.values[:3]]
        for i, spin in enumerate(self._calib_offset_spins):
            spin.blockSignals(True)
            spin.setValue(vals[i])
            spin.blockSignals(False)
        self._calib_current_mm = vals
        if len(res.values) >= 4:
            model_path = res.values[3].string_value
            if not self.model_path_edit.text().strip():
                self.model_path_edit.setText(model_path)
        self._object_settings_loaded = True
        self._object_settings_loading = False
        self._update_calib_current_label()
        self.ros_node.get_logger().info('object_detector 설정 불러오기 완료')

    def _calib_apply(self):
        if self.ros_node.pick_place_state != 'IDLE':
            self.ros_node.get_logger().warn('캘리브레이션 적용은 IDLE 상태에서만 가능합니다')
            return
        cli = self.ros_node.cli_object_set_parameters
        if not cli.service_is_ready():
            self.ros_node.get_logger().warn('object_detector set_parameters 서비스 미연결')
            return
        req = SetParameters.Request()
        vals = [s.value() for s in self._calib_offset_spins]
        for name, val in zip(self._CALIB_OFFSET_PARAMS, vals):
            rp = RclParameter()
            rp.name = name
            rp.value = ParameterValue()
            rp.value.type = ParameterType.PARAMETER_DOUBLE
            rp.value.double_value = float(val)
            req.parameters.append(rp)
        future = cli.call_async(req)
        future.add_done_callback(lambda f: self._on_calib_applied(f, vals))

    def _on_calib_applied(self, future, vals):
        try:
            results = future.result().results
            ok = bool(results) and all(result.successful for result in results)
        except Exception as e:
            self.ros_node.get_logger().error(f'캘리브레이션 적용 실패: {e}')
            return
        if ok:
            self._calib_current_mm = list(vals)
            self._update_calib_current_label()
            self.ros_node.get_logger().info('캘리브레이션 적용 완료')
        else:
            reason = next((r.reason for r in results if not r.successful), '')
            self.ros_node.get_logger().warn(f'캘리브레이션 적용 거절: {reason}')

    def _update_calib_current_label(self):
        vals = self._calib_current_mm
        for axis, value in zip(('X', 'Y', 'Z'), vals):
            text = '--.- mm' if value is None else f'{value:6.1f} mm'
            self.calib_current_labels[axis].setText(text)

    def _model_browse(self):
        start = self.model_path_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            'YOLO 모델 선택',
            start,
            'YOLO weights (*.pt);;All files (*)',
        )
        if path:
            self.model_path_edit.setText(path)
            self._model_path_edited()

    def _model_path_edited(self):
        path = self.model_path_edit.text().strip()
        self._settings['yolo_model_path'] = path
        self._save_gui_settings()

    def _model_apply(self, save: bool, silent: bool = False):
        path = self.model_path_edit.text().strip()
        if not path:
            if not silent:
                self.ros_node.get_logger().warn('모델 경로가 비어 있습니다')
            return
        if save:
            self._settings['yolo_model_path'] = path
            self._save_gui_settings()
        cli = self.ros_node.cli_object_set_parameters
        if not cli.service_is_ready():
            if not silent:
                self.ros_node.get_logger().warn('object_detector set_parameters 서비스 미연결')
            return
        req = SetParameters.Request()
        rp = RclParameter()
        rp.name = 'yolo_model'
        rp.value = ParameterValue()
        rp.value.type = ParameterType.PARAMETER_STRING
        rp.value.string_value = path
        req.parameters.append(rp)
        future = cli.call_async(req)
        future.add_done_callback(lambda f: self._on_model_applied(f, path, silent))

    def _on_model_applied(self, future, path: str, silent: bool):
        try:
            results = future.result().results
            ok = bool(results) and all(result.successful for result in results)
        except Exception as e:
            self.ros_node.get_logger().error(f'모델 경로 적용 실패: {e}')
            return
        if ok:
            if not silent:
                self.ros_node.get_logger().info(f'모델 경로 적용 완료: {path}')
        else:
            reason = next((r.reason for r in results if not r.successful), '')
            self.ros_node.get_logger().warn(f'모델 경로 적용 거절: {reason}')

    def _poll_system_reset(self):
        """시스템 리셋 진행 상태를 100ms 주기로 확인하고 단계별 처리를 수행한다."""
        if not self._system_reset_phase:
            return

        if self._system_reset_phase == 'shutting_down':
            if self._system_reset_proc and self._system_reset_proc.poll() is not None:
                # shutdown 완료 → launch 재시작
                self.system_reset_label.setText('🚀 노드 재시작 중...')
                self._system_reset_phase = 'restarting'
                try:
                    self._system_restart_proc = subprocess.Popen(
                        [
                            'ros2', 'launch',
                            'dsr_realsense_pick_place', 'pick_place.launch.py',
                            'mode:=real',
                            'host:=110.120.1.50',
                            'gui:=false',
                            'pre_cleanup:=false',
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as e:
                    self.ros_node.get_logger().error(f'재시작 실패: {e}')
                    self.system_reset_label.setText(f'❌ 재시작 실패: {e}')
                    self._system_reset_phase = ''
                    self.system_reset_button.setEnabled(True)

        elif self._system_reset_phase == 'restarting':
            # launch 프로세스는 계속 실행 중이 정상 — 10초 후 완료로 간주
            if self._system_reset_phase_until == 0.0:
                self._system_reset_phase_until = time.monotonic() + 10.0
            if time.monotonic() >= self._system_reset_phase_until:
                self.system_reset_label.setText('✅ 재시작 시작됨 (노드 기동 중...)')
                self._system_reset_phase = ''
                self._system_reset_phase_until = 0.0
                self.system_reset_button.setEnabled(True)

    def _update_activity_label(self):
        # status 점(색)만으론 "뭘 하는지" 모름 → 현재 동작/단계를 한 줄로. 그리퍼 초기화 중이면 우선 표시.
        st = self.ros_node.pick_place_state
        prog = self.ros_node.gripper_init_progress
        if not self.ros_node.gripper_hw_ready:
            # 그리퍼가 준비 안 됨 = 초기화 중. 진행 메시지(init_progress)가 끊겨도 "초기화 중" 유지.
            label = f'⚙️ 그리퍼 초기화 중 — {prog}' if prog else '⚙️ 그리퍼 초기화 중...'
            text, bg, fg = label, '#1a2a3a', '#cfe8ff'
        else:
            mp = {
                'IDLE':          ('🟢 대기 — 물체 선택 가능', '#0f2a0f', '#bfe8bf'),
                'HOME':          ('🏠 HOME 복귀 중...', '#1a2a3a', '#cfe8ff'),
                'DETECTING':     ('🔍 물체 검출 중...', '#1a2a3a', '#cfe8ff'),
                'PRE_PICK':      ('✋ 집기 준비 — 접근 중...', '#1a2a3a', '#cfe8ff'),
                'PICK':          ('✋ 물체 집는 중...', '#1a2a3a', '#cfe8ff'),
                'LIFT':          ('⬆ 들어올리는 중...', '#1a2a3a', '#cfe8ff'),
                'MOVE_TO_PLACE': ('→ 배치 위치로 이동 중...', '#1a2a3a', '#cfe8ff'),
                'PLACE':         ('🤝 내려놓는 중...', '#1a2a3a', '#cfe8ff'),
                'POST_PLACE':    ('🏠 복귀 중...', '#1a2a3a', '#cfe8ff'),
                'ERROR':         (f'🔴 에러 — {self.ros_node.last_error_text or "확인 필요"}', '#3a1010', '#ff9999'),
                'EMERGENCY_STOP':('🛑 긴급정지', '#3a1010', '#ff9999'),
            }
            text, bg, fg = mp.get(st, (f'• {st}', '#2a2a2a', '#cccccc'))
        self.activity_label.setText(text)
        self.activity_label.setStyleSheet(
            f'background-color: {bg}; color: {fg}; border-radius: 3px;'
            ' padding: 2px 8px; font-size: 12px; font-weight: bold;'
        )

    def _update_ui(self):
        # 시스템 리셋 진행 상태 폴링
        self._poll_system_reset()

        self.ros_node.refresh_system_status()
        self._maybe_load_object_settings()
        self._maybe_apply_saved_model_path()
        detected_snapshot = list(self.ros_node.detected_objects)

        # 카메라 영상은 최신 프레임이 있을 때만 갱신한다.
        if self.ros_node.latest_qimage is not None:
            pixmap = QPixmap.fromImage(self.ros_node.latest_qimage)
            self._draw_object_frames_on_pixmap(pixmap, detected_snapshot)
            scaled = pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
        self._update_system_status_bar()

        # 선택 라벨이 비어 있으면 자동 선택 상태로 표현한다.
        selected_text = self.ros_node.selected_label or '자동 선택'
        self.state_label.setText(f'Pick & Place 상태: {self.ros_node.pick_place_state}')
        # 에러 배너 — ERROR 상태에서만 '어떤 에러인지' 좌상단에 표시 (복구되면 자동 숨김)
        if self.ros_node.pick_place_state == 'ERROR' and self.ros_node.last_error_text:
            self.error_banner_label.setText(self.ros_node.last_error_text)
            self.error_banner_label.setVisible(True)
        else:
            self.error_banner_label.setVisible(False)
        self._update_activity_label()
        self.selection_label.setText(f'선택 물체: {selected_text}')
        self.selection_status_label.setText(self._build_selection_status())

        state        = self.ros_node.pick_place_state
        is_e_stopped = state == 'EMERGENCY_STOP'
        is_idle      = state == 'IDLE'
        is_active    = state not in ('IDLE', 'EMERGENCY_STOP')
        # cancel은 실제 픽 사이클 + HOME 이동 중에만 의미 있음. INITIALIZING/ERROR/BACKDRIVE 제외.
        is_in_cancelable_motion = state in (
            'DETECTING', 'PRE_PICK', 'PICK', 'LIFT',
            'MOVE_TO_PLACE', 'PLACE', 'POST_PLACE', 'HOME',
        )
        is_in_error  = state == 'ERROR'
        hw = self.ros_node.hw_state
        if self._reset_in_progress:
            if (not is_e_stopped and hw not in (6, 15)) or time.monotonic() > self._reset_deadline:
                self._reset_in_progress = False
        self._update_manual_command_feedback(state)

        # ── 서비스 / 하드웨어 연결 상태 (100ms 폴링) ─────────────────
        go_home_svc       = self.ros_node.cli_go_home.service_is_ready()
        recover_svc       = self.ros_node.cli_recover_to_home.service_is_ready()
        gripper_open_svc  = self.ros_node.cli_gripper_open.service_is_ready()
        gripper_close_svc = self.ros_node.cli_gripper_close.service_is_ready()
        pick_svc          = self.ros_node.cli_run_once.service_is_ready()
        speed_normal_svc  = self.ros_node.cli_speed_normal.service_is_ready()
        speed_reduced_svc = self.ros_node.cli_speed_reduced.service_is_ready()
        servo_off_svc     = self.ros_node.cli_servo_off.service_is_ready()
        servo_on_svc      = self.ros_node.cli_servo_on.service_is_ready()
        safety_normal_svc = self.ros_node.cli_safety_normal.service_is_ready()
        safety_bd_svc     = self.ros_node.cli_safety_backdrive.service_is_ready()
        e_stop_reset_svc  = self.ros_node.cli_e_stop_reset.service_is_ready()
        gripper_param_svc = self.ros_node.cli_gripper_set_parameters.service_is_ready()
        gripper_reinit_svc = self.ros_node.cli_gripper_reinit.service_is_ready()
        gripper_enable_svc = self.ros_node.cli_gripper_enable.service_is_ready()

        gripper_hw_ready  = self.ros_node.gripper_hw_ready
        # hw=-1: 미수신, 6: E-STOP, 15: NOT_READY → 이 세 상태에서는 수동 명령 불가
        hw_ok             = hw not in (-1, 6, 15)

        # ── 긴급 제어 버튼 ────────────────────────────────────────────
        # E-STOP: 항상 활성 (서비스 미연결이어도 클릭 가능해야 하는 최우선 안전 버튼)
        self.e_stop_button.setEnabled(True)
        self.cancel_button.setEnabled(is_in_cancelable_motion)
        # 에러 해제: ERROR 상태 + 서비스 연결 + 리셋 진행 중 아닐 때만.
        clear_error_svc = self.ros_node.cli_clear_error.service_is_ready()
        self.clear_error_button.setEnabled(
            is_in_error and clear_error_svc and not self._reset_in_progress
        )
        if self._reset_in_progress:
            self.e_stop_reset_button.setEnabled(False)
            self.e_stop_reset_button.setText('리셋 중...')
        else:
            self.e_stop_reset_button.setText('긴급정지 해제')
            # 긴급정지 해제: E-STOP 상태 + 서비스 연결 시에만 활성
            self.e_stop_reset_button.setEnabled(is_e_stopped and e_stop_reset_svc)

        # ── 수동 제어 버튼 ────────────────────────────────────────────
        # HW 준비 + 비정상 상태가 아닐 때 + 리셋 중이 아닐 때
        manual_enabled = (
            state in ('IDLE', 'DETECTING', 'ERROR')
            and not self._reset_in_progress
            and hw_ok
        )
        manual_busy     = self._manual_command is not None
        command_enabled = manual_enabled and not manual_busy
        # 설정·튜닝값(min_safe_z, 물체별 파지 강도, 그리퍼 정밀 전류) 편집 게이트.
        # 캘리브레이션과 동일하게 IDLE 전용 — 동작 중(DETECTING/PICK/LIFT/MOVE 등)엔 편집·적용 불가.
        config_edit_enabled = is_idle and not manual_busy and not self._reset_in_progress

        # HOME: pick_place 서비스 연결 필요
        self.home_button.setEnabled(command_enabled and go_home_svc)
        # 에러 복구 & HOME: recover 서비스 연결 필요
        self.recover_home_button.setEnabled(command_enabled and recover_svc)
        # 그리퍼 수동 조작: 그리퍼 HW 초기화 완료(INITIALIZE) + 각 서비스 연결 필요
        gripper_cmd_ok = command_enabled and gripper_hw_ready
        self.gripper_open_button.setEnabled(gripper_cmd_ok and gripper_open_svc)
        self.gripper_close_button.setEnabled(gripper_cmd_ok and gripper_close_svc)
        # [2026-06-09] 그리퍼 리셋 버튼 제거됨 — 여기서 참조하면 AttributeError로 _update_ui 전체가
        #   죽어 UI(전류/위치/물체버튼)가 통째로 멈춘다. 절대 되살리지 말 것.
        # 토크 ON/OFF: 초기화 완료 + 비활성(IDLE/DETECTING/ERROR) 상태에서만.
        #   모션 중(PICK/LIFT/MOVE)엔 command_enabled=False라 자동 차단 → 모션 중 토크 OFF 사고 방지.
        self.gripper_torque_on_button.setEnabled(gripper_cmd_ok and gripper_enable_svc)
        self.gripper_torque_off_button.setEnabled(gripper_cmd_ok and gripper_enable_svc)
        self._update_manual_button_texts()
        # 물체 선택 / 자동 선택: IDLE + 모든 서비스 준비 + 그리퍼 HW 완료 필요
        full_system_ready  = command_enabled and is_idle and pick_svc and gripper_hw_ready
        object_buttons_enabled = full_system_ready
        self.auto_button.setEnabled(full_system_ready)
        object_param_ready = (
            self.ros_node.cli_object_get_parameters.service_is_ready()
            and self.ros_node.cli_object_set_parameters.service_is_ready()
        )
        self.calib_load_button.setEnabled(self.ros_node.cli_object_get_parameters.service_is_ready())
        self.calib_apply_button.setEnabled(object_param_ready and is_idle)
        self.model_browse_button.setEnabled(True)
        self.model_apply_button.setEnabled(self.ros_node.cli_object_set_parameters.service_is_ready())

        # 검출/노출 조정 — confidence는 detector, 노출은 camera 서비스. 슬라이더는 항상 편집 가능.
        camera_param_svc = self.ros_node.cli_camera_set_parameters.service_is_ready()
        self.conf_apply_button.setEnabled(self.ros_node.cli_object_set_parameters.service_is_ready())
        self.auto_exposure_check.setEnabled(camera_param_svc)
        # 자동노출 ON이면 수동 슬라이더/적용 비활성 (효과 없으니 혼동 방지)
        manual_exp_enabled = camera_param_svc and not self.auto_exposure_check.isChecked()
        self.exposure_slider.setEnabled(manual_exp_enabled)
        self.exposure_spin.setEnabled(manual_exp_enabled)
        self.exposure_apply_button.setEnabled(manual_exp_enabled)

        # ── 그리퍼 정밀 제어 상태 및 활성화 제어 ─────────────────────────────
        # 실시간 상태 레이블 업데이트
        pres_curr = self.ros_node.gripper_present_current
        pres_pos = self.ros_node.gripper_present_position
        self.gripper_status_label.setText(f'실시간 - 전류: {pres_curr:.0f} mA | 위치: {pres_pos:.0f}')

        # 그리퍼 INIT/REINIT 라벨 — 진행 중이면 "INIT 5/15 | 47s | trying", 막힌 듯하면 stale 표시
        prog = self.ros_node.gripper_init_progress
        prog_t = self.ros_node.gripper_init_progress_t
        age = time.monotonic() - prog_t if prog_t > 0 else 999
        if gripper_hw_ready:
            self.gripper_init_label.setText('그리퍼: ✅ 준비')
            self.gripper_init_label.setStyleSheet(
                'color: #88ff88; font-weight: bold; padding: 2px 6px; border-radius: 4px;'
                'background-color: #003300;')
        elif prog and age < 30:
            # 최근 30초 안 메시지 → 진행 중일 가능성 (age가 카운트 되면 시각적 변화)
            indicator = '🔄' if age < 5 else '⏳'  # 5초 이내 갱신=빠른 진행, 그 이후=느림(stuck 의심)
            self.gripper_init_label.setText(f'그리퍼: {indicator} {prog} (수신 {age:.0f}s 전)')
            color = '#ffaa00' if age < 5 else '#ff5500'  # 늦으면 빨강 경향
            self.gripper_init_label.setStyleSheet(
                f'color: white; font-weight: bold; padding: 2px 6px; border-radius: 4px;'
                f'background-color: {color};')
        elif prog:
            # 메시지 30초 이상 stale → stuck/실패 가능성 큼
            self.gripper_init_label.setText(f'그리퍼: ⚠ STUCK? 마지막 메시지 {age:.0f}s 전: {prog}')
            self.gripper_init_label.setStyleSheet(
                'color: white; font-weight: bold; padding: 2px 6px; border-radius: 4px;'
                'background-color: #aa0000;')
        else:
            self.gripper_init_label.setText('그리퍼: 대기')
            self.gripper_init_label.setStyleSheet(
                'color: #aaa; font-weight: bold; padding: 2px 6px; border-radius: 4px;'
                'background-color: #2a2a2a;')

        # 실시간 그래프 데이터 추가
        self.realtime_graph.add_data(pres_curr)

        # 높은 전류 부하가 감지될 때 경고 표시 색상 부여
        if pres_curr >= 500.0:
            self.gripper_status_label.setStyleSheet(
                'color: #ff3333; font-weight: bold; background-color: #4a0000; padding: 4px; border-radius: 4px; font-family: monospace;'
            )
        elif pres_curr >= 300.0:
            self.gripper_status_label.setStyleSheet(
                'color: #ffff33; font-weight: bold; background-color: #4a4a00; padding: 4px; border-radius: 4px; font-family: monospace;'
            )
        else:
            self.gripper_status_label.setStyleSheet(
                'color: #33ff33; font-weight: bold; background-color: #003a00; padding: 4px; border-radius: 4px; font-family: monospace;'
            )
            
        # 그리퍼 파라미터 적용 버튼 및 컨트롤들 활성화 제어
        # 적용: 서비스 연결 + 그리퍼 HW 초기화 완료 + IDLE(설정 편집 게이트) + 이전 요청 완료
        gripper_apply_ok = (
            gripper_param_svc
            and gripper_hw_ready
            and config_edit_enabled
            and not getattr(self, '_gripper_apply_busy', False)
        )
        self.gripper_apply_button.setEnabled(gripper_apply_ok)

        # 슬라이더/스핀박스: 정밀 전류는 설정값이므로 IDLE 전용(동작 중 편집 불가).
        self.close_curr_slider.setEnabled(config_edit_enabled)
        self.close_curr_spin.setEnabled(config_edit_enabled)
        self.open_curr_slider.setEnabled(config_edit_enabled)
        self.open_curr_spin.setEnabled(config_edit_enabled)
        self.vel_slider.setEnabled(config_edit_enabled)
        self.vel_spin.setEnabled(config_edit_enabled)
        self.acc_slider.setEnabled(config_edit_enabled)
        self.acc_spin.setEnabled(config_edit_enabled)

        # ── 물체별 파지 강도 버튼 ─────────────────────────────────────
        # 적용: pick_place set_parameters 연결 + IDLE 전용. 저장: 파일 쓰기라 항상 가능.
        pickplace_param_svc = self.ros_node.cli_pickplace_set_parameters.service_is_ready()
        self.grip_strength_apply_button.setEnabled(config_edit_enabled and pickplace_param_svc)
        self.grip_strength_save_button.setEnabled(True)
        for _slider, _spin in self._grip_strength_rows.values():
            _slider.setEnabled(config_edit_enabled)
            _spin.setEnabled(config_edit_enabled)

        # ── TCP Z 안전 하한 ───────────────────────────────────────────
        # 적용: pick_place set_parameters 연결 + IDLE 전용. 저장: 파일 쓰기라 항상 가능.
        self.min_safe_z_apply_button.setEnabled(config_edit_enabled and pickplace_param_svc)
        self.min_safe_z_save_button.setEnabled(True)
        self.min_safe_z_spin.setEnabled(config_edit_enabled)

        # ── 그리퍼 브릿지 재시작 버튼 ─────────────────────────────────
        # 복구용이라 로봇 상태 무관하게 항상 활성. 재기동 진행 중(45s 창)엔만 비활성.
        gbr_busy = (self._gripper_bridge_restart_proc is not None
                    and time.monotonic() < self._gripper_bridge_restart_until)
        self.gripper_bridge_restart_button.setEnabled(not gbr_busy)
        if not gbr_busy and self._gripper_bridge_restart_proc is not None:
            if self.gripper_bridge_restart_label.text().startswith('⏳'):
                self.gripper_bridge_restart_label.setText('✅ 재기동 완료 — 그리퍼 ready 확인 후 사용')
            self._gripper_bridge_restart_proc = None

        # ── 안전 모드 버튼 ────────────────────────────────────────────
        # 속도 모드: E-STOP이 아닐 때 + 서비스 연결 필요
        self.speed_normal_button.setEnabled(not is_e_stopped and speed_normal_svc)
        self.speed_reduced_button.setEnabled(not is_e_stopped and speed_reduced_svc)
        # 서보 OFF: E-STOP 아닐 때 + 서비스 연결 / 서보 ON: SAFE_OFF(3,10) 또는 E-STOP + 서비스 연결
        is_safe_off = hw in (3, 10)   # STATE_SAFE_OFF, STATE_SAFE_OFF2
        self.servo_off_button.setEnabled(not is_e_stopped and servo_off_svc)
        self.servo_on_button.setEnabled((is_safe_off or is_e_stopped) and servo_on_svc)

        # ── HW 상태 레이블 ────────────────────────────────────────────
        hw_state_names = {
            0: 'INITIALIZING', 1: 'STANDBY', 2: 'MOVING',
            3: 'SAFE_OFF', 4: 'TEACHING', 5: 'SAFE_STOP',
            6: 'E-STOP', 7: 'HOMING', 8: 'RECOVERY',
            9: 'SAFE_STOP2', 10: 'SAFE_OFF2', 15: 'NOT_READY',
        }
        hw_name = hw_state_names.get(self.ros_node.hw_state, f'CODE={self.ros_node.hw_state}')
        hw_color = {
            1: '#1a6a1a',   # STANDBY   → 녹색
            2: '#1a4a8a',   # MOVING    → 파랑
            5: '#8a4a00',   # SAFE_STOP → 주황
            6: '#8a0000',   # E-STOP    → 빨강
            3: '#8a0000',   # SAFE_OFF  → 빨강
        }.get(self.ros_node.hw_state, '#444444')
        self.hw_state_label.setText(f'HW: {hw_name}')
        self.hw_state_label.setStyleSheet(
            f'font-weight: bold; padding: 4px 8px; border-radius: 4px;'
            f'background-color: {hw_color}; color: white;'
        )

        speed_name = '감속 모드' if self.ros_node.speed_mode == 1 else '정상 속도'
        speed_color = '#7a6000' if self.ros_node.speed_mode == 1 else '#1a5c1a'
        self.speed_mode_label.setText(f'속도: {speed_name}')
        self.speed_mode_label.setStyleSheet(
            f'font-weight: bold; padding: 4px 8px; border-radius: 4px;'
            f'background-color: {speed_color}; color: white;'
        )

        # ── Doosan 안전 모드 버튼 ────────────────────────────────────
        # 정상 운전: 서비스 연결 시 활성 (역구동 해제 수단이므로 비교적 관대)
        # 역구동: 이미 역구동 중이 아닐 때 + 서비스 연결
        is_backdrive = state == 'BACKDRIVE'
        self.safety_auto_button.setEnabled(safety_normal_svc)
        self.safety_backdrive_button.setEnabled(not is_backdrive and safety_bd_svc)

        # 역구동 중 라벨 업데이트
        if is_backdrive:
            self.safety_mode_label.setText('현재 안전 모드: 역구동 (중력보상 스트리밍 중)')
            self.safety_mode_label.setStyleSheet(
                'font-weight: bold; padding: 3px 6px; border-radius: 4px;'
                'background-color: #2a2a5a; color: #aaaaff;'
            )
        else:
            self.safety_mode_label.setText('현재 안전 모드: 정상 운전')
            self.safety_mode_label.setStyleSheet(
                'font-weight: bold; padding: 3px 6px; border-radius: 4px;'
                'background-color: #2a2a2a; color: white;'
            )

        # ── 배경색 경고 ───────────────────────────────────────────────
        if is_backdrive:
            self.setStyleSheet('QWidget { background-color: #0a0a2a; }')
        elif is_e_stopped:
            self.setStyleSheet('QWidget { background-color: #3a0000; }')
        else:
            self.setStyleSheet('')


        # 도달 가능한 물체만 통과 — 버튼·요약 모두 같은 필터 사용.
        reachable_snapshot = [
            item for item in detected_snapshot
            if self._is_object_reachable(item)
        ]

        # 같은 라벨의 물체가 여러 개 검출될 수 있으므로 버튼은 라벨 단위로만 만든다.
        labels = []
        for item in reachable_snapshot:
            label = item.get('label', 'unknown')
            if label not in labels:
                labels.append(label)

        self._refresh_buttons(self._stable_detection_labels(labels), object_buttons_enabled)
        self._refresh_summary(reachable_snapshot)

    def _update_manual_command_feedback(self, state: str):
        now = time.monotonic()
        if self._manual_command is not None:
            key = self._manual_command.get('key')
            wait_for_state = self._manual_command.get('wait_for_state', False)
            accepted = self._manual_command.get('accepted', False)
            progress_text = self._manual_command.get('progress_text', '')
            done_text = self._manual_command.get('done_text', '')
            min_busy_until = float(self._manual_command.get('min_busy_until', 0.0))

            if key == 'home' and state == 'HOME':
                self._manual_command_seen_active = True

            if state in ('ERROR', 'EMERGENCY_STOP'):
                self._finish_manual_command(f'{progress_text} 중단됨')
            elif wait_for_state and accepted and state == 'IDLE':
                self._finish_manual_command(done_text)
            elif not wait_for_state and accepted and now >= min_busy_until:
                self._finish_manual_command(done_text)
            elif now > self._manual_command_deadline:
                self._finish_manual_command(f'{progress_text} 확인 시간 초과')

        if self._manual_command is not None:
            self.command_status_label.setText(self._manual_command.get('progress_text', '명령 처리 중...'))
            return

        if self._manual_feedback and now <= self._manual_feedback_until:
            self.command_status_label.setText(self._manual_feedback)
        else:
            self._manual_feedback = ''
            self.command_status_label.setText('')

    def _update_manual_button_texts(self):
        texts = {
            'home': 'HOME 이동',
            'gripper_open': '그리퍼 OPEN',
            'gripper_close': '그리퍼 CLOSE',
            'recover_to_home': '에러 복구 & HOME 복귀',
        }
        if self._manual_command is not None:
            key = self._manual_command.get('key')
            texts[key] = self._manual_command.get('progress_text', texts.get(key, '처리 중...'))
        self.home_button.setText(texts['home'])
        self.gripper_open_button.setText(texts['gripper_open'])
        self.gripper_close_button.setText(texts['gripper_close'])
        self.recover_home_button.setText(texts['recover_to_home'])

    def _is_object_reachable(self, obj: dict) -> bool:
        """검출 물체가 도달 가능 영역 안에 있는지 — 박스 한계 + sqrt(x^2+y^2) 반경 둘 다 체크.
        한계 밖이면 버튼·요약 모두 가린다. pick_place_node의 _cb_pose 검사와 동일 룰 + 반경 추가."""
        pose = obj.get('pose') or {}
        try:
            x = float(pose.get('x', 0.0))
            y = float(pose.get('y', 0.0))
            z = float(pose.get('z', 0.0))
        except (TypeError, ValueError):
            return False
        n = self.ros_node
        if not (n.workspace_x_min <= x <= n.workspace_x_max):
            return False
        if not (n.workspace_y_min <= y <= n.workspace_y_max):
            return False
        if not (n.workspace_z_min <= z <= n.workspace_z_max):
            return False
        if (x * x + y * y) ** 0.5 > n.reach_radius_max:
            return False
        return True

    def _stable_detection_labels(self, labels: list):
        """짧은 검출 누락으로 물체 버튼이 깜빡이지 않도록 라벨 목록을 안정화한다."""
        if labels == self._candidate_labels:
            self._candidate_label_hits += 1
        else:
            self._candidate_labels = list(labels)
            self._candidate_label_hits = 1

        if self._candidate_label_hits >= self._label_stable_frames:
            self._stable_labels = list(self._candidate_labels)

        return self._stable_labels

    def _refresh_buttons(self, labels: list, enabled: bool):
        """검출된 라벨 목록에 맞게 버튼을 생성/표시/강조한다.

        버튼 관리 전략:
          - 버튼은 라벨 이름을 키로 dict(object_buttons)에 보관하고 처음 한 번만 생성한다.
          - 이후 호출에서는 visible 상태와 스타일만 갱신한다.
            (매 프레임 버튼을 삭제/재생성하면 레이아웃 깜빡임과 메모리 낭비 발생)
          - 이번 프레임에 없는 라벨의 버튼은 hide()하고, 다시 나타나면 show()한다.
          - 현재 selected_label과 일치하는 버튼은 파란색으로 강조한다.

        그리드 배치: 2열 그리드 (row = idx // 2, col = idx % 2)
        """
        active_labels = set(labels)
        for button in self.object_buttons.values():
            self.button_grid.removeWidget(button)

        for idx, label in enumerate(labels):
            button = self.object_buttons.get(label)
            if button is None:
                button = QPushButton(label)
                button.clicked.connect(lambda checked=False, text=label: self._select_label(text))
                self.object_buttons[label] = button
            button.setVisible(True)
            button.setEnabled(enabled)
            if label == self.ros_node.selected_label and self.ros_node.selected_label:
                button.setStyleSheet(
                    'background-color: #1f6feb; color: white; font-weight: bold;'
                )
            else:
                button.setStyleSheet('')
            self.button_grid.addWidget(button, idx // 2, idx % 2)

        for label, button in self.object_buttons.items():
            if label not in active_labels:
                button.setVisible(False)

    def _refresh_summary(self, detected_objects: list):
        # 우측 하단 요약은 "현재 검출된 물체 목록"을 사람이 빠르게 읽기 위한 영역이다.
        if not detected_objects:
            self.object_summary.setText('검출된 물체가 없습니다.')
            return

        lines = []
        for item in detected_objects:
            pose = item.get('pose', {})
            yaw = pose.get('yaw_deg', None)
            yaw_text = f'{yaw:+.1f}deg' if isinstance(yaw, (int, float)) else 'N/A'
            lines.append(
                f"[{item.get('label', 'unknown')}] conf={item.get('confidence', 0.0):.2f}\n"
                f"  XYZ=({pose.get('x', 0.0):+.3f}, {pose.get('y', 0.0):+.3f}, {pose.get('z', 0.0):+.3f}) m\n"
                f"  Yaw={yaw_text}"
            )
        self.object_summary.setText('\n\n'.join(lines))

    def _update_system_status_bar(self):
        colors = {
            'ok': '#1a7f37',
            'warn': '#9a6700',
            'bad': '#cf222e',
        }
        for key, state in self.ros_node.system_status_items:
            label = self.system_status_labels.get(key)
            if label is None:
                continue
            label.setStyleSheet(
                f'background-color: {colors.get(state, "#666")}; color: white;'
                'border-radius: 3px; font-size: 11px; font-weight: bold;'
            )

    def _draw_object_frames_on_pixmap(self, pixmap: QPixmap, detected_objects: list):
        """검출 물체의 픽셀 중심에 간단한 좌표계(X/Z) 오버레이를 그린다."""
        if pixmap.isNull():
            return
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)

            x_pen = QPen(QColor(255, 90, 90), 3)      # X축: 빨강
            z_pen = QPen(QColor(80, 220, 255), 3)     # Z축(테이블 법선): 하늘색
            center_pen = QPen(QColor(255, 255, 0), 3)
            text_pen = QPen(QColor(255, 255, 255), 1)
            axis_len = 42

            for item in detected_objects:
                u = int(item.get('pixel_u', -1))
                v = int(item.get('pixel_v', -1))
                if u < 0 or v < 0:
                    continue

                pose = item.get('pose', {})
                yaw_deg = pose.get('yaw_deg', None)

                painter.setPen(center_pen)
                painter.drawEllipse(u - 3, v - 3, 6, 6)

                if isinstance(yaw_deg, (int, float)):
                    yaw_rad = math.radians(float(yaw_deg))
                    dx = axis_len * math.cos(yaw_rad)
                    dy = -axis_len * math.sin(yaw_rad)
                    painter.setPen(x_pen)
                    painter.drawLine(u, v, int(round(u + dx)), int(round(v + dy)))
                    painter.drawText(int(round(u + dx + 6)), int(round(v + dy - 6)), 'LONG')

                painter.setPen(z_pen)
                painter.drawLine(u, v, u, v - axis_len)
                painter.drawText(u + 4, v - axis_len - 4, 'Z')

                label = item.get('label', 'obj')
                yaw_text = f'{float(yaw_deg):+.1f}deg' if isinstance(yaw_deg, (int, float)) else 'yaw=N/A'
                tag_text = f'{label} | {yaw_text}'
                tag_x = u + 10
                tag_y = v + 10
                tag_w = max(118, 8 * len(tag_text))
                tag_h = 24
                painter.fillRect(tag_x, tag_y, tag_w, tag_h, QColor(20, 20, 20, 180))
                painter.setPen(QPen(QColor(255, 190, 60), 1))
                painter.drawRect(tag_x, tag_y, tag_w, tag_h)
                painter.setPen(text_pen)
                painter.drawText(tag_x + 8, tag_y + 16, tag_text)
        finally:
            painter.end()

    def _build_selection_status(self):
        # 사용자가 아무 것도 고르지 않았으면 자동 선택 모드 상태를 명확히 보여 준다.
        if not self.ros_node.selected_label:
            return '선택 상태: 자동으로 가장 가까운 물체를 사용'

        labels = [item.get('label', '') for item in self.ros_node.detected_objects]
        if self.ros_node.selected_label in labels:
            return f'선택 상태: {self.ros_node.selected_label} 검출됨'
        return f'선택 상태: {self.ros_node.selected_label} 대기 중'



def main(args=None):
    """Qt 이벤트 루프와 ROS 2 spin을 단일 프로세스에서 통합 실행한다.

    통합 방식:
      - QApplication.exec_()이 Qt 이벤트 루프를 점유하므로
        rclpy.spin()을 별도 스레드에서 돌리는 대신
        QTimer로 10ms마다 spin_once()를 호출하는 방식을 사용한다.
      - 이렇게 하면 ROS 콜백과 Qt 이벤트가 모두 메인 스레드에서 처리되어
        스레드 안전성 문제 없이 공유 데이터(latest_qimage 등)에 접근할 수 있다.

    종료 흐름:
      사용자가 창을 닫으면 app.exec_() 반환 → destroy_node() → shutdown() 순서로 정리.
    """
    rclpy.init(args=args)
    node = PickPlaceGuiNode()

    app = QApplication(sys.argv)
    gui = PickPlaceGui(node)
    gui.show()
    gui.raise_()
    gui.activateWindow()

    # ROS 콜백 처리용 타이머: 10ms마다 spin_once() 호출
    # timeout_sec=0.0: 대기 없이 현재 큐에 있는 콜백만 즉시 처리
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0))
    timer.start(10)   # 10ms = 약 100Hz, 카메라 30fps에 비해 충분히 빠름

    exit_code = app.exec_()   # Qt 이벤트 루프 진입 (창 닫힐 때까지 블로킹)
    timer.stop()
    node.cleanup_hardware()
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
