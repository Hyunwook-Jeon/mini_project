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
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import QLibraryInfo
from rclpy.node import Node
from sensor_msgs.msg import Image

from rcl_interfaces.msg import Parameter as RclParameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from std_msgs.msg import Int32, String

from std_srvs.srv import Trigger

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = QLibraryInfo.location(QLibraryInfo.PluginsPath)


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

        # ROS 토픽으로 받은 영상/검출 결과를 Qt 위젯에서 바로 쓸 수 있게
        # 화면 표시용 상태를 멤버 변수로 유지한다.
        self.use_local_yolo = bool(self.get_parameter('use_local_yolo').value)
        self.bridge = CvBridge() if CvBridge is not None else None
        self.latest_qimage = None
        self.detected_objects = []
        self.selected_label = ''
        self.pick_place_state = 'IDLE'
        self._latest_raw_detections = []
        self.last_image_time = 0.0
        self.last_objects_time = 0.0
        self.last_state_time = 0.0
        self.last_hw_state_time = 0.0
        self.last_speed_mode_time = 0.0
        self.system_status_items = []
        self._last_system_status_check = 0.0

        # GUI는 직접 로봇을 움직이지 않고 "어떤 물체를 집을지"만 알린다.
        self.pub_selected = self.create_publisher(String, '/selected_object_label', 10)

        self.cli_run_once      = self.create_client(Trigger, '/pick_place/run_once')
        self.cli_go_home       = self.create_client(Trigger, '/pick_place/go_home')
        self.cli_gripper_open  = self.create_client(Trigger, '/gripper/open')
        self.cli_gripper_close = self.create_client(Trigger, '/gripper/close')
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

        # 로봇 하드웨어 상태 / 속도 모드 (pick_place_node 폴링 결과 수신)
        self.hw_state   = -1   # -1 = unknown
        self.speed_mode = 0    # 0 = NORMAL
        self.create_subscription(Int32, '/robot_hw_state',  self._cb_hw_state, 10)
        self.create_subscription(Int32, '/robot_speed_mode', self._cb_speed_mode, 10)

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
        self.detected_objects = payload.get('objects', [])
        self.selected_label = payload.get('selected_label', '')
        self.last_objects_time = time.monotonic()

    def _cb_state(self, msg: String):
        # 상태 문자열은 pick_place_node가 발행하는 값을 그대로 사용한다.
        self.pick_place_state = msg.data
        self.last_state_time = time.monotonic()

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

        self.system_status_items = [
            ('CAM', 'ok' if fresh(self.last_image_time) else 'bad'),
            ('DET', 'ok' if fresh(self.last_objects_time) else 'bad'),
            ('PICK', 'ok' if ready(self.cli_run_once) and fresh(self.last_state_time) else 'bad'),
            ('GRIP', 'ok' if ready(self.cli_gripper_open) and ready(self.cli_gripper_close) else 'bad'),
            ('HW', 'ok' if fresh(self.last_hw_state_time) else 'warn'),
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

        # 좌측은 카메라 영상, 우측은 상태/선택 패널로 나누어 배치한다.
        self.setWindowTitle('DSR RealSense Pick & Place GUI')
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.resize(1100, 720)
        self.move(40, 40)

        root = QHBoxLayout(self)

        left_box = QVBoxLayout()
        self.system_status_labels = {}
        self.system_status_bar = QWidget()
        self.system_status_bar.setFixedSize(276, 24)
        status_bar_layout = QHBoxLayout(self.system_status_bar)
        status_bar_layout.setContentsMargins(0, 0, 0, 4)
        status_bar_layout.setSpacing(4)
        for key in ('CAM', 'DET', 'PICK', 'GRIP', 'HW', 'SPD'):
            label = QLabel(key)
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(42, 20)
            label.setStyleSheet(
                'background-color: #666; color: white; border-radius: 3px;'
                'font-size: 11px; font-weight: bold;'
            )
            self.system_status_labels[key] = label
            status_bar_layout.addWidget(label)
        left_box.addWidget(self.system_status_bar, 0, Qt.AlignLeft)

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
        left_box.addWidget(compact_settings_group)

        self.image_label = QLabel('카메라 영상 대기 중...')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet(
            'background-color: #1e1e1e; color: white; border-radius: 10px;'
        )
        left_box.addWidget(self.image_label)

        right_panel = QVBoxLayout()

        # ── 긴급 제어 패널 (항상 최상단, 가장 눈에 띄게) ──────────────────
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

        emergency_layout.addWidget(self.e_stop_button)
        emergency_layout.addWidget(self.cancel_button)
        emergency_layout.addWidget(self.e_stop_reset_button)

        status_group = QGroupBox('상태')
        status_layout = QVBoxLayout(status_group)
        self.state_label = QLabel('Pick & Place 상태: IDLE')
        self.selection_label = QLabel('선택 물체: 자동 선택')
        self.selection_status_label = QLabel('선택 상태: 자동으로 가장 가까운 물체를 사용')
        self.command_status_label = QLabel('')
        self.command_status_label.setStyleSheet('color: #b0b0b0; font-weight: bold;')
        status_layout.addWidget(self.state_label)
        status_layout.addWidget(self.selection_label)
        status_layout.addWidget(self.selection_status_label)
        status_layout.addWidget(self.command_status_label)

        control_group = QGroupBox('수동 제어')
        control_layout = QVBoxLayout(control_group)
        self.home_button = QPushButton('HOME 이동')
        self.home_button.clicked.connect(self._go_home)
        self.gripper_open_button = QPushButton('그리퍼 OPEN')
        self.gripper_open_button.clicked.connect(self._gripper_open)
        self.gripper_close_button = QPushButton('그리퍼 CLOSE')
        self.gripper_close_button.clicked.connect(self._gripper_close)
        control_layout.addWidget(self.home_button)
        control_layout.addWidget(self.gripper_open_button)
        control_layout.addWidget(self.gripper_close_button)


        # ── 로봇 안전 모드 패널 ───────────────────────────────────────────
        safety_group = QGroupBox('로봇 안전 모드')
        safety_layout = QVBoxLayout(safety_group)

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
        self.speed_normal_button.setMinimumHeight(34)
        self.speed_normal_button.setStyleSheet(
            'QPushButton { background-color: #1a5c1a; color: white;'
            '  font-weight: bold; border-radius: 5px; }'
            'QPushButton:hover { background-color: #22881a; }'
            'QPushButton:disabled { background-color: #444; color: #888; }'
        )
        self.speed_normal_button.clicked.connect(self._speed_normal)

        self.speed_reduced_button = QPushButton('🟡 감속 모드')
        self.speed_reduced_button.setMinimumHeight(34)
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
        self.servo_off_button.setMinimumHeight(34)
        self.servo_off_button.setStyleSheet(
            'QPushButton { background-color: #5a0050; color: white;'
            '  font-weight: bold; border-radius: 5px; }'
            'QPushButton:hover { background-color: #880077; }'
            'QPushButton:disabled { background-color: #444; color: #888; }'
        )
        self.servo_off_button.clicked.connect(self._servo_off)

        self.servo_on_button = QPushButton('🟢 서보 ON')
        self.servo_on_button.setMinimumHeight(34)
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

        # ── Doosan 내장 안전 모드 패널 ──────────────────────────────────
        dsr_safety_group = QGroupBox('Doosan 안전 모드')
        dsr_safety_layout = QVBoxLayout(dsr_safety_group)

        self.safety_mode_label = QLabel('현재 안전 모드: 알 수 없음')
        self.safety_mode_label.setStyleSheet(
            'font-weight: bold; padding: 3px 6px; border-radius: 4px;'
            'background-color: #2a2a2a; color: white;'
        )
        dsr_safety_layout.addWidget(self.safety_mode_label)

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
        dsr_safety_layout.addLayout(safety_mode_row)

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

        right_panel.addWidget(emergency_group)
        right_panel.addWidget(status_group)
        right_panel.addWidget(control_group)
        right_panel.addWidget(safety_group)
        right_panel.addWidget(dsr_safety_group)

        right_panel.addWidget(object_group)
        right_panel.addStretch(1)

        root.addLayout(left_box, 2)
        root.addLayout(right_panel, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_ui)
        self.timer.start(100)

    def _select_label(self, label: str):
        self.ros_node.publish_selected_label(label)
        if label:
            self.ros_node.call_trigger_service(self.ros_node.cli_run_once, 'pick_place/run_once')

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

    def _update_ui(self):
        self.ros_node.refresh_system_status()
        self._maybe_load_object_settings()
        self._maybe_apply_saved_model_path()

        # 카메라 영상은 최신 프레임이 있을 때만 갱신한다.
        if self.ros_node.latest_qimage is not None:
            pixmap = QPixmap.fromImage(self.ros_node.latest_qimage)
            self._draw_object_frames_on_pixmap(pixmap)
            scaled = pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
        self._update_system_status_bar()

        # 선택 라벨이 비어 있으면 자동 선택 상태로 표현한다.
        selected_text = self.ros_node.selected_label or '자동 선택'
        self.state_label.setText(f'Pick & Place 상태: {self.ros_node.pick_place_state}')
        self.selection_label.setText(f'선택 물체: {selected_text}')
        self.selection_status_label.setText(self._build_selection_status())

        state        = self.ros_node.pick_place_state
        is_e_stopped = state == 'EMERGENCY_STOP'
        is_idle      = state == 'IDLE'
        is_active    = state not in ('IDLE', 'EMERGENCY_STOP')
        hw = self.ros_node.hw_state
        if self._reset_in_progress:
            if (not is_e_stopped and hw not in (6, 15)) or time.monotonic() > self._reset_deadline:
                self._reset_in_progress = False
        self._update_manual_command_feedback(state)

        # ── 긴급 제어 버튼 ────────────────────────────────────────────
        self.e_stop_button.setEnabled(not is_e_stopped)
        self.cancel_button.setEnabled(is_active)
        if self._reset_in_progress:
            self.e_stop_reset_button.setEnabled(False)
            self.e_stop_reset_button.setText('리셋 중...')
        else:
            self.e_stop_reset_button.setText('긴급정지 해제')
            self.e_stop_reset_button.setEnabled(is_e_stopped)

        # ── 수동 제어 버튼 ────────────────────────────────────────────
        manual_enabled = state in ('IDLE', 'DETECTING', 'ERROR') and not self._reset_in_progress and hw not in (6, 15)
        manual_busy = self._manual_command is not None
        command_enabled = manual_enabled and not manual_busy
        object_buttons_enabled = command_enabled and is_idle
        self.home_button.setEnabled(command_enabled)
        self.gripper_open_button.setEnabled(command_enabled)
        self.gripper_close_button.setEnabled(command_enabled)
        self._update_manual_button_texts()
        self.auto_button.setEnabled(command_enabled and is_idle)
        object_param_ready = (
            self.ros_node.cli_object_get_parameters.service_is_ready()
            and self.ros_node.cli_object_set_parameters.service_is_ready()
        )
        self.calib_load_button.setEnabled(self.ros_node.cli_object_get_parameters.service_is_ready())
        self.calib_apply_button.setEnabled(object_param_ready and is_idle)
        self.model_browse_button.setEnabled(True)
        self.model_apply_button.setEnabled(self.ros_node.cli_object_set_parameters.service_is_ready())

        # ── 안전 모드 버튼 ────────────────────────────────────────────
        # 속도 모드: EMERGENCY_STOP이 아닐 때 전환 가능
        self.speed_normal_button.setEnabled(not is_e_stopped)
        self.speed_reduced_button.setEnabled(not is_e_stopped)
        # 서보 OFF: EMERGENCY_STOP 아닐 때 / 서보 ON: HW 상태가 SAFE_OFF(3,10)일 때
        is_safe_off = hw in (3, 10)   # STATE_SAFE_OFF, STATE_SAFE_OFF2
        self.servo_off_button.setEnabled(not is_e_stopped)
        self.servo_on_button.setEnabled(is_safe_off or is_e_stopped)

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

        # ── Doosan 안전 모드 버튼 — 항상 활성 (역구동/비상정지 해제 수단이므로) ──
        is_backdrive = state == 'BACKDRIVE'
        self.safety_auto_button.setEnabled(True)
        self.safety_backdrive_button.setEnabled(not is_backdrive)

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


        # 같은 라벨의 물체가 여러 개 검출될 수 있으므로 버튼은 라벨 단위로만 만든다.
        labels = []
        for item in self.ros_node.detected_objects:
            label = item.get('label', 'unknown')
            if label not in labels:
                labels.append(label)

        self._refresh_buttons(self._stable_detection_labels(labels), object_buttons_enabled)
        self._refresh_summary()

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
        }
        if self._manual_command is not None:
            key = self._manual_command.get('key')
            texts[key] = self._manual_command.get('progress_text', texts.get(key, '처리 중...'))
        self.home_button.setText(texts['home'])
        self.gripper_open_button.setText(texts['gripper_open'])
        self.gripper_close_button.setText(texts['gripper_close'])

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

    def _refresh_summary(self):
        # 우측 하단 요약은 "현재 검출된 물체 목록"을 사람이 빠르게 읽기 위한 영역이다.
        if not self.ros_node.detected_objects:
            self.object_summary.setText('검출된 물체가 없습니다.')
            return

        lines = []
        for item in self.ros_node.detected_objects:
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

    def _draw_object_frames_on_pixmap(self, pixmap: QPixmap):
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

            for item in self.ros_node.detected_objects:
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
