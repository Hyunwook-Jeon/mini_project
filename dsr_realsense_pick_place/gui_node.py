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

import json
import os
import sys
import math
from pathlib import Path

import numpy as np
import rclpy

try:
    from cv_bridge import CvBridge
    _CV_BRIDGE_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - runtime env dependent
    CvBridge = None
    _CV_BRIDGE_IMPORT_ERROR = e

# Qt plugin 경로를 특정 시스템 경로로 강제하면
# PyQt5/Qt 런타임 버전이 다른 환경에서 xcb 로딩 충돌이 발생할 수 있다.
# 기본 탐색 경로(가상환경/사용자 site-packages 포함)를 그대로 사용한다.

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class PickPlaceGuiNode(Node):
    def __init__(self):
        super().__init__('pick_place_gui')

        # ROS 디버그 토픽 구독 모드 / 로컬 YOLO 모드 선택
        self.declare_parameter('use_local_yolo', True)
        self.declare_parameter('weights_path', '')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('fov_h_deg', 60.0)
        self.declare_parameter('default_object_height_m', 0.12)
        self.declare_parameter('origin_x', -0.80)
        self.declare_parameter('origin_y', 0.0)
        self.declare_parameter('origin_z', -0.96)

        # ROS 토픽으로 받은 영상/검출 결과를 Qt 위젯에서 바로 쓸 수 있게
        # 화면 표시용 상태를 멤버 변수로 유지한다.
        self.use_local_yolo = bool(self.get_parameter('use_local_yolo').value)
        self.bridge = CvBridge() if CvBridge is not None else None
        self.latest_qimage = None
        self.detected_objects = []
        self.selected_label = ''
        self.pick_place_state = 'IDLE'
        self._latest_raw_detections = []

        # GUI는 직접 로봇을 움직이지 않고 "어떤 물체를 집을지"만 알린다.
        self.pub_selected = self.create_publisher(String, '/selected_object_label', 10)
        if self.use_local_yolo:
            self._init_local_yolo()
        else:
            if self.bridge is None:
                raise RuntimeError(
                    f'cv_bridge import 실패: {_CV_BRIDGE_IMPORT_ERROR}. '
                    'use_local_yolo=true 로 실행하거나 ROS python 환경을 정리하세요.'
                )
            self.create_subscription(Image, '/detection_debug_image', self._cb_image, 10)
            self.create_subscription(String, '/detected_objects', self._cb_objects, 10)
        self.create_subscription(String, '/pick_place_state', self._cb_state, 10)

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def _find_default_weights(self) -> Path | None:
        root = self._repo_root() / 'runs'
        cands = list(root.rglob('best.pt'))
        if not cands:
            return None
        return max(cands, key=lambda p: p.stat().st_mtime)

    def _init_local_yolo(self):
        weights = str(self.get_parameter('weights_path').value).strip()
        if weights:
            self.weights_path = Path(weights).expanduser().resolve()
        else:
            found = self._find_default_weights()
            self.weights_path = found if found is not None else Path('yolov8n.pt')

        self.model = YOLO(str(self.weights_path))
        self.model_names = (
            self.model.names
            if isinstance(self.model.names, dict)
            else dict(enumerate(self.model.names))
        )
        self.cap = cv2.VideoCapture(int(self.get_parameter('camera_index').value))
        if not self.cap.isOpened():
            raise RuntimeError('카메라를 열 수 없습니다. camera_index 파라미터를 확인하세요.')

        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.fov_h_deg = float(self.get_parameter('fov_h_deg').value)
        self.default_object_height_m = float(self.get_parameter('default_object_height_m').value)
        self.origin_x = float(self.get_parameter('origin_x').value)
        self.origin_y = float(self.get_parameter('origin_y').value)
        self.origin_z = float(self.get_parameter('origin_z').value)

        self.local_timer = self.create_timer(0.033, self._tick_local_yolo)
        self.get_logger().info(f'로컬 YOLO 모드 시작: weights={self.weights_path}')

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

    def _tick_local_yolo(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        h, w = frame.shape[:2]
        fx, fy, cx, cy = self._intrinsics_from_fov(w, h, self.fov_h_deg)
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
                z_m = self._estimate_depth_m(bh, fy, self.default_object_height_m)
                if math.isnan(z_m):
                    continue

                x_opt = ((cx_box - cx) / fx) * z_m
                y_opt = ((cy_box - cy) / fy) * z_m
                x_cam, y_cam, z_cam = self._camera_to_project_camera_coords(x_opt, y_opt, z_m)
                x_abs, y_abs, z_abs = self._to_absolute_coords(x_cam, y_cam, z_cam)

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

        self._latest_raw_detections = raw_dets
        self.detected_objects = objects
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

    def _cb_state(self, msg: String):
        # 상태 문자열은 pick_place_node가 발행하는 값을 그대로 사용한다.
        self.pick_place_state = msg.data

    def publish_selected_label(self, label: str):
        # 빈 문자열은 "자동 선택" 모드로 해석된다.
        self.selected_label = label
        msg = String()
        msg.data = label
        self.pub_selected.publish(msg)


class PickPlaceGui(QWidget):
    def __init__(self, ros_node: PickPlaceGuiNode):
        super().__init__()
        self.ros_node = ros_node
        self.object_buttons = {}

        # 좌측은 카메라 영상, 우측은 상태/선택 패널로 나누어 배치한다.
        self.setWindowTitle('DSR RealSense Pick & Place GUI')
        self.resize(1100, 720)

        root = QHBoxLayout(self)

        left_box = QVBoxLayout()
        self.image_label = QLabel('카메라 영상 대기 중...')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet(
            'background-color: #1e1e1e; color: white; border-radius: 10px;'
        )
        left_box.addWidget(self.image_label)

        right_panel = QVBoxLayout()

        status_group = QGroupBox('상태')
        status_layout = QVBoxLayout(status_group)
        self.state_label = QLabel('Pick & Place 상태: IDLE')
        self.selection_label = QLabel('선택 물체: 자동 선택')
        self.selection_status_label = QLabel('선택 상태: 자동으로 가장 가까운 물체를 사용')
        status_layout.addWidget(self.state_label)
        status_layout.addWidget(self.selection_label)
        status_layout.addWidget(self.selection_status_label)

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

        right_panel.addWidget(status_group)
        right_panel.addWidget(object_group)
        right_panel.addStretch(1)

        root.addLayout(left_box, 2)
        root.addLayout(right_panel, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_ui)
        self.timer.start(100)

    def _select_label(self, label: str):
        # 버튼을 누르면 선택 라벨만 ROS 토픽으로 보낸다.
        # 실제 목표 선택은 object_detector 쪽에서 다시 수행한다.
        self.ros_node.publish_selected_label(label)

    def _update_ui(self):
        # 카메라 영상은 최신 프레임이 있을 때만 갱신한다.
        if self.ros_node.latest_qimage is not None:
            pixmap = QPixmap.fromImage(self.ros_node.latest_qimage)
            scaled = pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

        # 선택 라벨이 비어 있으면 자동 선택 상태로 표현한다.
        selected_text = self.ros_node.selected_label or '자동 선택'
        self.state_label.setText(f'Pick & Place 상태: {self.ros_node.pick_place_state}')
        self.selection_label.setText(f'선택 물체: {selected_text}')
        self.selection_status_label.setText(self._build_selection_status())

        # 같은 라벨의 물체가 여러 개 검출될 수 있으므로 버튼은 라벨 단위로만 만든다.
        labels = []
        for item in self.ros_node.detected_objects:
            label = item.get('label', 'unknown')
            if label not in labels:
                labels.append(label)

        self._refresh_buttons(labels)
        self._refresh_summary()

    def _refresh_buttons(self, labels: list):
        """검출된 라벨 목록에 맞게 버튼을 생성/표시/강조한다.

        버튼 관리 전략:
          - 버튼은 라벨 이름을 키로 dict(object_buttons)에 보관하고 처음 한 번만 생성한다.
          - 이후 호출에서는 visible 상태와 스타일만 갱신한다.
            (매 프레임 버튼을 삭제/재생성하면 레이아웃 깜빡임과 메모리 낭비 발생)
          - 이번 프레임에 없는 라벨의 버튼은 hide()하고, 다시 나타나면 show()한다.
          - 현재 selected_label과 일치하는 버튼은 파란색으로 강조한다.

        그리드 배치: 2열 그리드 (row = idx // 2, col = idx % 2)
        """
        active_labels = set(labels)   # 현재 프레임에서 검출된 라벨 집합
        for idx, label in enumerate(labels):
            button = self.object_buttons.get(label)
            if button is None:
                # 처음 등장한 라벨: 새 버튼 생성 후 dict와 그리드에 등록
                button = QPushButton(label)
                # lambda 캡처 주의: text=label로 현재 값을 고정해야
                # 루프 변수 label이 나중에 변해도 클릭 시 올바른 값이 전달된다
                button.clicked.connect(lambda checked=False, text=label: self._select_label(text))
                self.object_buttons[label] = button
                row = idx // 2   # 2열 그리드
                col = idx % 2
                self.button_grid.addWidget(button, row, col)
            button.setVisible(True)
            # 선택된 라벨 버튼: GitHub 파란색(#1f6feb)으로 강조
            if label == self.ros_node.selected_label and self.ros_node.selected_label:
                button.setStyleSheet(
                    'background-color: #1f6feb; color: white; font-weight: bold;'
                )
            else:
                button.setStyleSheet('')   # 기본 스타일 복원

        # 이번 프레임에 없는 라벨 버튼은 숨김 처리 (삭제하지 않고 재사용 대기)
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
            lines.append(
                f"- {item.get('label', 'unknown')} | "
                f"conf={item.get('confidence', 0.0):.2f} | "
                f"x={pose.get('x', 0.0):.3f}, y={pose.get('y', 0.0):.3f}, z={pose.get('z', 0.0):.3f}"
            )
        self.object_summary.setText('\n'.join(lines))

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

    # ROS 콜백 처리용 타이머: 10ms마다 spin_once() 호출
    # timeout_sec=0.0: 대기 없이 현재 큐에 있는 콜백만 즉시 처리
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0))
    timer.start(10)   # 10ms = 약 100Hz, 카메라 30fps에 비해 충분히 빠름

    exit_code = app.exec_()   # Qt 이벤트 루프 진입 (창 닫힐 때까지 블로킹)
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
