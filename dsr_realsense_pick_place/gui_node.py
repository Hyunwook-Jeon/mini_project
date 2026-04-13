"""
gui_node.py
-----------
PyQt5 기반 Pick & Place GUI.

기능:
  - 카메라 디버그 영상 표시
  - 검출된 물체 버튼 표시
  - 원하는 물체 선택
  - 현재 상태 표시
"""

import json
import os
import sys

import numpy as np
import rclpy
from cv_bridge import CvBridge

# OpenCV 패키지의 Qt 플러그인과 충돌하지 않도록
# 시스템 PyQt5 플러그인 경로를 먼저 고정한다.
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'
os.environ['QT_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'

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


class PickPlaceGuiNode(Node):
    def __init__(self):
        super().__init__('pick_place_gui')

        self.bridge = CvBridge()
        self.latest_qimage = None
        self.detected_objects = []
        self.selected_label = ''
        self.pick_place_state = 'IDLE'

        self.pub_selected = self.create_publisher(String, '/selected_object_label', 10)
        self.create_subscription(Image, '/detection_debug_image', self._cb_image, 10)
        self.create_subscription(String, '/detected_objects', self._cb_objects, 10)
        self.create_subscription(String, '/pick_place_state', self._cb_state, 10)

    def _cb_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # OpenCV GUI 기능은 쓰지 않고, 배열 순서만 바꿔 Qt 이미지로 변환한다.
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        height, width, channel = rgb.shape
        bytes_per_line = channel * width
        self.latest_qimage = QImage(
            rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).copy()

    def _cb_objects(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('detected_objects JSON 파싱 실패')
            return
        self.detected_objects = payload.get('objects', [])
        self.selected_label = payload.get('selected_label', '')

    def _cb_state(self, msg: String):
        self.pick_place_state = msg.data

    def publish_selected_label(self, label: str):
        msg = String()
        msg.data = label
        self.pub_selected.publish(msg)


class PickPlaceGui(QWidget):
    def __init__(self, ros_node: PickPlaceGuiNode):
        super().__init__()
        self.ros_node = ros_node
        self.object_buttons = {}

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
        self.ros_node.publish_selected_label(label)

    def _update_ui(self):
        if self.ros_node.latest_qimage is not None:
            pixmap = QPixmap.fromImage(self.ros_node.latest_qimage)
            scaled = pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

        selected_text = self.ros_node.selected_label or '자동 선택'
        self.state_label.setText(f'Pick & Place 상태: {self.ros_node.pick_place_state}')
        self.selection_label.setText(f'선택 물체: {selected_text}')
        self.selection_status_label.setText(self._build_selection_status())

        labels = []
        for item in self.ros_node.detected_objects:
            label = item.get('label', 'unknown')
            if label not in labels:
                labels.append(label)

        self._refresh_buttons(labels)
        self._refresh_summary()

    def _refresh_buttons(self, labels: list):
        active_labels = set(labels)
        for idx, label in enumerate(labels):
            button = self.object_buttons.get(label)
            if button is None:
                button = QPushButton(label)
                button.clicked.connect(lambda checked=False, text=label: self._select_label(text))
                self.object_buttons[label] = button
                row = idx // 2
                col = idx % 2
                self.button_grid.addWidget(button, row, col)
            button.setVisible(True)
            if label == self.ros_node.selected_label and self.ros_node.selected_label:
                button.setStyleSheet(
                    'background-color: #1f6feb; color: white; font-weight: bold;'
                )
            else:
                button.setStyleSheet('')

        for label, button in self.object_buttons.items():
            if label not in active_labels:
                button.setVisible(False)

    def _refresh_summary(self):
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
        if not self.ros_node.selected_label:
            return '선택 상태: 자동으로 가장 가까운 물체를 사용'

        labels = [item.get('label', '') for item in self.ros_node.detected_objects]
        if self.ros_node.selected_label in labels:
            return f'선택 상태: {self.ros_node.selected_label} 검출됨'
        return f'선택 상태: {self.ros_node.selected_label} 대기 중'


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceGuiNode()

    app = QApplication(sys.argv)
    gui = PickPlaceGui(node)
    gui.show()

    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0))
    timer.start(10)

    exit_code = app.exec_()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
