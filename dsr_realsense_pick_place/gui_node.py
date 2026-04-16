"""
gui_node.py
-----------
PyQt5 기반 Pick & Place GUI 노드.

화면 구성:
  좌측: 카메라 디버그 영상 (640×480, yolo_live_cam_3d_metrics 와 동일:
        YOLO plot 에서 filled mask 끔(masks=False) + 윤곽/PCA 축 오버레이)
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
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
import torch

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
from geometry_msgs.msg import PoseStamped
from ultralytics import YOLO
from vision_display_utils import (
    camera_to_project_camera_coords,
    quadrant_obj_frame_from_normal,
)
from yolo_live_cam_3d_metrics import (
    Detection3DResult,
    draw_detection_overlay,
    draw_frame_hint_overlay,
    load_class_heights,
    process_detection,
)

try:
    from ultralytics import FastSAM
except ImportError:  # pragma: no cover - runtime env dependent
    FastSAM = None  # type: ignore[misc, assignment]


class PickPlaceGuiNode(Node):
    def __init__(self):
        super().__init__('pick_place_gui')

        # true: best.pt 로컬 YOLO + RealSense(pydeps) 또는 웹캠. false: /detection_debug_image ROS.
        # pick_place.launch.py 는 전체 구동 시 use_local_yolo=false 로 덮어써 카메라 이중 점유를 막음.
        self.declare_parameter('use_local_yolo', True)
        # 상대경로는 패키지 루트(저장소 루트) 기준 — train_yolo 산출물 기본 위치
        self.declare_parameter('weights_path', 'runs/weights/weights/best.pt')
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
        self.declare_parameter('rs_wait_ms', 15000)
        self.declare_parameter('origin_x', -0.80)
        self.declare_parameter('origin_y', 0.0)
        self.declare_parameter('origin_z', -0.96)
        self.declare_parameter('calib_dx_mm', -20.0)
        self.declare_parameter('calib_dy_mm', -20.0)
        self.declare_parameter('calib_dz_mm', 140.0)
        self.declare_parameter('pca_arrow_m', 0.08)
        self.declare_parameter('pca_z_band_m', 0.03)
        self.declare_parameter('pca_ellipse_scale', 0.85)
        self.declare_parameter('depth_pca_fallback', True)
        # False: 마스크 초록 윤곽(draw_mask_contour_style) 비표시
        self.declare_parameter('show_contour', False)
        # outline만: 마스크에 피팅 타원(cv2.ellipse)을 그리지 않음
        self.declare_parameter('contour_style', 'outline')
        self.declare_parameter('pca_step', 4)
        self.declare_parameter('fastsam_weights', '')
        self.declare_parameter('seg_weights', '')
        self.declare_parameter('seg_aux_imgsz', 640)
        self.declare_parameter('seg_crop_pad', 0.15)
        self.declare_parameter('yolo_device', '')
        self.declare_parameter('yolo_no_half', False)
        self.declare_parameter('show_yolo_bboxes', True)
        self.declare_parameter('class_heights_json', '')
        self.declare_parameter('data_yaml', '')
        self.declare_parameter('selected_object_info_topic', '/selected_object_info')
        # object_detector 가 발행하는 토픽 (런치에서 네임스페이스를 쓰면 여기서 remap)
        self.declare_parameter('debug_image_topic', '/detection_debug_image')
        self.declare_parameter('detected_objects_topic', '/detected_objects')

        # ROS 토픽으로 받은 영상/검출 결과를 Qt 위젯에서 바로 쓸 수 있게
        # 화면 표시용 상태를 멤버 변수로 유지한다.
        self.use_local_yolo = bool(self.get_parameter('use_local_yolo').value)
        self.bridge = CvBridge() if CvBridge is not None else None
        self.latest_qimage = None
        self.detected_objects = []
        self.selected_label = ''
        self.selected_object = None
        self.pick_place_state = 'IDLE'
        self._latest_raw_detections = []

        # GUI는 직접 로봇을 움직이지 않고 "어떤 물체를 집을지"만 알린다.
        self.pub_selected = self.create_publisher(String, '/selected_object_label', 10)
        self.pub_objects = self.create_publisher(String, '/detected_objects', 10)
        self.pub_pose = self.create_publisher(PoseStamped, '/detected_object_pose', 10)
        self.pub_selected_pose = self.create_publisher(PoseStamped, '/selected_object_pose', 10)
        selected_info_topic = str(self.get_parameter('selected_object_info_topic').value)
        self.pub_selected_info = self.create_publisher(String, selected_info_topic, 10)
        if self.use_local_yolo:
            self._init_local_yolo()
        else:
            if self.bridge is None:
                raise RuntimeError(
                    f'cv_bridge import 실패: {_CV_BRIDGE_IMPORT_ERROR}. '
                    'use_local_yolo=true 로 실행하거나 ROS python 환경을 정리하세요.'
                )
            dbg_topic = str(self.get_parameter('debug_image_topic').value)
            det_topic = str(self.get_parameter('detected_objects_topic').value)
            # object_detector 의 디버그 영상은 SensorData QoS 로 발행한다.
            self.create_subscription(
                Image,
                dbg_topic,
                self._cb_image,
                qos_profile_sensor_data,
            )
            self.create_subscription(String, det_topic, self._cb_objects, 10)
            self.get_logger().info(
                f'GUI ROS 영상 모드(use_local_yolo=false): {dbg_topic}, {det_topic} '
                '(QoS=sensor_data). 영상이 없으면 realsense2_camera·object_detector 가 '
                '떠 있는지, 카메라 토픽 이름이 object_detector 와 일치하는지 확인하세요.'
            )
        self.create_subscription(String, '/pick_place_state', self._cb_state, 10)

    def _mini_project_root(self) -> Path:
        """mini_project 저장소 루트 (``runs/.../best.pt`` 상대경로 기준).

        - 소스 / symlink-install: ``.../mini_project/dsr_realsense_pick_place/gui_node.py``
          → 한 단계 위가 아니라 **패키지 부모**가 저장소 루트.
        - 일반 colcon 설치: ``__file__`` 이 site-packages 아래면 ``parent.parent`` 는
          저장소가 아니다 → ``site-packages`` 부모는 버리고 cwd·상위·src 탐색.
        """
        env = os.environ.get('MINI_PROJECT_ROOT', '').strip()
        if env:
            p = Path(env).expanduser().resolve()
            if p.is_dir():
                return p

        default_best = ('runs', 'weights', 'weights', 'best.pt')

        def has_default_best(d: Path) -> bool:
            return (Path(d).joinpath(*default_best)).is_file()

        def is_repo_root(d: Path) -> bool:
            return (d / 'dsr_realsense_pick_place' / 'package.xml').is_file() and (
                (d / 'runs').is_dir() or (d / 'config').is_dir()
            )

        here = Path(__file__).resolve()
        pkg = here.parent

        # 표준 레이아웃: .../<repo>/dsr_realsense_pick_place/{gui_node.py, package.xml}
        # colcon build 는 .../build/dsr_realsense_pick_place/dsr_realsense_pick_place/gui_node.py
        # 처럼 부모가 ``runs/`` 없는 빌드 트리이므로, 부모에 저장소 흔적이 있을 때만 채택한다.
        if pkg.name == 'dsr_realsense_pick_place' and (pkg / 'package.xml').is_file():
            par = pkg.parent
            if par.name not in ('site-packages', 'dist-packages'):
                if (
                    (par / 'runs').is_dir()
                    or (par / 'config').is_dir()
                    or has_default_best(par)
                ):
                    return par

        for d in (here.parent, *here.parents):
            if is_repo_root(d):
                return d
            if has_default_best(d):
                return d

        cwd = Path.cwd().resolve()
        for d in (cwd, *cwd.parents):
            if is_repo_root(d):
                return d
            if has_default_best(d):
                return d
            src = d / 'src'
            if src.is_dir():
                try:
                    subs = [p for p in src.iterdir() if p.is_dir()]
                    # best.pt 가 있는 소스 트리를 우선 (워크스페이스에 패키지가 여럿일 때)
                    subs.sort(key=lambda s: (not has_default_best(s), str(s)))
                    for sub in subs:
                        if is_repo_root(sub) or has_default_best(sub):
                            return sub
                except OSError:
                    pass

        # build/install 경로에서 __file__ 조상만으로는 저장소를 못 찾는 경우가 많다.
        for d in (*cwd.parents, *here.parents):
            if has_default_best(d) or is_repo_root(d):
                return d

        return here.parent.parent

    def _find_best_pt(self, search_under: Path) -> Path | None:
        cands = list(search_under.rglob('best.pt'))
        if not cands:
            return None
        return max(cands, key=lambda p: p.stat().st_mtime)

    def _init_local_yolo(self):
        root = self._mini_project_root()
        weights = str(self.get_parameter('weights_path').value).strip()
        require_best_pt = bool(self.get_parameter('require_best_pt').value)
        if weights:
            configured = Path(weights).expanduser()
            if configured.is_absolute():
                self.weights_path = configured.resolve()
            else:
                # 상대경로는 저장소 루트 기준 (ros2 run 시 cwd와 무관)
                self.weights_path = (root / configured).resolve()
        else:
            # yolo_live_cam_3d_metrics.py 와 동일하게 runs 아래 최신 best.pt를 기본 사용
            found = self._find_best_pt(root / 'runs')
            if found is None and require_best_pt:
                raise RuntimeError(
                    'runs 아래에서 best.pt를 찾지 못했습니다. '
                    'weights_path 파라미터에 best.pt 경로를 지정하세요.'
                )
            self.weights_path = found if found is not None else Path('yolov8n.pt')

        if not self.weights_path.is_file():
            found = self._find_best_pt(root / 'runs')
            default_pt = root / 'runs' / 'weights' / 'weights' / 'best.pt'
            if found is not None:
                self.get_logger().warn(
                    f'weights_path 대상이 없어 runs 아래 최신 best.pt 사용: {found} '
                    f'(저장소 루트={root})'
                )
                self.weights_path = found
            elif default_pt.is_file():
                self.get_logger().warn(
                    f'weights_path 대상이 없어 기본 경로 사용: {default_pt} '
                    f'(저장소 루트={root})'
                )
                self.weights_path = default_pt.resolve()

        if require_best_pt and self.weights_path.name != 'best.pt':
            raise RuntimeError(
                f'require_best_pt=true 인데 모델이 best.pt가 아닙니다: {self.weights_path}'
            )
        if not self.weights_path.is_file() and self.weights_path.name == 'best.pt':
            raise RuntimeError(
                f'best.pt 파일이 없습니다: {self.weights_path}\n'
                f'저장소 루트로 인식한 경로: {root}\n'
                '다른 위치면 weights_path에 절대경로를 주거나 '
                '환경변수 MINI_PROJECT_ROOT=/path/to/mini_project 를 설정하세요.'
            )

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
        self.rs_wait_ms = int(self.get_parameter('rs_wait_ms').value)
        self.origin_x = float(self.get_parameter('origin_x').value)
        self.origin_y = float(self.get_parameter('origin_y').value)
        self.origin_z = float(self.get_parameter('origin_z').value)
        self.calib_dx_mm = float(self.get_parameter('calib_dx_mm').value)
        self.calib_dy_mm = float(self.get_parameter('calib_dy_mm').value)
        self.calib_dz_mm = float(self.get_parameter('calib_dz_mm').value)
        self.pca_arrow_m = float(self.get_parameter('pca_arrow_m').value)
        self.pca_z_band_m = float(self.get_parameter('pca_z_band_m').value)
        self.pca_ellipse_scale = float(self.get_parameter('pca_ellipse_scale').value)
        self.depth_pca_fallback = bool(self.get_parameter('depth_pca_fallback').value)
        self.show_contour = bool(self.get_parameter('show_contour').value)
        self.contour_style = str(self.get_parameter('contour_style').value)
        self.pca_step = int(self.get_parameter('pca_step').value)
        self.seg_aux_imgsz = int(self.get_parameter('seg_aux_imgsz').value)
        self.seg_crop_pad = float(self.get_parameter('seg_crop_pad').value)
        self.show_yolo_bboxes = bool(self.get_parameter('show_yolo_bboxes').value)
        yolo_dev = str(self.get_parameter('yolo_device').value).strip()
        self._yolo_dev = yolo_dev or ('0' if torch.cuda.is_available() else 'cpu')
        self._yolo_use_half = (
            (not bool(self.get_parameter('yolo_no_half').value))
            and self._yolo_dev != 'cpu'
            and torch.cuda.is_available()
        )

        ch_raw = str(self.get_parameter('class_heights_json').value).strip()
        dy_raw = str(self.get_parameter('data_yaml').value).strip()
        ch_p = Path(ch_raw).expanduser().resolve() if ch_raw else None
        if ch_p is not None and not ch_p.is_file():
            self.get_logger().warn(f'class_heights_json 없음, 무시: {ch_p}')
            ch_p = None
        if dy_raw:
            dy_p = Path(dy_raw).expanduser().resolve()
        else:
            dy_p = (root / 'datasets' / 'yolo_final' / 'data.yaml').resolve()
        if dy_p is not None and not dy_p.is_file():
            dy_p = None
        self.class_heights = load_class_heights(ch_p, dy_p)

        self.fastsam_model = None
        fw = str(self.get_parameter('fastsam_weights').value).strip()
        if fw:
            if FastSAM is None:
                self.get_logger().warn('FastSAM import 실패 — fastsam_weights 무시')
            else:
                try:
                    self.fastsam_model = FastSAM(fw)
                    self.get_logger().info(f'FastSAM 로드: {fw}')
                except Exception as e:
                    self.get_logger().warn(f'FastSAM 로드 실패: {e}')

        self.seg_aux_model = None
        sw = str(self.get_parameter('seg_weights').value).strip()
        if sw:
            sp = Path(sw).expanduser().resolve()
            if not sp.is_file():
                self.get_logger().warn(f'seg_weights 파일 없음: {sp}')
            else:
                try:
                    self.seg_aux_model = YOLO(str(sp))
                    self.get_logger().info(f'보조 세그 YOLO 로드: {sp}')
                except Exception as e:
                    self.get_logger().warn(f'보조 세그 YOLO 로드 실패: {e}')

        self._margs = SimpleNamespace(
            pca_z_band_m=self.pca_z_band_m,
            pca_ellipse_scale=self.pca_ellipse_scale,
            no_depth_pca_fallback=not self.depth_pca_fallback,
            pca_step=self.pca_step,
            seg_aux_imgsz=self.seg_aux_imgsz,
            seg_crop_pad=self.seg_crop_pad,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            origin_z=self.origin_z,
            calib_dx_mm=self.calib_dx_mm,
            calib_dy_mm=self.calib_dy_mm,
            calib_dz_mm=self.calib_dz_mm,
            default_object_height_m=self.default_object_height_m,
            show_contour=self.show_contour,
            contour_style=self.contour_style,
            pca_arrow_m=self.pca_arrow_m,
        )

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
        mode = 'RealSense depth' if self.use_realsense else 'OpenCV camera'
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

    def _obj_frame_for_publish(self, det: Detection3DResult, cx: float, cy: float) -> dict:
        """토픽 JSON용 obj_frame — 항상 4분면 법선 기반 축."""
        return quadrant_obj_frame_from_normal(
            det.normal, det.ux_show, det.uy_show, cx, cy
        )

    def _tick_local_yolo(self):
        depth_m = None
        if self.use_realsense:
            try:
                frames = self.pipeline.wait_for_frames(
                    timeout_ms=max(1000, int(self.rs_wait_ms))
                )
            except RuntimeError as e:
                self.get_logger().warn(
                    f'RealSense wait_for_frames 실패: {e}. '
                    'realsense2_camera 등이 같은 장치를 쓰면 끄거나 '
                    '`use_local_yolo:=false` 로 ROS /detection_debug_image 를 쓰세요.',
                    throttle_duration_sec=5.0,
                )
                return
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
        t0 = time.perf_counter()
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self._yolo_dev,
            half=self._yolo_use_half,
            verbose=False,
            retina_masks=True,
        )
        t1 = time.perf_counter()
        r0 = results[0]
        out = r0.plot(
            boxes=self.show_yolo_bboxes,
            labels=True,
            conf=True,
            line_width=2,
            masks=False,
        )
        fps = 1.0 / max(t1 - t0, 1e-6)
        draw_frame_hint_overlay(out, fps, self.use_realsense)

        raw_dets: list = []
        objects: list = []

        if r0.boxes is not None and len(r0.boxes) > 0:
            boxes = r0.boxes.xyxy.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)
            confs = r0.boxes.conf.cpu().numpy().astype(float)
            line_y = 72
            for i, b in enumerate(boxes):
                cid = int(clss[i]) if i < len(clss) else 0
                conf = float(confs[i]) if i < len(confs) else 0.0
                det = process_detection(
                    r0=r0,
                    det_index=i,
                    box=b,
                    cls_id=cid,
                    names=self.model_names,
                    frame=frame,
                    depth_m=depth_m,
                    use_rs=self.use_realsense,
                    w=w,
                    h=h,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    class_heights=self.class_heights,
                    default_object_height_m=self.default_object_height_m,
                    args=self._margs,
                    dev=self._yolo_dev,
                    fastsam_model=self.fastsam_model,
                    seg_aux_model=self.seg_aux_model,
                )
                if not math.isfinite(det.z_m) or math.isnan(det.z_m):
                    continue
                line_y = draw_detection_overlay(
                    out,
                    det,
                    line_y=line_y,
                    w=w,
                    h=h,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    args=self._margs,
                    use_rs=self.use_realsense,
                )
                x1, y1, x2, y2 = [float(b[j]) for j in range(4)]
                cx_box = det.cx_box
                cy_box = det.cy_box
                raw_dets.append(
                    (
                        int(round(cx_box)),
                        int(round(cy_box)),
                        int(max(x2 - x1, 1.0)),
                        int(max(y2 - y1, 1.0)),
                        det.label,
                        conf,
                    )
                )
                zm = float(det.z_m)
                x_opt = ((det.ux_show - cx) / fx) * zm
                y_opt = ((det.uy_show - cy) / fy) * zm
                x_cam, y_cam, z_cam = camera_to_project_camera_coords(x_opt, y_opt, zm)
                objects.append(
                    {
                        'label': det.label,
                        'confidence': conf,
                        'depth_m': zm,
                        'pixel_u': int(round(det.ux_show)),
                        'pixel_v': int(round(det.uy_show)),
                        'pose': {
                            'x': float(det.x_abs),
                            'y': float(det.y_abs),
                            'z': float(det.z_abs),
                            'frame_id': 'absolute_frame',
                        },
                        'pose_cam': {
                            'x': float(x_cam),
                            'y': float(y_cam),
                            'z': float(z_cam),
                            'frame_id': 'camera_color_optical_frame',
                        },
                        'normal_cam': [
                            float(det.normal[0]),
                            float(det.normal[1]),
                            float(det.normal[2]),
                        ],
                        'obj_frame': self._obj_frame_for_publish(det, cx, cy),
                    }
                )

        rgb = np.ascontiguousarray(out[:, :, ::-1])
        hh, ww, channel = rgb.shape
        bytes_per_line = channel * ww
        self.latest_qimage = QImage(
            rgb.data, ww, hh, bytes_per_line, QImage.Format_RGB888
        ).copy()

        self._latest_raw_detections = raw_dets
        self.detected_objects = objects
        self._update_selected_label_from_local_detections()
        self._publish_local_detection_topics()

    def _update_selected_label_from_local_detections(self):
        if not self._latest_raw_detections:
            return
        if not self.selected_label:
            # 자동 선택 모드에서는 selected_label을 비워 둔다.
            return
        labels = [obj.get('label', '') for obj in self.detected_objects]
        if self.selected_label not in labels:
            self.selected_label = ''

    def _publish_local_detection_topics(self):
        """로컬 YOLO 모드에서도 object_detector와 동일한 핵심 토픽을 발행한다."""
        selected = self._find_current_selected_object()
        payload = {
            'selected_label': self.selected_label,
            'selected_object': selected,
            'objects': self.detected_objects,
        }
        objects_msg = String()
        objects_msg.data = json.dumps(payload)
        self.pub_objects.publish(objects_msg)

        if selected is None:
            return

        pose = selected.get('pose', {})
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = str(pose.get('frame_id', 'absolute_frame'))
        ps.pose.position.x = float(pose.get('x', 0.0))
        ps.pose.position.y = float(pose.get('y', 0.0))
        ps.pose.position.z = float(pose.get('z', 0.0))
        ps.pose.orientation.w = 1.0
        self.pub_pose.publish(ps)
        self.pub_selected_pose.publish(ps)

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
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(
                f'detection_debug_image 변환 실패 (encoding={getattr(msg, "encoding", None)}): {e}',
                throttle_duration_sec=5.0,
            )
            return
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
        self.selected_object = payload.get('selected_object')
        self._publish_selected_object_info(reason='detected_objects_update')

    def _cb_state(self, msg: String):
        # 상태 문자열은 pick_place_node가 발행하는 값을 그대로 사용한다.
        self.pick_place_state = msg.data

    def publish_selected_label(self, label: str):
        # 빈 문자열은 "자동 선택" 모드로 해석된다.
        self.selected_label = label
        msg = String()
        msg.data = label
        self.pub_selected.publish(msg)
        if self.use_local_yolo:
            self._publish_local_detection_topics()
        self._publish_selected_object_info(reason='label_selected')

    def _find_current_selected_object(self):
        if self.selected_object:
            return self.selected_object
        if not self.detected_objects:
            return None
        if self.selected_label:
            same_label = [o for o in self.detected_objects if o.get('label') == self.selected_label]
            if not same_label:
                return None
            return min(same_label, key=lambda o: float(o.get('depth_m', float('inf'))))
        return min(self.detected_objects, key=lambda o: float(o.get('depth_m', float('inf'))))

    def _publish_selected_object_info(self, reason: str):
        """현재 선택 상태와 좌표 정보를 JSON으로 발행한다."""
        selected = self._find_current_selected_object()
        payload = {
            'reason': reason,
            'selected_label': self.selected_label,
            'selection_mode': 'manual' if self.selected_label else 'auto',
            'selected_object': selected,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.pub_selected_info.publish(msg)


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
        if ros_node.use_local_yolo:
            wait_msg = '카메라 영상 대기 중...'
        else:
            wait_msg = (
                'ROS 모드: /detection_debug_image 대기 중...\n'
                '(로컬 카메라만 쓰려면 ros2 run ... gui_node --ros-args -p use_local_yolo:=true)'
            )
        self.image_label = QLabel(wait_msg)
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
        self.selected_detail_label = QLabel(
            '선택 좌표 정보:\n- 절대좌표: N/A\n- 카메라좌표: N/A\n- 픽셀: N/A\n- depth: N/A'
        )
        self.selected_detail_label.setWordWrap(True)
        status_layout.addWidget(self.state_label)
        status_layout.addWidget(self.selection_label)
        status_layout.addWidget(self.selection_status_label)
        status_layout.addWidget(self.selected_detail_label)

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
        self.selected_detail_label.setText(self._build_selected_detail_text())

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
            normal = item.get('normal_cam', [0.0, 0.0, 1.0])
            nx = float(normal[0]) if len(normal) > 0 else 0.0
            ny = float(normal[1]) if len(normal) > 1 else 0.0
            nz = float(normal[2]) if len(normal) > 2 else 1.0
            lines.append(
                f"- {item.get('label', 'unknown')} | "
                f"centroid_uv=({item.get('pixel_u', -1)},{item.get('pixel_v', -1)}) | "
                f"dist_m={item.get('depth_m', 0.0):.3f} | "
                f"ABS_m=({pose.get('x', 0.0):+.3f},{pose.get('y', 0.0):+.3f},{pose.get('z', 0.0):+.3f}) | "
                f"normal_cam=({nx:+.3f},{ny:+.3f},{nz:+.3f})"
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

    def _build_selected_detail_text(self):
        selected = self.ros_node._find_current_selected_object()
        if not selected:
            return '선택 좌표 정보:\n- 절대좌표: N/A\n- 카메라좌표: N/A\n- 픽셀: N/A\n- depth: N/A'

        pose_abs = selected.get('pose', {})
        pose_cam = selected.get('pose_cam', {})
        normal = selected.get('normal_cam', [0.0, 0.0, 1.0])
        obj_frame = selected.get('obj_frame', {})
        abs_frame = pose_abs.get('frame_id', 'unknown')
        cam_frame = pose_cam.get('frame_id', 'unknown')
        nx = float(normal[0]) if len(normal) > 0 else 0.0
        ny = float(normal[1]) if len(normal) > 1 else 0.0
        nz = float(normal[2]) if len(normal) > 2 else 1.0

        return (
            '선택 좌표 정보:\n'
            f"- label: {selected.get('label', 'unknown')} (conf={selected.get('confidence', 0.0):.2f})\n"
            f"- centroid_uv=({selected.get('pixel_u', -1)},{selected.get('pixel_v', -1)}), dist_m={selected.get('depth_m', 0.0):.3f}\n"
            f"- 절대좌표[{abs_frame}]: x={pose_abs.get('x', 0.0):.3f}, "
            f"y={pose_abs.get('y', 0.0):.3f}, z={pose_abs.get('z', 0.0):.3f} m\n"
            f"- 카메라좌표[{cam_frame}]: x={pose_cam.get('x', 0.0):.3f}, "
            f"y={pose_cam.get('y', 0.0):.3f}, z={pose_cam.get('z', 0.0):.3f} m\n"
            f"- normal_cam=({nx:+.3f},{ny:+.3f},{nz:+.3f})\n"
            f"- obj_frame[{obj_frame.get('source', '?')}] mode={obj_frame.get('x_axis_mode', '?')} "
            f"X={obj_frame.get('axis_x', [1.0, 0.0, 0.0])} "
            f"Y={obj_frame.get('axis_y', [0.0, 1.0, 0.0])} "
            f"Z={obj_frame.get('axis_z', [0.0, 0.0, 1.0])}"
        )


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
    node.cleanup_hardware()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
