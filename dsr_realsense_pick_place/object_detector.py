"""
object_detector.py
------------------
RealSense RGB-D + YOLOv8 기반 객체 검출 노드.

동작 요약:
  1. 컬러 / 깊이 / 카메라 정보 토픽을 시간 동기화하여 수신
  2. YOLOv8 (또는 fallback 색상 검출)로 픽셀 좌표 및 클래스 추출
  3. 검출 bbox 중심 주변 depth 샘플을 MAD 필터링으로 정제하여 깊이(m) 산출
  4. RealSense SDK deproject 함수로 픽셀 → 카메라 3D 좌표 변환
  5. TF2를 이용해 카메라 좌표계 → 로봇 베이스 좌표계로 변환
  6. GUI 선택 라벨이 있으면 해당 물체만, 없으면 가장 가까운 물체 선택 후 발행

구독:
  /camera/color/image_raw                    (sensor_msgs/Image)      - RGB 컬러 이미지
  /camera/aligned_depth_to_color/image_raw   (sensor_msgs/Image)      - 컬러에 정렬된 깊이 이미지
  /camera/color/camera_info                  (sensor_msgs/CameraInfo) - 카메라 내부 파라미터
  /selected_object_label                     (std_msgs/String)        - GUI 선택 라벨

발행:
  /detected_object_pose    (geometry_msgs/PoseStamped) - 최종 선택된 물체의 베이스 좌표
  /selected_object_pose    (geometry_msgs/PoseStamped) - pick_place_node가 구독하는 타겟 좌표
  /detected_objects        (std_msgs/String)           - 검출 물체 전체 목록 (JSON 문자열)
  /detection_debug_image   (sensor_msgs/Image)         - bbox / 깊이 정보가 그려진 디버그 이미지
"""

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
import rclpy.duration
import numpy as np
import cv2
import pyrealsense2 as rs
import math

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (transform 메서드 등록용)
from tf_transformations import quaternion_from_matrix

from pca_mask_utils import (
        build_object_mask_for_pca,
        collect_points_cam_from_mask,
        object_frame_pca_short_x_long_y,
    )
from ultralytics import YOLO
try:
    from ultralytics import FastSAM
except ImportError:
    FastSAM = None

def _ray_unit_opencv(
    u: float, v: float, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0
    r = np.array([x, y, z], dtype=np.float64)
    return r / (np.linalg.norm(r) + 1e-12)

def _surface_normal_toward_camera(ray_to_point: np.ndarray) -> np.ndarray:
    return -ray_to_point

def _normal_from_depth_patch(
    depth_m: np.ndarray, u: float, v: float, fx: float, fy: float, cx: float, cy: float, patch: int = 9
) -> np.ndarray | None:
    h, w = depth_m.shape
    ui, vi = int(round(u)), int(round(v))
    r = patch // 2
    x1, y1 = max(0, ui - r), max(0, vi - r)
    x2, y2 = min(w - 1, ui + r), min(h - 1, vi + r)
    patch_d = depth_m[y1:y2+1, x1:x2+1].astype(np.float64)
    m = np.isfinite(patch_d) & (patch_d > 0.05) & (patch_d < 10.0)
    if np.count_nonzero(m) < patch * 2: return None
    ys, xs = np.indices(patch_d.shape)
    z = patch_d
    x3 = ((x1 + xs) - cx) / fx * z
    y3 = ((y1 + ys) - cy) / fy * z
    pts = np.stack([x3[m], y3[m], z[m]], axis=1)
    if pts.shape[0] < 8: return None
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    n = vh[-1, :]
    n /= (np.linalg.norm(n) + 1e-12)
    toward_cam = -centroid / (np.linalg.norm(centroid) + 1e-12)
    if np.dot(n, toward_cam) < 0: n = -n
    return n.astype(np.float64)


class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector')

        # ── 파라미터 선언 ────────────────────────────────────────────────
        # 토픽/프레임 이름을 파라미터로 빼 두면 launch 나 yaml 에서 쉽게 바꿀 수 있다.
        # 하드코딩을 피하고 config/pick_place_params.yaml 에서 중앙 관리한다.
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        # aligned_depth_to_color: 깊이 이미지를 컬러 프레임 해상도/시점에 정렬한 토픽.
        # 이 토픽을 쓰면 컬러 픽셀 좌표로 바로 깊이를 조회할 수 있어 별도 reprojection 불필요.
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        # robot_base_frame: TF tree에서 로봇 고정 기준 프레임 이름. Doosan은 'base_link'.
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('use_yolo', True)
        # yolo_model: 'yolov8n.pt' 처럼 모델 크기 코드만 써도 ultralytics가 자동으로 로컬 캐시
        # 또는 네트워크에서 다운로드한다. n(nano) < s < m < l < x 순서로 정확도/속도 트레이드오프.
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        # target_classes: 검출 대상 클래스 이름 목록 (COCO 기준). 빈 리스트이면 전체 클래스 허용.
        self.declare_parameter('target_classes', ['bottle', 'cup', 'bowl', 'sports ball', 'orange', 'apple'])
        # depth_scale: RealSense depth 이미지의 raw 값(uint16, mm 단위)을 m 단위로 바꾸는 계수.
        # D400 시리즈 기본값은 0.001 (1 raw = 1 mm).
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('min_depth_m', 0.15)   # 카메라 최소 유효 거리 (m)
        self.declare_parameter('max_depth_m', 1.5)    # 작업 공간 최대 깊이 (m)
        # depth_sample_radius: bbox 중심 주변에서 샘플링할 반경 (픽셀 단위).
        # 반경이 클수록 노이즈에 강하지만 엣지 근처에서 오차 증가.
        self.declare_parameter('depth_sample_radius', 5)
        # depth_center_ratio: 샘플링 원 안에서 실제로 사용할 비율 (0~1).
        # 1.0이면 반경 안 모든 픽셀, 0.6이면 중심 60% 영역만 사용.
        self.declare_parameter('depth_center_ratio', 0.6)
        # depth_outlier_mad_scale: MAD 기반 이상치 제거 스케일 팩터.
        # 값이 작을수록 이상치 기준이 엄격해져 더 많은 샘플이 제거된다.
        self.declare_parameter('depth_outlier_mad_scale', 2.5)
        self.declare_parameter('selected_object_topic', '/selected_object_label')
        # absolute_origin_in_camera_*:
        # 절대좌표계 원점이 카메라 좌표계에서 어디에 있는지(m) 지정.
        # 예) 원점이 카메라 기준 (-0.80, 0.0, -0.96)이면
        #     물체 절대좌표 = 물체카메라좌표 - 원점카메라좌표
        #                  = (x - (-0.80), y - 0.0, z - (-0.96))
        self.declare_parameter('absolute_origin_in_camera_x_m', -0.80)
        self.declare_parameter('absolute_origin_in_camera_y_m', 0.0)
        self.declare_parameter('absolute_origin_in_camera_z_m', -0.96) # 이 줄 뒤에 아래 내용 추가
        # absolute_calib_*_mm:
        # 절대좌표 계산 후 추가로 더하는 축별 보정값(mm).
        # 기본값: X -20mm, Y -20mm, Z +140mm
        self.declare_parameter('absolute_calib_x_mm', -20.0) # 이 줄 뒤에 아래 내용 추가
        self.declare_parameter('absolute_calib_y_mm', -20.0)
        self.declare_parameter('absolute_calib_z_mm', 140.0)
        self.declare_parameter('use_realsense', True) # RealSense 사용 여부
        # True면 위 원점 오프셋으로 절대좌표를 계산하고,
        # False면 기존처럼 TF 카메라→베이스 변환을 사용한다.
        self.declare_parameter('use_manual_absolute_origin', True)
        # ... (기존 파라미터 선언) ...
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('pca_z_band_m', 0.045)
        self.declare_parameter('pca_ellipse_scale', 0.38)
        self.declare_parameter('pca_step', 4)
        self.declare_parameter('use_fastsam_for_pca', False)
        self.declare_parameter('fastsam_weights', '')
        self.declare_parameter('use_seg_aux_for_pca', False)
        self.declare_parameter('seg_aux_weights', '')
        self.declare_parameter('seg_aux_imgsz', 640)
        self.declare_parameter('seg_crop_pad', 0.15)
        self.declare_parameter('no_depth_pca_fallback', False)
        p = self.get_parameter
        # 자주 쓰는 파라미터는 멤버 변수로 꺼내 두고 이후 계산에 재사용한다.
        self.camera_frame = p('camera_frame').value
        self.robot_base_frame = p('robot_base_frame').value
        self.use_yolo = p('use_yolo').value
        self.conf_thresh = p('confidence_threshold').value
        self.target_classes = p('target_classes').value
        self.depth_scale = p('depth_scale').value
        self.min_depth = p('min_depth_m').value
        self.max_depth = p('max_depth_m').value
        self.depth_r = p('depth_sample_radius').value
        self.depth_center_ratio = p('depth_center_ratio').value
        self.depth_outlier_mad_scale = p('depth_outlier_mad_scale').value
        self.abs_origin_cam_x = p('absolute_origin_in_camera_x_m').value
        self.abs_origin_cam_y = p('absolute_origin_in_camera_y_m').value
        self.abs_origin_cam_z = p('absolute_origin_in_camera_z_m').value
        self.abs_calib_x_m = float(p('absolute_calib_x_mm').value) / 1000.0
        self.abs_calib_y_m = float(p('absolute_calib_y_mm').value) / 1000.0 # 이 줄 뒤에 아래 내용 추가
        self.abs_calib_z_m = float(p('absolute_calib_z_mm').value) / 1000.0
        self.use_realsense = p('use_realsense').value
        self.get_logger().info(f"DEBUG: ObjectDetectorNode __init__ from {__file__} completed. self.use_realsense = {self.use_realsense}")
        # ... (기존 파라미터 할당) ...
        self.device = p('device').value
        self.pca_z_band_m = p('pca_z_band_m').value
        self.pca_ellipse_scale = p('pca_ellipse_scale').value
        self.pca_step = p('pca_step').value
        self.use_fastsam_for_pca = p('use_fastsam_for_pca').value
        self.fastsam_weights = p('fastsam_weights').value
        self.use_seg_aux_for_pca = p('use_seg_aux_for_pca').value
        self.seg_aux_weights = p('seg_aux_weights').value
        self.seg_aux_imgsz = p('seg_aux_imgsz').value
        self.seg_crop_pad = p('seg_crop_pad').value
        self.no_depth_pca_fallback = p('no_depth_pca_fallback').value
        self.use_manual_absolute_origin = p('use_manual_absolute_origin').value
        self.selected_object_label = ''
        self.last_logged_selected_label = None

        # RealSense SDK의 deproject 함수를 쓰기 위해 rs.intrinsics 객체를 저장한다.
        self.intrinsics = None
        self.fastsam_model = None
        self.seg_aux_model = None

        # ── YOLO 모델 로드 ───────────────────────────────────────────────
        self.model = None # 이 줄 뒤에 아래 내용 추가
        if self.use_yolo:
            self._load_yolo()

        # ── TF2 ─────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── CvBridge ────────────────────────────────────────────────────
        self.bridge = CvBridge()
        # ... (기존 코드) ...
        # 동기화된 최신 프레임을 저장해 두고, 검출은 같은 타이밍의 데이터로만 수행한다.
        self.latest_cv_color = None
        self.latest_cv_depth_mm = None

        # ── 구독 ────────────────────────────────────────────────────────
        # 컬러/깊이/카메라정보를 ApproximateTimeSynchronizer로 묶어서 수신한다.
        # 세 메시지의 타임스탬프가 slop(0.1초) 이내로 가까울 때만 콜백이 실행된다.
        # 이렇게 해야 서로 다른 시점의 프레임이 섞여 3D 좌표가 흔들리는 문제를 방지한다.
        # (예: t=0의 컬러에 t=0.2의 depth를 쓰면 움직이는 물체의 좌표가 틀릴 수 있음)
        self.color_sub = message_filters.Subscriber(
            self, Image, p('color_topic').value
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, p('depth_topic').value
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, p('camera_info_topic').value
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=10,   # 버퍼에 최대 10개 메시지를 보관하며 매칭 시도
            slop=0.1,        # 타임스탬프 허용 오차 (초). 카메라 fps가 30이면 0.033초 간격이므로 여유 있게 설정 # 이 줄 뒤에 아래 내용 추가
        )
        self.ts.registerCallback(self._cb_synced_camera)

        # --- 디버깅용 개별 토픽 수신 확인 ---
        # 아래 로그가 보이면 토픽 자체는 발행되나, 동기화 콜백이 실행 안되는 상황.
        self.create_subscription(Image, p('color_topic').value,
            lambda msg: self.get_logger().info('DEBUG: Color 토픽 수신 중...', throttle_duration_sec=5.0), 10)
        self.create_subscription(Image, p('depth_topic').value,
            lambda msg: self.get_logger().info('DEBUG: Depth 토픽 수신 중...', throttle_duration_sec=5.0), 10)
        self.create_subscription(CameraInfo, p('camera_info_topic').value,
            lambda msg: self.get_logger().info('DEBUG: CameraInfo 토픽 수신 중...', throttle_duration_sec=5.0), 10)

        self.create_subscription(String, p('selected_object_topic').value,
                                 self._cb_selected_object, 10)

        # ── 발행 ────────────────────────────────────────────────────────
        self.pub_pose = self.create_publisher(PoseStamped,
                                              '/detected_object_pose', 10)
        self.pub_selected_pose = self.create_publisher(PoseStamped,
                                                       '/selected_object_pose', 10)
        self.pub_objects = self.create_publisher(String, '/detected_objects', 10)
        self.pub_debug = self.create_publisher(Image,
                                               '/detection_debug_image', 10)

        self.get_logger().info('컬러/뎁스/카메라정보 토픽 동기화 대기 중...') # 이 줄 뒤에 아래 내용 추가
        self.get_logger().info('ObjectDetectorNode 시작')

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def _candidate_search_roots(self) -> list[Path]:
        roots: list[Path] = []
        seen: set[Path] = set()
        # ... (기존 코드) ...
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

        return roots # 이 줄 뒤에 아래 내용 추가

    def _resolve_model_name(self, model_name: str) -> str:
        model_path = Path(model_name).expanduser()
        if (
            not model_name or
            not model_path.suffix or
            model_path.is_absolute() or
            ('/' not in model_name and '\\' not in model_name)
        ):
            return model_name

        candidates = [root / model_path for root in self._candidate_search_roots()]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())

        matches: list[Path] = []
        for root in self._candidate_search_roots():
            matches.extend(p for p in root.rglob(model_path.name) if p.is_file())

        if matches:
            if len(matches) > 1:
                suffix = model_path.as_posix()
                for match in matches:
                    if match.as_posix().endswith(suffix):
                        return str(match.resolve())
            return str(matches[0].resolve())
        # ... (기존 코드) ...
        return str((self._repo_root() / model_path).resolve())

    # ────────────────────────────────────────────────────────────────────
    # YOLO 로드
    # ──────────────────────────────────────────────────────────────────── # 이 줄 뒤에 아래 내용 추가
    def _load_yolo(self):
        model_name = str(self.get_parameter('yolo_model').value).strip()
        model_name = self._resolve_model_name(model_name)
        try:
            # model_name 이 파일 경로면 로컬 파일을, 문자열이면 기본 weight 이름을 읽는다.
            self.model = YOLO(model_name)
            self.get_logger().info(f'YOLO 모델 로드 완료: {model_name}')
        except ImportError:
            self.get_logger().warn(
                'ultralytics 패키지 없음 → 색상 기반 검출로 전환. '
                '설치: pip install ultralytics'
            )
            self.use_yolo = False
        except Exception as e:
            self.get_logger().warn(
                f'YOLO 모델 로드 실패({model_name}): {e} '
                '→ 색상 기반 검출로 전환'
            )
            self.use_yolo = False

        if self.use_fastsam_for_pca and self.fastsam_weights and FastSAM:
            try:
                self.fastsam_model = FastSAM(self._resolve_model_name(self.fastsam_weights))
                self.get_logger().info(f'FastSAM 모델 로드 완료: {self.fastsam_weights}')
            except Exception as e:
                self.get_logger().warn(f'FastSAM 모델 로드 실패({self.fastsam_weights}): {e}')

        if self.use_seg_aux_for_pca and self.seg_aux_weights:
            try:
                self.seg_aux_model = YOLO(self._resolve_model_name(self.seg_aux_weights))
                self.get_logger().info(f'보조 세그 YOLO 모델 로드 완료: {self.seg_aux_weights}')
            except Exception as e:
                self.get_logger().warn(f'보조 세그 YOLO 모델 로드 실패({self.seg_aux_weights}): {e}')

    # ────────────────────────────────────────────────────────────────────
    # 콜백
    # ────────────────────────────────────────────────────────────────────
    def _cb_synced_camera(self, color_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        self.get_logger().info('DEBUG: 동기화된 카메라 프레임 수신! (Synced callback triggered)', throttle_duration_sec=5.0)
        # 세 토픽이 같은 시점 기준으로 묶여 들어왔을 때만 내부 버퍼를 갱신한다.
        try:
            self.latest_cv_color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            self.latest_cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge 변환 오류: {e}', throttle_duration_sec=3.0)
            return

        if self.intrinsics is None:
            # camera_info 메시지를 RealSense SDK의 rs.intrinsics 객체로 변환한다.
            # rs2_deproject_pixel_to_point() 함수에 이 객체가 필요하기 때문에 한 번만 변환한다.
            #
            # ROS CameraInfo.K 행렬 구조 (3x3 row-major):
            #   [fx  0  cx]       K[0]=fx, K[2]=cx (주점 x)
            #   [ 0 fy  cy]  →    K[4]=fy, K[5]=cy (주점 y)
            #   [ 0  0   1]
            intr = rs.intrinsics()
            intr.width = info_msg.width
            intr.height = info_msg.height
            intr.ppx = info_msg.k[2]   # 주점(principal point) x 좌표
            intr.ppy = info_msg.k[5]   # 주점 y 좌표
            intr.fx = info_msg.k[0]    # x축 초점 거리 (픽셀 단위)
            intr.fy = info_msg.k[4]    # y축 초점 거리 (픽셀 단위)
            # RealSense D400 시리즈는 plumb_bob(Brown-Conrady) 왜곡 모델을 사용한다.
            if info_msg.distortion_model in ('plumb_bob', 'rational_polynomial'):
                intr.model = rs.distortion.brown_conrady
            else:
                intr.model = rs.distortion.none
            intr.coeffs = list(info_msg.d)  # 왜곡 계수 [k1, k2, p1, p2, k3]
            self.intrinsics = intr
            self.get_logger().info('카메라 내장 파라미터(Intrinsics) 수신 완료.')

        # 동기화된 프레임이 들어올 때마다 바로 검출까지 이어서 수행한다.
        self._detect_and_publish() # 이 줄 뒤에 아래 내용 추가

    def _cb_selected_object(self, msg: String):
        # 빈 문자열이면 자동 선택 모드로 간주한다.
        self.selected_object_label = msg.data.strip()
        if self.selected_object_label != self.last_logged_selected_label:
            label_text = self.selected_object_label if self.selected_object_label else '자동 선택'
            self.get_logger().info(f'선택 물체 변경: {label_text}')
            self.last_logged_selected_label = self.selected_object_label

    # ────────────────────────────────────────────────────────────────────
    # 메인 검출 루프
    # ──────────────────────────────────────────────────────────────────── # 이 줄 뒤에 아래 내용 추가
    def _detect_and_publish(self):
        # 검출 전에 최소한 컬러/깊이/카메라 파라미터가 모두 준비돼 있어야 한다.
        if self.latest_cv_color is None or self.latest_cv_depth_mm is None:
            self.get_logger().warn('이미지 미수신 (color or depth None)', throttle_duration_sec=3.0)
            return
        if self.intrinsics is None:
            self.get_logger().warn('RealSense intrinsics 미수신', throttle_duration_sec=3.0)
            return

        color_img = self.latest_cv_color.copy() # BGR image
        depth_img_raw = self.latest_cv_depth_mm # uint16 mm
        depth_img_m = depth_img_raw.astype(np.float32) * self.depth_scale # float32 meters

        debug_img = color_img.copy() # For drawing debug info
        candidates = []

        # Get camera intrinsics (fx, fy, cx, cy) for normal calculation
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.ppx
        cy = self.intrinsics.ppy

        if self.use_yolo and self.model:
            yolo_results = self.model(color_img, conf=self.conf_thresh, verbose=False)
            self.get_logger().info(f"DEBUG: YOLO 추론 완료. {len(yolo_results[0].boxes) if yolo_results[0].boxes is not None else 0}개의 박스 검출.", throttle_duration_sec=5.0)
            r0 = yolo_results[0]
            if r0.boxes is None or len(r0.boxes) == 0:
                self.get_logger().warn('DEBUG: 검출된 객체가 없어 리턴합니다. (conf_thresh 확인 필요)', throttle_duration_sec=5.0)
                self._publish_detected_objects([])
                # 디버그 이미지를 발행해야 GUI에 영상이 표시됨
                self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug_img, 'bgr8'))
                return

            for i, box in enumerate(r0.boxes):
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                if self.target_classes and label not in self.target_classes:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                u = (x1 + x2) // 2
                v = (y1 + y2) // 2
                w_box = x2 - x1
                h_box = y2 - y1

                depth_m = self._estimate_depth_m(depth_img_raw, u, v)
                if depth_m is None:
                    self.get_logger().warn(f"DEBUG: [{label}] 객체의 깊이 추정 실패. 건너뜁니다.", throttle_duration_sec=2.0)
                    continue

                # Calculate surface normal for PCA
                normal_cam = None
                if self.use_realsense and depth_img_m is not None:
                    normal_cam = _normal_from_depth_patch(
                        depth_img_m, u, v, fx, fy, cx, cy, patch=9
                    )
                if normal_cam is None: # Fallback to ray-based normal if depth patch fails or not RealSense
                    ray = _ray_unit_opencv(u, v, fx, fy, cx, cy)
                    normal_cam = _surface_normal_toward_camera(ray)

                # PCA for object orientation and size
                pca_info = None
                if normal_cam is not None:
                    mask_obj, _ = build_object_mask_for_pca(
                        r0,
                        i, # Pass the index of the current box
                        depth_img_m, # depth in meters
                        x1, y1, x2, y2, # bbox coords
                        u, v, # bbox center
                        color_img.shape[1], color_img.shape[0], # image width, height
                        depth_m, # z_ref_m
                        self.pca_z_band_m,
                        self.pca_ellipse_scale,
                        use_depth_fallback=not self.no_depth_pca_fallback,
                        frame_bgr=color_img,
                        fastsam_model=self.fastsam_model,
                        seg_aux_model=self.seg_aux_model,
                        device=self.device,
                        seg_aux_imgsz=self.seg_aux_imgsz,
                        seg_crop_pad=self.seg_crop_pad,
                    )

                    if mask_obj is not None:
                        pts_cam = collect_points_cam_from_mask(
                            depth_img_m, # depth in meters
                            mask_obj,
                            fx, fy, cx, cy,
                            step=self.pca_step,
                        )
                        if pts_cam is not None:
                            pca_info = object_frame_pca_short_x_long_y(pts_cam, normal_cam)

                # Create PoseStamped
                pose_cam = self._pixel_to_camera_pose(u, v, depth_m)

                # If PCA info is available, update orientation
                yaw_deg = None
                if pca_info is not None:
                    # axis_x, axis_y, axis_z are in camera frame
                    # R_cam_obj: rotation matrix from object frame to camera frame
                    # Columns are object's X, Y, Z axes in camera frame
                    R_cam_obj = np.array([
                        pca_info['axis_x'],
                        pca_info['axis_y'],
                        pca_info['axis_z']
                    ]).T # Transpose to get columns as basis vectors

                    # tf_transformations.quaternion_from_matrix는 4x4 행렬을 예상합니다.
                    # 3x3 회전 행렬을 4x4 변환 행렬로 확장합니다.
                    T_cam_obj = np.identity(4)
                    T_cam_obj[:3, :3] = R_cam_obj
                    q_cam_obj = quaternion_from_matrix(T_cam_obj)

                    pose_cam.pose.orientation.x = q_cam_obj[0]
                    pose_cam.pose.orientation.y = q_cam_obj[1]
                    pose_cam.pose.orientation.z = q_cam_obj[2]
                    pose_cam.pose.orientation.w = q_cam_obj[3]

                    # Calculate yaw_deg for JSON/GUI display (angle of object's X-axis in camera's XY plane)
                    # Use object's Y-axis (long axis) for yaw_deg display
                    obj_y_proj_cam_xy = pca_info['axis_y'][:2]
                    yaw_rad = math.atan2(obj_y_proj_cam_xy[1], obj_y_proj_cam_xy[0])
                    yaw_deg = math.degrees(yaw_rad)

                # Transform to absolute pose (robot base frame)
                pose_abs = self._to_absolute_pose(pose_cam)
                if pose_abs is None:
                    continue

                pos = pose_abs.pose.position
                candidates.append({
                    'label': label,
                    'confidence': conf,
                    'depth_m': depth_m,
                    'pixel_u': u,
                    'pixel_v': v,
                    'pose': pose_abs, # This now contains orientation
                    'pose_dict': {
                        'x': pos.x,
                        'y': pos.y,
                        'z': pos.z,
                        'yaw_deg': yaw_deg, # Add yaw_deg for GUI/JSON
                        'std_minor_m': pca_info['std_minor_m'] if pca_info else None,
                        'std_major_m': pca_info['std_major_m'] if pca_info else None,
                    },
                })

                # Draw debug info on image
                cv2.rectangle(debug_img,
                              (x1, y1), # Use x1, y1, x2, y2 from box
                              (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(debug_img,
                            f'{label} {conf:.2f} | {depth_m:.3f}m',
                            (x1, y1 - 6), # Use x1, y1 for text position
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else: # Color-based detection fallback
            color_detections = self._detect_color(color_img)
            for u, v, w_box, h_box, label, conf in color_detections:
                depth_m = self._estimate_depth_m(depth_img_raw, u, v)
                if depth_m is None:
                    continue

                # Create PoseStamped (orientation remains w=1.0)
                pose_cam = self._pixel_to_camera_pose(u, v, depth_m)
                pose_abs = self._to_absolute_pose(pose_cam)
                if pose_abs is None:
                    continue

                pos = pose_abs.pose.position
                candidates.append({
                    'label': label,
                    'confidence': conf,
                    'depth_m': depth_m,
                    'pixel_u': u,
                    'pixel_v': v,
                    'pose': pose_abs,
                    'pose_dict': {
                        'x': pos.x,
                        'y': pos.y,
                        'z': pos.z,
                        'yaw_deg': None, # No PCA, no yaw
                        'std_minor_m': None,
                        'std_major_m': None,
                    },
                })

                # Draw debug info on image
                cv2.rectangle(debug_img,
                              (u - w_box // 2, v - h_box // 2),
                              (u + w_box // 2, v + h_box // 2),
                              (0, 255, 0), 2)
                cv2.putText(debug_img,
                            f'{label} {conf:.2f} | {depth_m:.3f}m',
                            (u - w_box // 2, v - h_box // 2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish detected objects (JSON)
        self._publish_detected_objects(candidates)

        # 디버그 영상은 GUI와 현장 확인용으로 별도 토픽에 내보낸다.
        self.pub_debug.publish(
            self.bridge.cv2_to_imgmsg(debug_img, 'bgr8'))

        selected = self._choose_target(candidates)
        if selected is None:
            self.get_logger().warn(f"DEBUG: 최종 Pick 대상을 선택하지 못했습니다. (후보: {len(candidates)}개)", throttle_duration_sec=2.0)
            return

        # 선택 결과는 "일반 검출 결과"와 "실제 pick 대상으로 쓸 결과"를 둘 다 발행한다.
        pose_abs = selected['pose']
        pos = pose_abs.pose.position
        quat = pose_abs.pose.orientation
        self.pub_pose.publish(pose_abs)
        self.pub_selected_pose.publish(pose_abs)
        self.get_logger().info(
            f'[{selected["label"]}] 절대좌표: '
            f'x={pos.x:.3f} y={pos.y:.3f} z={pos.z:.3f} m, '
            f'quat=[{quat.x:.3f},{quat.y:.3f},{quat.z:.3f},{quat.w:.3f}]'
        )

    def _estimate_depth_m(self, depth_img: np.ndarray, u: int, v: int):
        """bbox 중심 픽셀 (u, v) 주변에서 안정적인 깊이(m)를 추정한다.

        단계:
          1. 중심 주변 r픽셀 정사각형 ROI 추출
          2. ROI 안에서 원형 마스크로 중심 가까운 픽셀만 선택
          3. depth=0(측정 실패) 및 범위 밖 픽셀 제거
          4. MAD 기반 이상치 제거 후 중앙값 반환

        반환: 깊이(m), 유효한 샘플이 없으면 None
        """
        r = max(1, int(self.depth_r))
        h_img, w_img = depth_img.shape[:2]
        # 이미지 경계를 벗어나지 않도록 ROI 좌표를 클램핑한다.
        x0 = max(0, u - r)
        x1 = min(w_img, u + r + 1)
        y0 = max(0, v - r)
        y1 = min(h_img, v + r + 1)
        roi = depth_img[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        # ROI 내 각 픽셀의 중심까지 거리를 계산해 원형 마스크를 만든다.
        # depth_center_ratio로 원 반경을 조절하면 bbox 엣지 근처 노이즈를 줄일 수 있다.
        yy, xx = np.indices(roi.shape)
        center_y = v - y0
        center_x = u - x0
        dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        max_dist = max(1.0, float(r) * max(0.1, self.depth_center_ratio))

        # RealSense depth에서 0은 측정 실패(구멍, 반사 등) 픽셀이므로 반드시 제외한다.
        valid_mask = roi > 0
        valid_mask &= dist <= max_dist
        # depth_scale(0.001) 곱해 raw uint16(mm) → float32(m) 변환
        samples = roi[valid_mask].astype(np.float32) * self.depth_scale
        if samples.size == 0:
            return None

        # 카메라 최소 인식 거리(min_depth_m) 및 작업 공간 최대 거리(max_depth_m) 밖 샘플 제거
        samples = samples[(samples >= self.min_depth) & (samples <= self.max_depth)]
        if samples.size == 0:
            return None

        # ── MAD(Median Absolute Deviation) 이상치 제거 ────────────────
        # MAD는 중앙값 기준 절대 편차의 중앙값으로, 표준편차보다 이상치에 강건하다.
        # 알고리즘:
        #   median = 전체 샘플 중앙값
        #   MAD    = median(|x_i - median|)
        #   유효 범위: |x_i - median| ≤ depth_outlier_mad_scale × MAD
        # 유리면, 금속 반사, 배경이 부분적으로 bbox에 포함될 때 튀는 값을 제거한다.
        median = float(np.median(samples))
        abs_dev = np.abs(samples - median)
        mad = float(np.median(abs_dev))

        if mad > 0.0:
            filtered = samples[abs_dev <= self.depth_outlier_mad_scale * mad]
            if filtered.size > 0:
                samples = filtered

        # 평균 대신 중앙값을 사용해 남은 이상치의 영향도 최소화한다.
        return float(np.median(samples))

    # ──────────────────────────────────────────────────────────────────── # 이 줄 뒤에 아래 내용 추가
    # 픽셀 + depth → 카메라 프레임 PoseStamped
    # ────────────────────────────────────────────────────────────────────
    def _pixel_to_camera_pose(self, u: int, v: int, depth_m: float) -> PoseStamped:
        """픽셀 좌표 (u, v)와 깊이 depth_m(m)을 카메라 3D 좌표로 변환한다.

        RealSense SDK의 rs2_deproject_pixel_to_point()는 핀홀 모델 + 왜곡 보정을 적용해
        픽셀 좌표를 카메라 광학 좌표계(camera_color_optical_frame)의 3D 점으로 변환한다.

        이 프로젝트에서는 optical frame 축을 다음과 같이 재정의해 사용한다.
          X: 왼쪽 (+)   [기본 optical X(오른쪽 +)의 반대]
          Y: 아래쪽 (+) [기본 optical Y와 동일]
          Z: 카메라 뒤쪽 (+) [기본 optical Z(앞쪽 +)의 반대]
        """
        # rs2_deproject_pixel_to_point: (intrinsics, [u, v], depth_m) → [X, Y, Z]
        # 내부적으로 수식: X = (u - ppx) / fx * Z, Y = (v - ppy) / fy * Z
        X, Y, Z = rs.rs2_deproject_pixel_to_point(
            self.intrinsics,
            [float(u), float(v)],
            float(depth_m),
        )
        X = -X
        Z = -Z

        ps = PoseStamped()
        ps.header.frame_id = self.camera_frame   # 'camera_color_optical_frame'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = X
        ps.pose.position.y = Y
        ps.pose.position.z = Z
        ps.pose.orientation.w = 1.0   # 방향은 단위 quaternion (회전 없음)
        return ps
    # ... (기존 코드) ...
    # ──────────────────────────────────────────────────────────────────── # 이 줄 뒤에 아래 내용 추가
    # 카메라 프레임 → 로봇 베이스 프레임 변환
    # ────────────────────────────────────────────────────────────────────
    def _transform_to_base(self, pose_cam: PoseStamped):
        """카메라 광학 좌표계의 PoseStamped를 로봇 베이스 좌표계로 변환한다.

        TF tree 구성:
          base_link  ←[static TF, hand-eye 캘리브레이션 결과]← camera_color_optical_frame

        static TF는 launch 파일의 static_transform_publisher가 발행한다.
        TF 값이 실제 카메라 위치와 다르면 pick 좌표가 틀리므로 캘리브레이션이 중요하다.

        timeout=0.1초: TF 트리가 아직 준비되지 않았을 때 무한 대기를 방지한다.
        실패 시 None 반환 → 해당 검출 결과는 이번 프레임에서 제외된다.
        """
        try:
            # tf_buffer.transform()은 내부적으로 TF tree를 탐색해 변환 행렬을 찾고
            # pose_cam의 position/orientation에 해당 변환을 적용한다.
            pose_base = self.tf_buffer.transform(
                pose_cam,
                self.robot_base_frame,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            return pose_base
        except Exception as e:
            self.get_logger().warn(f'TF 변환 실패: {e}')
            return None
    # ... (기존 코드) ...
    def _to_absolute_pose(self, pose_cam: PoseStamped):
        """카메라 좌표를 절대좌표로 변환한다.

        use_manual_absolute_origin=True:
          절대원점의 카메라좌표(ox, oy, oz)를 사용해
          p_abs = p_cam - o_cam
        use_manual_absolute_origin=False:
          기존 TF(camera -> robot_base_frame) 변환 사용
        """
        if not self.use_manual_absolute_origin:
            return self._transform_to_base(pose_cam)

        pose_abs = PoseStamped()
        pose_abs.header = pose_cam.header
        pose_abs.header.frame_id = 'absolute_frame'
        pose_abs.pose.position.x = (
            pose_cam.pose.position.x - self.abs_origin_cam_x + self.abs_calib_x_m
        )
        pose_abs.pose.position.y = (
            pose_cam.pose.position.y - self.abs_origin_cam_y + self.abs_calib_y_m
        )
        pose_abs.pose.position.z = (
            pose_cam.pose.position.z - self.abs_origin_cam_z + self.abs_calib_z_m
        )
        pose_abs.pose.orientation = pose_cam.pose.orientation
        return pose_abs
    # ... (기존 코드) ...
    def _publish_detected_objects(self, candidates: list):
        # GUI가 별도 커스텀 메시지 없이 바로 읽을 수 있도록 JSON 문자열로 묶어 발행한다.
        msg = String()
        msg.data = json.dumps({
            'selected_label': self.selected_object_label,
            'objects': [
                {
                    'label': item['label'],
                    'confidence': item['confidence'],
                    'depth_m': item['depth_m'],
                    'pixel_u': item['pixel_u'],
                    'pixel_v': item['pixel_v'],
                    'pose': item['pose_dict'],
                }
                for item in candidates
            ],
        })
        self.pub_objects.publish(msg)
    # ... (기존 코드) ...
    def _choose_target(self, candidates: list):
        # 선택한 라벨이 있으면 그 라벨만 남기고,
        # 그렇지 않으면 전체 후보 중 가장 가까운 물체를 pick 대상으로 사용한다.
        filtered = candidates
        if self.selected_object_label:
            filtered = [
                item for item in candidates
                if item['label'] == self.selected_object_label
            ]
        if not filtered:
            if self.selected_object_label:
                self.get_logger().warn(
                    f'선택한 물체({self.selected_object_label})가 현재 화면에서 검출되지 않음',
                    throttle_duration_sec=2.0
                )
            return None
        return min(filtered, key=lambda item: item['depth_m'])
    # ... (기존 코드) ...
    # ──────────────────────────────────────────────────────────────────── # 이 줄 뒤에 아래 내용 추가
    # YOLO 검출
    # ────────────────────────────────────────────────────────────────────
    def _detect_yolo(self, img: np.ndarray) -> list:
        """YOLOv8로 이미지에서 물체를 검출하고 (u, v, w, h, label, conf) 리스트를 반환한다.

        반환 형식: [(중심 u, 중심 v, bbox 폭, bbox 높이, 클래스 이름, confidence), ...]
          - 이 형식으로 통일해야 이후 depth 추정(_estimate_depth_m)과 시각화 코드가
            YOLO/색상 검출 방식과 무관하게 동일하게 동작한다.
          - target_classes가 설정된 경우 해당 클래스가 아닌 결과는 필터링한다.
        """
        # verbose=False: 매 프레임마다 터미널에 검출 결과가 출력되지 않도록 설정
        results = self.model(img, conf=self.conf_thresh, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]   # COCO 클래스 이름 (예: 'bottle')
                # target_classes가 빈 리스트이면 모든 클래스 통과, 아니면 목록 내 클래스만 허용
                if self.target_classes and label not in self.target_classes:
                    continue
                conf = float(box.conf[0])
                # xyxy 형식: [x_min, y_min, x_max, y_max] (픽셀 좌표)
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                u = (x1 + x2) // 2   # bbox 중심 x (깊이 샘플링 기준점)
                v = (y1 + y2) // 2   # bbox 중심 y
                w = x2 - x1
                h = y2 - y1
                detections.append((u, v, w, h, label, conf))
        return detections
    # ... (기존 코드) ...
    # ──────────────────────────────────────────────────────────────────── # 이 줄 뒤에 아래 내용 추가
    # 색상 기반 검출 (YOLO 없을 때 fallback – 빨간 물체 검출)
    # ────────────────────────────────────────────────────────────────────
    def _detect_color(self, img: np.ndarray) -> list:
        """YOLO를 사용할 수 없을 때 단순 HSV 색상 기반으로 빨간 물체를 검출한다.

        ultralytics 미설치 또는 모델 로드 실패 시 자동으로 이 경로로 대체된다.
        데모/테스트 용도이므로 빨간색 물체만 찾는다.

        빨간색 HSV 범위:
          OpenCV HSV에서 빨간색은 색상(H) 0°와 360° 양쪽에 걸쳐 있어
          두 범위를 OR 합산해야 한다.
            - 범위1: H=[0~10]   (노란 계열 빨강)
            - 범위2: H=[160~180] (보라 계열 빨강)
          S(채도) ≥ 100: 흰색/회색 배경을 제외하기 위한 최솟값
          V(명도) ≥ 80: 어두운 그림자 영역 제외

        모폴로지 연산:
          CLOSE(7×7): 물체 내부 구멍을 채워 윤곽이 끊기지 않게 함
          OPEN(7×7):  작은 노이즈 블롭 제거

        면적 임계값 800px²: 멀리 있는 작은 반사광 등 제거
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 빨간색은 HSV 색상 원에서 0도와 360도(=180도) 근처 두 영역에 분포
        m1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 100, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 내부 홀 메우기
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 작은 노이즈 제거

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 800:   # 너무 작은 영역(노이즈) 제외
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            u = x + w // 2
            v = y + h // 2
            # confidence는 의미 없으므로 1.0으로 고정 (YOLO 형식과 통일)
            detections.append((u, v, w, h, 'red_object', 1.0))
        return detections
    # ... (기존 코드) ...

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
