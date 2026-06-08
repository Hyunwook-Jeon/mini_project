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
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
import rclpy.duration
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import cv2
import pyrealsense2 as rs

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (transform 메서드 등록용)
from rcl_interfaces.msg import SetParametersResult


class TrackedDetectionManager:
    """ultralytics tracker ID를 사람이 읽기 쉬운 표시 번호로 매핑한다 (클래스별 격리).

    번호는 **클래스마다 따로** 1부터 시작 → `doll_1`과 `unknown_1`이 동시 공존.
    클래스별 state(tracks/reserved)를 분리해 두면 rebind도 자연스럽게 같은 클래스 안에서만 일어남.

    핵심 기능
      - 같은 (class, tid) → 같은 표시 번호 유지
      - 검출 잠깐 끊겨도 grace_sec 동안 GUI에 유지(깜빡임 방지)
      - tid 사라진 후 reserve_sec 동안 같은 클래스 + 같은 위치(임계 안)에 새 tid 오면 rebind
      - 다중 후보 시 가장 가까운 reserve 항목으로 매칭 (closest-match tie-break)
      - **클래스 변경 감지**: 같은 tid가 클래스 사이를 옮기면 옛 클래스에서 만료 처리하고
        새 클래스에 신규 등록 (라벨 튐을 최소화하기 위해 옛 display는 reserve로 이동)

    시간 단위는 모두 초(seconds, monotonic) — 가변 FPS에 안정적.
    """

    def __init__(
        self,
        appear_sec: float,
        grace_sec: float,
        reserve_sec: float,
        rebind_threshold_m: float,
    ):
        self.appear_sec = float(appear_sec)
        self.grace_sec = float(grace_sec)
        self.reserve_sec = float(reserve_sec)
        self.rebind_threshold_m = float(rebind_threshold_m)
        # class_name -> {"tracks": {tid: entry}, "reserved": {display: {free_at, last_pos}}}
        #   entry = {"display": int, "first_seen": t, "last_seen": t, "last_pos": (x,y,z),
        #            "shown": bool, "payload": Any | None}
        self._by_class: dict[str, dict] = {}
        # 빠른 역참조 — 동일 tid가 어떤 클래스에 속해 있는지 추적(클래스 변경 감지용).
        self._tid_to_class: dict[int, str] = {}

    def reset(self) -> None:
        self._by_class.clear()
        self._tid_to_class.clear()

    def get_display(self, tracker_id: int) -> int | None:
        """tid의 현재 display 번호를 조회. 클래스 격리 상태와 무관하게 동작."""
        cls = self._tid_to_class.get(tracker_id)
        if cls is None:
            return None
        state = self._by_class.get(cls)
        if state is None:
            return None
        e = state["tracks"].get(tracker_id)
        return e["display"] if e else None

    def set_payload(self, tracker_id: int, payload) -> None:
        """update() 후 candidate dict를 캐싱한다. grace 중 동일 payload를 재발행하기 위함."""
        cls = self._tid_to_class.get(tracker_id)
        if cls is None:
            return
        state = self._by_class.get(cls)
        if state is None:
            return
        e = state["tracks"].get(tracker_id)
        if e is not None:
            e["payload"] = payload

    def update(
        self,
        tracker_id: int,
        class_name: str,
        pos_xyz: tuple[float, float, float],
        now: float | None = None,
    ) -> tuple[int, bool]:
        """현재 검출을 manager에 알리고 (표시 번호, 노출 여부)를 받는다."""
        t = now if now is not None else time.monotonic()

        # 1) 클래스 변경 감지 — 같은 tid가 다른 클래스에서 옮겨오면 옛 entry 만료
        old_class = self._tid_to_class.get(tracker_id)
        if old_class is not None and old_class != class_name:
            self._expire_tid_from_class(tracker_id, old_class, t)

        # 2) 현 클래스 state 확보
        state = self._get_or_create_state(class_name)
        entry = state["tracks"].get(tracker_id)
        if entry is None:
            # 신규 — 같은 클래스 reserve에서 가까운 후보 있으면 rebind, 없으면 새 번호
            display = self._pick_rebind_or_new(state, pos_xyz)
            entry = {
                "display": display,
                "first_seen": t,
                "last_seen": t,
                "last_pos": pos_xyz,
                "shown": False,
                "payload": None,
            }
            state["tracks"][tracker_id] = entry
            self._tid_to_class[tracker_id] = class_name
        else:
            entry["last_seen"] = t
            entry["last_pos"] = pos_xyz

        # appear 임계: 처음 본 이후 appear_sec 지나야 노출
        if not entry["shown"] and (t - entry["first_seen"]) >= self.appear_sec:
            entry["shown"] = True
        return entry["display"], entry["shown"]

    def visible_lost_tracks(self, now: float | None = None) -> list[tuple[int, dict]]:
        """이번 프레임에 검출되지 않았지만 grace 안이라 아직 GUI에 보여야 하는 트랙들.
        반환: [(tid, entry)]."""
        t = now if now is not None else time.monotonic()
        out: list[tuple[int, dict]] = []
        for state in self._by_class.values():
            for tid, e in state["tracks"].items():
                if e["shown"] and 0.0 < (t - e["last_seen"]) <= self.grace_sec:
                    out.append((tid, e))
        return out

    def cleanup_expired(self, now: float | None = None) -> None:
        """grace 초과한 tid는 reserve로 이동, reserve 초과한 display는 free,
        빈 클래스 state는 메모리 절약차 제거."""
        t = now if now is not None else time.monotonic()
        empty_classes: list[str] = []
        for class_name, state in self._by_class.items():
            tracks = state["tracks"]
            reserved = state["reserved"]
            # grace 초과 tid 정리
            dead = [tid for tid, e in tracks.items() if (t - e["last_seen"]) > self.grace_sec]
            for tid in dead:
                e = tracks.pop(tid)
                self._tid_to_class.pop(tid, None)
                # shown까지 갔던 트랙만 reserve로 (false positive는 폐기)
                if e["shown"]:
                    reserved[e["display"]] = {
                        "free_at": t + self.reserve_sec,
                        "last_pos": e["last_pos"],
                    }
            # reserve 만료 정리
            expired = [d for d, r in reserved.items() if t >= r["free_at"]]
            for d in expired:
                reserved.pop(d, None)
            if not tracks and not reserved:
                empty_classes.append(class_name)
        for c in empty_classes:
            self._by_class.pop(c, None)

    # ── 내부 ─────────────────────────────────────────────────────
    def _get_or_create_state(self, class_name: str) -> dict:
        s = self._by_class.get(class_name)
        if s is None:
            s = {"tracks": {}, "reserved": {}}
            self._by_class[class_name] = s
        return s

    def _expire_tid_from_class(self, tracker_id: int, old_class: str, t: float) -> None:
        """tid가 다른 클래스로 옮겨갈 때 — 옛 클래스에서 entry 제거.
        shown 상태였다면 display는 reserve로 보내 grace+reserve 동안 회수 보호."""
        old_state = self._by_class.get(old_class)
        if old_state is None:
            return
        e = old_state["tracks"].pop(tracker_id, None)
        if e is not None and e["shown"]:
            old_state["reserved"][e["display"]] = {
                "free_at": t + self.reserve_sec,
                "last_pos": e["last_pos"],
            }
        self._tid_to_class.pop(tracker_id, None)

    def _pick_rebind_or_new(self, state: dict, pos_xyz: tuple[float, float, float]) -> int:
        """같은 클래스 state의 reserve 중 임계 안에서 가장 가까운 항목으로 rebind. 없으면 새 번호."""
        best_d = self.rebind_threshold_m
        best_disp = None
        for disp, r in state["reserved"].items():
            d = _dist3(r["last_pos"], pos_xyz)
            if d <= best_d:
                best_d = d
                best_disp = disp
        if best_disp is not None:
            state["reserved"].pop(best_disp, None)
            return best_disp
        return self._next_free_number(state)

    def _next_free_number(self, state: dict) -> int:
        used = {e["display"] for e in state["tracks"].values()} | set(state["reserved"].keys())
        n = 1
        while n in used:
            n += 1
        return n


def _dist3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


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
        # target_classes: 검출 대상 클래스 이름 목록.
        # ⚠️ ROS2 humble은 yaml의 빈 배열 []에서 string array 타입을 못 추론해 노드가 죽는다.
        # 빈 string [""]를 default로 쓰면 yaml 미지정 시에도 type=STRING_ARRAY가 보장되고,
        # 코드에서는 빈 문자열을 filter out해 결과적으로 "전체 통과"가 된다.
        self.declare_parameter('target_classes', [''])
        # ── 추적/디바운스 (검출 안정화: tracker ID → 클래스별 표시 번호) ──
        # 통합 학습 모델 가정: 우리가 학습시킨 클래스(known_classes)는 클래스명 그대로 ("doll_1"),
        # 모델이 'object'로 출력한(또는 known set 밖의) 검출은 'unknown_N'으로 표시.
        # 짧은 occlusion·검출 jitter는 manager가 흡수해 깜빡임 최소화.
        self.declare_parameter('tracker_type', 'botsort_custom.yaml')
        self.declare_parameter('debounce_appear_sec', 0.0)
        self.declare_parameter('debounce_grace_sec', 1.0)
        self.declare_parameter('display_number_reserve_sec', 2.0)
        self.declare_parameter('rebind_position_threshold_m', 0.03)
        # 통계 로그 주기 (매 N 프레임마다 raw/tracker_passed/shown 출력. 0=꺼짐).
        # 검출률 측정용 — appear/tracker 필터로 얼마나 떨어지는지 시각화.
        self.declare_parameter('detection_stats_log_period', 30)
        # known_classes — 정답(학습 인지) 클래스 목록. 이 외 검출은 unknown_N으로 표시.
        # 통상 pick_place_node의 grip_class_names와 동일 값으로 유지하는 것이 권장 (라벨↔강도 일치).
        # 빈 문자열만 있으면 → 모든 클래스를 unknown 처리(권장 X, 통합 모델 전 임시).
        self.declare_parameter('known_classes', [''])
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

        self.declare_parameter('use_object_yaw_for_grasp', True)
        self.declare_parameter('yaw_axis_reference', 'long')
        self.declare_parameter('yaw_depth_band_m', 0.03)
        self.declare_parameter('yaw_min_mask_pixels', 40)
        # absolute_origin_in_camera_*:
        # 절대좌표계 원점이 카메라 좌표계에서 어디에 있는지(m) 지정.
        # 예) 원점이 카메라 기준 (-0.80, 0.0, -0.96)이면
        #     물체 절대좌표 = 물체카메라좌표 - 원점카메라좌표
        #                  = (x - (-0.80), y - 0.0, z - (-0.96))
        self.declare_parameter('absolute_origin_in_camera_x_m', -0.80)
        self.declare_parameter('absolute_origin_in_camera_y_m', 0.0)
        self.declare_parameter('absolute_origin_in_camera_z_m', -0.96)
        # absolute_calib_*_mm:
        # 절대좌표 계산 후 추가로 더하는 축별 보정값(mm).
        # 기본값: X -20mm, Y -20mm, Z +140mm
        self.declare_parameter('absolute_calib_x_mm', -20.0)
        self.declare_parameter('absolute_calib_y_mm', -20.0)
        self.declare_parameter('absolute_calib_z_mm', 140.0)
        # True면 위 원점 오프셋으로 절대좌표를 계산하고,
        # False면 기존처럼 TF 카메라→베이스 변환을 사용한다.
        self.declare_parameter('use_manual_absolute_origin', True)

        p = self.get_parameter
        # 자주 쓰는 파라미터는 멤버 변수로 꺼내 두고 이후 계산에 재사용한다.
        self.camera_frame = p('camera_frame').value
        self.robot_base_frame = p('robot_base_frame').value
        self.use_yolo = p('use_yolo').value
        self.conf_thresh = p('confidence_threshold').value
        # 빈 문자열은 무시 — yaml의 [""] / 의도된 "전체 통과" 케이스를 깔끔히 처리
        _raw_classes = list(p('target_classes').value)
        self.target_classes = [c.strip() for c in _raw_classes if c and c.strip()]
        # known_classes: 정답(학습) 클래스 set — 빈 문자열 무시. 'object'는 자동으로 unknown 처리.
        _raw_known = list(p('known_classes').value)
        self._known_classes = {c.strip() for c in _raw_known if c and c.strip()}
        # 'object'를 known_classes에 넣는 건 의미 없음(이미 unknown 마커). 자동 제외.
        self._known_classes.discard('object')
        # 추적/디바운스
        self.tracker_type = p('tracker_type').value
        self._track_manager = TrackedDetectionManager(
            appear_sec=float(p('debounce_appear_sec').value),
            grace_sec=float(p('debounce_grace_sec').value),
            reserve_sec=float(p('display_number_reserve_sec').value),
            rebind_threshold_m=float(p('rebind_position_threshold_m').value),
        )
        self._tracker_yaml_path = self._resolve_tracker_yaml(self.tracker_type)
        self.get_logger().info(
            f'추적 설정: tracker={self._tracker_yaml_path or "(ultralytics 기본)"}, '
            f'appear={p("debounce_appear_sec").value}s, grace={p("debounce_grace_sec").value}s, '
            f'reserve={p("display_number_reserve_sec").value}s, '
            f'rebind={p("rebind_position_threshold_m").value}m')
        # 통계 카운터 — 매 N 프레임마다 출력
        self._stats_period = int(p('detection_stats_log_period').value)
        self._stats_frame = 0
        self._stats_raw = 0          # detector.track() 통과한 box(.id 부여된) 개수
        self._stats_no_id = 0        # box.id is None으로 skip된 개수
        self._stats_shown = 0        # appear 임계 통과해 GUI에 노출된 개수
        self._stats_lost_grace = 0   # 검출은 안 됐지만 grace 안이라 노출된 개수
        self.get_logger().info(
            f'known_classes(정답 클래스) = {sorted(self._known_classes) if self._known_classes else "(없음 — 모두 unknown_N으로 표시)"}'
        )
        self.get_logger().info(
            '  ↑ pick_place의 grip_class_names와 동일하게 유지해야 라벨↔강도가 일관됨'
        )
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
        self.abs_calib_y_m = float(p('absolute_calib_y_mm').value) / 1000.0
        self.abs_calib_z_m = float(p('absolute_calib_z_mm').value) / 1000.0
        self.use_manual_absolute_origin = p('use_manual_absolute_origin').value
        self.use_object_yaw_for_grasp = p('use_object_yaw_for_grasp').value
        self.yaw_axis_reference = str(p('yaw_axis_reference').value).strip().lower()
        self.yaw_depth_band_m = float(p('yaw_depth_band_m').value)
        self.yaw_min_mask_pixels = int(p('yaw_min_mask_pixels').value)

        self.selected_object_label = ''
        self.last_logged_selected_label = None

        # pick_place 상태 구독 — LIFT/MOVE_TO_PLACE 중에는 "검출되지 않음" WARN을 억제한다.
        self._pick_place_state = ''

        # RealSense SDK의 deproject 함수를 쓰기 위해 rs.intrinsics 객체를 저장한다.
        self.intrinsics = None

        # ── YOLO 모델 로드 ───────────────────────────────────────────────
        self.model = None
        if self.use_yolo:
            self._load_yolo()
        # 모델 로드 후, known_classes와 model.names가 어긋났는지 안내 (안전망)
        self._warn_class_alignment()
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        # ── TF2 ─────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── CvBridge ────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # 동기화된 최신 프레임을 저장해 두고, 검출은 같은 타이밍의 데이터로만 수행한다.
        self.latest_cv_color = None
        self.latest_cv_depth_mm = None

        # ── 구독 ────────────────────────────────────────────────────────
        # 컬러/깊이/카메라정보를 ApproximateTimeSynchronizer로 묶어서 수신한다.
        # 세 메시지의 타임스탬프가 slop(0.1초) 이내로 가까울 때만 콜백이 실행된다.
        # 이렇게 해야 서로 다른 시점의 프레임이 섞여 3D 좌표가 흔들리는 문제를 방지한다.
        # (예: t=0의 컬러에 t=0.2의 depth를 쓰면 움직이는 물체의 좌표가 틀릴 수 있음)
        self.color_sub = message_filters.Subscriber(
            self,
            Image,
            p('color_topic').value,
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            p('depth_topic').value,
            qos_profile=qos_profile_sensor_data,
        )
        self.info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            p('camera_info_topic').value,
            qos_profile=qos_profile_sensor_data,
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=10,   # 버퍼에 최대 10개 메시지를 보관하며 매칭 시도
            slop=0.1,        # 타임스탬프 허용 오차 (초). 카메라 fps가 30이면 0.033초 간격이므로 여유 있게 설정
        )
        self.ts.registerCallback(self._cb_synced_camera)
        self.create_subscription(String, p('selected_object_topic').value,
                                 self._cb_selected_object, 10)
        self.create_subscription(String, '/pick_place_state',
                                 self._cb_pick_place_state, 10)

        # ── 발행 ────────────────────────────────────────────────────────
        self.pub_pose = self.create_publisher(PoseStamped,
                                              '/detected_object_pose', 10)
        self.pub_selected_pose = self.create_publisher(PoseStamped,
                                                       '/selected_object_pose', 10)
        # 선택된 물체의 클래스명 — pick_place가 물체별 그리퍼 강도를 정할 때 사용한다.
        # 좌표(pub_selected_pose)와 함께 발행해 자동/수동 선택 모두에서 라벨을 알 수 있게 한다.
        self.pub_selected_class = self.create_publisher(String, '/selected_object_class', 10)
        self.pub_objects = self.create_publisher(String, '/detected_objects', 10)
        self.pub_debug = self.create_publisher(
            Image, '/detection_debug_image', qos_profile_sensor_data
        )

        self.get_logger().info('컬러/뎁스/카메라정보 토픽 동기화 대기 중...')
        self.get_logger().info('ObjectDetectorNode 시작')

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
            try:
                matches.extend(p for p in root.rglob(model_path.name) if p.is_file())
            except OSError:
                pass

        if matches:
            if len(matches) > 1:
                suffix = model_path.as_posix()
                for match in matches:
                    if match.as_posix().endswith(suffix):
                        return str(match.resolve())
            return str(matches[0].resolve())

        return str((self._repo_root() / model_path).resolve())

    def _resolve_tracker_yaml(self, name: str) -> str | None:
        """tracker yaml 파일을 우선 share/<pkg>/config에서 찾고 없으면 후보 루트에서 검색.
        못 찾으면 None 반환 → ultralytics .track()에 None을 넘기면 기본 botsort.yaml 사용."""
        if not name:
            return None
        cand: list[Path] = []
        try:
            from ament_index_python.packages import get_package_share_directory
            share = Path(get_package_share_directory('dsr_realsense_pick_place'))
            cand.append(share / 'config' / name)
        except Exception:
            pass
        for root in self._candidate_search_roots():
            cand.append(root / 'config' / name)
            cand.append(root / name)
        for p in cand:
            if p.is_file():
                return str(p.resolve())
        return None

    # ────────────────────────────────────────────────────────────────────
    # YOLO 로드
    # ────────────────────────────────────────────────────────────────────
    def _load_yolo(self, model_name: str | None = None, keep_previous: bool = False):

        model_name = str(model_name if model_name is not None else self.get_parameter('yolo_model').value).strip()
        model_name = self._resolve_model_name(model_name)
        old_model = self.model
        old_use_yolo = self.use_yolo

        try:
            from ultralytics import YOLO
            # model_name 이 파일 경로면 로컬 파일을, 문자열이면 기본 weight 이름을 읽는다.
            self.model = YOLO(model_name)
            self.use_yolo = True
            # 모델이 바뀌면 tracker 내부 상태가 다른 모델 기반이라 표시 번호도 비워야 일관성 유지.
            if hasattr(self, '_track_manager'):
                self._track_manager.reset()
            self.get_logger().info(f'YOLO 모델 로드 완료: {model_name}')
            return True
        except ImportError:
            self.get_logger().warn(
                'ultralytics 패키지 없음 → 색상 기반 검출로 전환. '
                '설치: pip install ultralytics'
            )
            if keep_previous:
                self.model = old_model
                self.use_yolo = old_use_yolo
            else:
                self.use_yolo = False
            return False
        except Exception as e:
            self.get_logger().warn(
                f'YOLO 모델 로드 실패({model_name}): {e} '
                '→ 색상 기반 검출로 전환'
            )
            if keep_previous:
                self.model = old_model
                self.use_yolo = old_use_yolo
            else:
                self.use_yolo = False
            return False

    def _warn_class_alignment(self) -> None:
        """known_classes(우리 설정)와 model.names(모델이 실제 학습한 클래스) 정합성 점검.
        둘이 어긋나면 라벨이 의도와 다르게 나갈 수 있으므로 시작 시 warn 발행."""
        if not self.model:
            return
        try:
            model_classes = {str(v).strip() for v in self.model.names.values()}
        except Exception:
            return
        if not model_classes:
            return
        known = set(self._known_classes)
        # (a) 모델이 알지만 known_classes 미등록 → unknown_N으로 표시될 클래스 (정보)
        extra_in_model = model_classes - known - {'object'}
        if extra_in_model:
            self.get_logger().warn(
                f'⚠️ 모델은 학습했지만 known_classes 미등록 = {sorted(extra_in_model)} '
                f'→ unknown_N으로 표시됨. yaml의 known_classes/grip_class_names 에 추가하세요.')
        # (b) known_classes에 적었지만 모델이 모름 → 검출 자체 안 됨 (경고)
        missing_in_model = known - model_classes
        if missing_in_model:
            self.get_logger().warn(
                f'⚠️ known_classes 등록됐지만 모델이 학습 안 함 = {sorted(missing_in_model)} '
                f'→ 검출 자체가 안 됨. yaml에서 빼거나 모델 재학습 필요.')
        if not extra_in_model and not missing_in_model:
            self.get_logger().info(
                f'✅ known_classes ↔ model.names 정합 OK (학습된 클래스 = {sorted(model_classes)})')

    def _on_parameters_changed(self, params):
        calib_updates = {}
        model_update = None
        for param in params:
            if param.name in (
                'absolute_calib_x_mm',
                'absolute_calib_y_mm',
                'absolute_calib_z_mm',
            ):
                try:
                    calib_updates[param.name] = float(param.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{param.name}: 숫자 값이 필요합니다',
                    )
            elif param.name == 'yolo_model':
                model_update = str(param.value).strip()
            elif param.name == 'confidence_threshold':
                # YOLO 검출 임계. _detect_yolo가 매 프레임 self.conf_thresh를 읽으므로 즉시 반영.
                try:
                    new_val = float(param.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False, reason='confidence_threshold: 숫자 값 필요')
                if not (0.0 <= new_val <= 1.0):
                    return SetParametersResult(
                        successful=False, reason='confidence_threshold: 0.0~1.0 범위')
                self.conf_thresh = new_val
                self.get_logger().info(f'confidence_threshold → {new_val:.2f}')
            elif param.name == 'known_classes':
                # 정답 클래스 set 라이브 갱신 — 다음 프레임부터 라벨 prefix 적용 바뀜.
                # GUI 강도 편집기 적용 시 동기 호출되어 sync.
                try:
                    raw = list(param.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False, reason='known_classes: 문자열 배열 필요')
                new_known = {str(c).strip() for c in raw if c and str(c).strip()}
                new_known.discard('object')
                self._known_classes = new_known
                self.get_logger().info(
                    f'known_classes 갱신 → {sorted(new_known) if new_known else "(빈 set)"}')
                # 변경 즉시 alignment 재확인 (모델은 그대로라 mismatch 체크 결과만 바뀜)
                self._warn_class_alignment()

        if model_update is not None:
            if not model_update:
                return SetParametersResult(successful=False, reason='yolo_model 경로가 비어 있습니다')
            if not self._load_yolo(model_update, keep_previous=True):
                return SetParametersResult(
                    successful=False,
                    reason='YOLO 모델 로드 실패. 기존 모델을 유지합니다',
                )

        if 'absolute_calib_x_mm' in calib_updates:
            self.abs_calib_x_m = calib_updates['absolute_calib_x_mm'] / 1000.0
        if 'absolute_calib_y_mm' in calib_updates:
            self.abs_calib_y_m = calib_updates['absolute_calib_y_mm'] / 1000.0
        if 'absolute_calib_z_mm' in calib_updates:
            self.abs_calib_z_m = calib_updates['absolute_calib_z_mm'] / 1000.0

        if calib_updates:
            self.get_logger().info(
                '수동 캘리브레이션 갱신: '
                f'X={self.abs_calib_x_m * 1000.0:.1f}mm, '
                f'Y={self.abs_calib_y_m * 1000.0:.1f}mm, '
                f'Z={self.abs_calib_z_m * 1000.0:.1f}mm'
            )

        return SetParametersResult(successful=True)

    # ────────────────────────────────────────────────────────────────────
    # 콜백
    # ────────────────────────────────────────────────────────────────────
    def _cb_synced_camera(self, color_msg: Image, depth_msg: Image, info_msg: CameraInfo):
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
        self._detect_and_publish()

    def _cb_selected_object(self, msg: String):
        # 빈 문자열이면 자동 선택 모드로 간주한다.
        self.selected_object_label = msg.data.strip()
        if self.selected_object_label != self.last_logged_selected_label:
            label_text = self.selected_object_label if self.selected_object_label else '자동 선택'
            self.get_logger().info(f'선택 물체 변경: {label_text}')
            self.last_logged_selected_label = self.selected_object_label

    def _cb_pick_place_state(self, msg: String):
        self._pick_place_state = msg.data.strip()

    # ────────────────────────────────────────────────────────────────────
    # 메인 검출 루프
    # ────────────────────────────────────────────────────────────────────
    def _detect_and_publish(self):
        # 검출 전에 최소한 컬러/깊이/카메라 파라미터가 모두 준비돼 있어야 한다.
        if self.latest_cv_color is None or self.latest_cv_depth_mm is None:
            self.get_logger().warn('이미지 미수신 (color or depth None)', throttle_duration_sec=3.0)
            return
        if self.intrinsics is None:
            self.get_logger().warn('RealSense intrinsics 미수신', throttle_duration_sec=3.0)
            return

        color_img = self.latest_cv_color.copy()
        depth_img = self.latest_cv_depth_mm

        # YOLO를 우선 사용하고, 불가능하면 간단한 색상 기반 검출로 대체한다.
        detections = (self._detect_yolo(color_img) if self.use_yolo and self.model
                      else self._detect_color(color_img))

        debug_img = color_img.copy()
        candidates = []
        now_t = time.monotonic()

        for u, v, w, h, label_class, conf, tid in detections:
            # bbox 중심 주변에서 안정적인 depth 대표값을 먼저 구한다.
            depth_m = self._estimate_depth_m(depth_img, u, v)
            if depth_m is None:
                continue

            # 픽셀 좌표를 RealSense optical frame 3D 좌표로 바꾼다.
            pose_optical = self._pixel_to_optical_pose(u, v, depth_m)
            pose_abs = self._to_absolute_pose(pose_optical)
            if pose_abs is None:
                continue

            yaw_deg = None
            if self.use_object_yaw_for_grasp:
                yaw_deg = self._estimate_object_yaw_deg(depth_img, u, v, w, h, depth_m)
                if yaw_deg is not None and self.use_manual_absolute_origin:
                    self._set_pose_yaw_deg(pose_abs, yaw_deg)

            pos = pose_abs.pose.position

            # 추적 manager로 표시 번호 결정 (yolo 결과만 — color fallback은 tid=None이라 그대로 사용)
            if tid is not None:
                self._stats_raw += 1
                display_num, should_show = self._track_manager.update(
                    tid, label_class, (pos.x, pos.y, pos.z), now=now_t)
                if not should_show:
                    continue
                self._stats_shown += 1
            else:
                display_num = None

            # 임시 라벨 — 아래에서 클래스별 카운트 보고 단독이면 prefix만, 여럿이면 prefix_N로 최종 결정
            is_known = label_class != 'object' and label_class in self._known_classes
            prefix = label_class if is_known else 'unknown'
            display_label = f'{prefix}_{display_num}' if display_num is not None else prefix

            # 디버그 영상은 표시 번호 기반으로 그린다 (사용자가 보는 번호와 일치)
            cv2.rectangle(debug_img,
                          (u - w // 2, v - h // 2),
                          (u + w // 2, v + h // 2),
                          (0, 255, 0), 2)
            cv2.putText(debug_img,
                        f'{display_label} {label_class} {conf:.2f} | {depth_m:.3f}m',
                        (u - w // 2, v - h // 2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            candidate = {
                'label': display_label,      # GUI 표시 + _choose_target 필터용 (count 보고 아래에서 재확정)
                'class_name': label_class,   # 원본 yolo 클래스 (그리퍼 강도 룩업)
                'tracker_id': tid,           # ultralytics tracker id (디버그/GUI 안정화용)
                'display_num': display_num,  # 클래스 내 표시 번호 (None = color fallback)
                'confidence': conf,
                'depth_m': depth_m,
                'pixel_u': u,
                'pixel_v': v,
                'pose': pose_abs,
                'pose_dict': {
                    'x': pos.x,
                    'y': pos.y,
                    'z': pos.z,
                    'yaw_deg': yaw_deg,
                },
            }
            candidates.append(candidate)
            # manager에 캐시 — grace 동안(다음 프레임 검출 빠질 때) 같은 후보를 재발행.
            if tid is not None:
                self._track_manager.set_payload(tid, candidate)

            if yaw_deg is not None:
                cv2.putText(debug_img,
                            f'yaw {yaw_deg:+.1f} deg',
                            (u - w // 2, v + h // 2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 2)

        # 이번 프레임에 검출 안 됐지만 grace 안이라 GUI엔 유지해야 하는 트랙들을 추가 발행.
        # _choose_target에도 같이 넘겨, 사용자가 그 사이 클릭해도 마지막 좌표로 처리 가능.
        for _tid, _entry in self._track_manager.visible_lost_tracks(now=now_t):
            cached = _entry.get('payload')
            if cached is not None:
                candidates.append(cached)
                self._stats_lost_grace += 1
        self._track_manager.cleanup_expired(now=now_t)

        # ── 통계 로그 (매 N 프레임마다) ──
        if self._stats_period > 0:
            self._stats_frame += 1
            if self._stats_frame >= self._stats_period:
                total_box = self._stats_raw + self._stats_no_id
                self.get_logger().info(
                    f'[검출통계 {self._stats_frame}프레임] '
                    f'tracker 통과={self._stats_raw}, '
                    f'box.id=None skip={self._stats_no_id} '
                    f'(전체 {total_box}건 중 {self._stats_no_id*100//max(total_box,1)}% drop), '
                    f'GUI 노출(appear OK)={self._stats_shown}, grace 유지={self._stats_lost_grace}'
                )
                self._stats_frame = 0
                self._stats_raw = self._stats_no_id = self._stats_shown = self._stats_lost_grace = 0

        # ── 카운트 기반 라벨 재확정 ─────────────────────────────────────
        # 클래스 안에 보이는 인스턴스가 1개면 "doll" / 2+면 "doll_1","doll_2".
        # candidate dict는 manager의 payload와 같은 참조라 여기서 수정하면
        # 다음 프레임 grace 재발행 시에도 동일 라벨이 유지됨.
        from collections import Counter
        class_counts = Counter(c.get('class_name', '') for c in candidates)
        for c in candidates:
            cls = c.get('class_name', '')
            is_known = cls != 'object' and cls in self._known_classes
            prefix = cls if is_known else 'unknown'
            disp = c.get('display_num')
            if class_counts.get(cls, 0) <= 1 or disp is None:
                c['label'] = prefix
            else:
                c['label'] = f'{prefix}_{disp}'

        # GUI 버튼 위치 안정화 — 클래스명 + display_num 순으로 정렬해서 발행 순서 일관
        candidates.sort(key=lambda c: (
            c.get('class_name', ''),
            c.get('display_num') if c.get('display_num') is not None else 0,
        ))
        self._publish_detected_objects(candidates)

        # 디버그 영상은 GUI와 현장 확인용으로 별도 토픽에 내보낸다.
        self.pub_debug.publish(
            self.bridge.cv2_to_imgmsg(debug_img, 'bgr8'))

        selected = self._choose_target(candidates)
        if selected is None:
            return

        # 선택 결과는 "일반 검출 결과"와 "실제 pick 대상으로 쓸 결과"를 둘 다 발행한다.
        pose_base = selected['pose']
        pos = pose_base.pose.position
        self.pub_pose.publish(pose_base)
        self.pub_selected_pose.publish(pose_base)
        # 그리퍼 강도 룩업용 — 표시 라벨([1]) 아닌 원본 클래스 이름을 발행
        self.pub_selected_class.publish(String(data=selected.get('class_name', selected['label'])))
        self.get_logger().info(
            f'[{selected["label"]}] 절대좌표: '
            f'x={pos.x:.3f} y={pos.y:.3f} z={pos.z:.3f} m '
            f'yaw={self._pose_yaw_deg(selected["pose"]):+.1f} deg',
            throttle_duration_sec=1.0,
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

    def _estimate_object_yaw_deg(
        self,
        depth_img: np.ndarray,
        u: int,
        v: int,
        box_w: int,
        box_h: int,
        depth_m: float,
    ) -> float | None:
        """깊이 밴드 마스크의 2D PCA로 물체의 평면 yaw를 추정한다.

        yaw_axis_reference:
          - long  -> PCA 긴축(major axis) 기준
          - short -> PCA 단축(minor axis) 기준
        """
        if not math.isfinite(depth_m):
            return None

        h_img, w_img = depth_img.shape[:2]
        x0 = max(0, u - box_w // 2)
        x1 = min(w_img, u + box_w // 2 + 1)
        y0 = max(0, v - box_h // 2)
        y1 = min(h_img, v + box_h // 2 + 1)
        if x1 - x0 < 6 or y1 - y0 < 6:
            return None

        roi = depth_img[y0:y1, x0:x1].astype(np.float32) * self.depth_scale
        valid = roi > 0.0
        valid &= np.abs(roi - float(depth_m)) <= self.yaw_depth_band_m
        if int(np.count_nonzero(valid)) < self.yaw_min_mask_pixels:
            return None

        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        mask[y0:y1, x0:x1] = valid.astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        ys, xs = np.nonzero(mask)
        if ys.size < self.yaw_min_mask_pixels:
            return None

        pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        mean, eigvecs = cv2.PCACompute(pts, mean=None, maxComponents=2)
        if eigvecs is None or eigvecs.shape[0] == 0:
            return None

        axis_index = 1 if self.yaw_axis_reference == 'short' and eigvecs.shape[0] > 1 else 0
        axis_u = float(eigvecs[axis_index][0])
        axis_v = float(eigvecs[axis_index][1])

        # 이미지 축(u right, v down) -> project camera XY(x left, y down)
        proj_x = axis_v
        proj_y = -axis_u
        yaw_deg = math.degrees(math.atan2(proj_y, proj_x))
        return self._normalize_grasp_yaw_deg(yaw_deg)

    def _normalize_grasp_yaw_deg(self, yaw_deg: float) -> float:
        """그리퍼 180도 대칭을 고려해 yaw를 [-90, 90) 범위로 접는다."""
        wrapped = ((float(yaw_deg) + 180.0) % 360.0) - 180.0
        if wrapped >= 90.0:
            wrapped -= 180.0
        if wrapped < -90.0:
            wrapped += 180.0
        return wrapped

    def _set_pose_yaw_deg(self, pose: PoseStamped, yaw_deg: float):
        yaw_rad = math.radians(float(yaw_deg))
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw_rad * 0.5)
        pose.pose.orientation.w = math.cos(yaw_rad * 0.5)

    def _pose_yaw_deg(self, pose: PoseStamped) -> float:
        qz = float(pose.pose.orientation.z)
        qw = float(pose.pose.orientation.w)
        return math.degrees(2.0 * math.atan2(qz, qw))

    # ────────────────────────────────────────────────────────────────────
    # 픽셀 + depth → RealSense optical frame PoseStamped
    # ────────────────────────────────────────────────────────────────────
    def _pixel_to_optical_pose(self, u: int, v: int, depth_m: float) -> PoseStamped:
        """픽셀 좌표 (u, v)와 깊이 depth_m(m)을 RealSense optical 3D 좌표로 변환한다.

        RealSense SDK의 rs2_deproject_pixel_to_point()는 핀홀 모델 + 왜곡 보정을 적용해
        픽셀 좌표를 카메라 광학 좌표계(camera_color_optical_frame)의 3D 점으로 변환한다.
        """
        # rs2_deproject_pixel_to_point: (intrinsics, [u, v], depth_m) → [X, Y, Z]
        # 내부적으로 수식: X = (u - ppx) / fx * Z, Y = (v - ppy) / fy * Z
        X, Y, Z = rs.rs2_deproject_pixel_to_point(
            self.intrinsics,
            [float(u), float(v)],
            float(depth_m),
        )

        ps = PoseStamped()
        ps.header.frame_id = self.camera_frame   # 'camera_color_optical_frame'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = X
        ps.pose.position.y = Y
        ps.pose.position.z = Z
        ps.pose.orientation.w = 1.0   # 방향은 단위 quaternion (회전 없음)
        return ps

    def _optical_to_project_camera_pose(self, pose_optical: PoseStamped) -> PoseStamped:
        """RealSense optical frame을 yolo_live_cam_3d_metrics 프로젝트 좌표계로 바꾼다.

        변환 규칙은 yolo_live_cam_3d_metrics / gui_node와 동일하다.
          X: 왼쪽 (+)   = -optical_x
          Y: 아래쪽 (+) =  optical_y
          Z: 카메라 뒤쪽 (+) = -optical_z
        """
        pose_project = PoseStamped()
        pose_project.header = pose_optical.header
        pose_project.header.frame_id = 'project_camera_frame'
        pose_project.pose.position.x = -pose_optical.pose.position.x
        pose_project.pose.position.y = pose_optical.pose.position.y
        pose_project.pose.position.z = -pose_optical.pose.position.z
        pose_project.pose.orientation = pose_optical.pose.orientation
        return pose_project

    # ────────────────────────────────────────────────────────────────────
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

    def _to_absolute_pose(self, pose_optical: PoseStamped):
        """카메라 좌표를 절대좌표로 변환한다.

        use_manual_absolute_origin=True:
          yolo_live_cam_3d_metrics와 같은 프로젝트 카메라 좌표계로 먼저 바꾼 뒤
          절대원점의 카메라좌표(ox, oy, oz)를 사용해
          p_abs = p_project_cam - o_project_cam
        use_manual_absolute_origin=False:
          RealSense optical frame 기준 TF(camera -> robot_base_frame) 변환 사용
        """
        if not self.use_manual_absolute_origin:
            return self._transform_to_base(pose_optical)

        pose_cam = self._optical_to_project_camera_pose(pose_optical)

        pose_abs = PoseStamped()
        pose_abs.header = pose_cam.header
        # 수동 원점 방식의 결과도 pick_place_node에서는 로봇 베이스 기준
        # 절대좌표로 사용하므로 frame_id를 base frame과 일치시킨다.
        pose_abs.header.frame_id = self.robot_base_frame
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

    def _publish_detected_objects(self, candidates: list):
        # GUI가 별도 커스텀 메시지 없이 바로 읽을 수 있도록 JSON 문자열로 묶어 발행한다.
        # label은 표시용([1] 등), class_name은 원본 yolo 클래스, tracker_id는 안정 키(GUI가
        # 이 값을 키로 쓰면 시각적 깜빡임 추가 억제 가능).
        msg = String()
        msg.data = json.dumps({
            'selected_label': self.selected_object_label,
            'objects': [
                {
                    'label': item['label'],
                    'class_name': item.get('class_name', item['label']),
                    'tracker_id': item.get('tracker_id'),
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

    def _choose_target(self, candidates: list):
        """사용자가 라벨을 골랐으면 그 라벨로 우선 정확 매칭, 없으면 prefix(=클래스 그룹)로 fallback.
        둘 다 없으면 자동 선택(가장 가까운 것)."""
        sel = self.selected_object_label
        if sel:
            # 1차: 정확 매칭 (예: 사용자가 "doll_2" 또는 "doll" 누름 → 같은 라벨)
            exact = [item for item in candidates if item['label'] == sel]
            if exact:
                return min(exact, key=lambda item: item['depth_m'])
            # 2차: prefix 매칭 — 사장님이 "doll" 누른 사이 doll이 늘어 "doll_1","doll_2"가 됐을 때
            # 또는 "doll_2"가 사라지고 단독 "doll"이 됐을 때. 같은 클래스 그룹 내에서 가장 가까운 것.
            prefix = sel.rsplit('_', 1)[0] if '_' in sel else sel
            prefixed = [
                item for item in candidates
                if item['label'] == prefix or item['label'].startswith(f'{prefix}_')
            ]
            if prefixed:
                if self._pick_place_state not in {'LIFT', 'MOVE_TO_PLACE'}:
                    self.get_logger().info(
                        f'선택({sel}) 정확 매칭 없음 → 같은 클래스({prefix}) 그룹에서 가장 가까운 것 선택',
                        throttle_duration_sec=2.0,
                    )
                return min(prefixed, key=lambda item: item['depth_m'])
            # 아무것도 없음
            _suppress_states = {'LIFT', 'MOVE_TO_PLACE'}
            if self._pick_place_state not in _suppress_states:
                self.get_logger().warn(
                    f'선택한 물체({sel})가 현재 화면에서 검출되지 않음',
                    throttle_duration_sec=2.0,
                )
            return None
        # 자동 선택 — 가장 가까운 것
        if not candidates:
            return None
        return min(candidates, key=lambda item: item['depth_m'])

    # ────────────────────────────────────────────────────────────────────
    # YOLO 검출
    # ────────────────────────────────────────────────────────────────────
    def _detect_yolo(self, img: np.ndarray) -> list:
        """YOLOv8 + 추적으로 이미지에서 물체를 검출.

        반환: [(u, v, w, h, label_class, conf, tracker_id), ...]
          - label_class는 yolo의 원본 클래스 이름(예: 'object'/'doll').
          - tracker_id는 ultralytics BoT-SORT/ByteTrack이 부여하는 고유 ID(int).
          - tracker가 ID 미부여 상태(첫 프레임 일부)면 그 검출은 skip.
        """
        track_kwargs = dict(conf=self.conf_thresh, persist=True, verbose=False)
        if self._tracker_yaml_path:
            track_kwargs['tracker'] = self._tracker_yaml_path
        results = self.model.track(img, **track_kwargs)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                # tracker가 아직 ID를 부여 못한 검출은 안정적이지 않으므로 skip
                if box.id is None:
                    self._stats_no_id += 1
                    continue
                cls_id = int(box.cls[0])
                label_class = self.model.names[cls_id]   # 원본 클래스 이름
                # target_classes가 빈 리스트이면 모든 클래스 통과
                if self.target_classes and label_class not in self.target_classes:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                u = (x1 + x2) // 2
                v = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                tid = int(box.id[0])
                detections.append((u, v, w, h, label_class, conf, tid))
        return detections

    # ────────────────────────────────────────────────────────────────────
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
            detections.append((u, v, w, h, 'red_object', 1.0, None))
        return detections


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
