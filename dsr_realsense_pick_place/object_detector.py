"""
object_detector.py
------------------
RealSense RGB-D + YOLOv8 기반 객체 검출 노드.

구독:
  /camera/color/image_raw          (sensor_msgs/Image)
  /camera/aligned_depth_to_color/image_raw  (sensor_msgs/Image)
  /camera/color/camera_info        (sensor_msgs/CameraInfo)

발행:
  /detected_object_pose            (geometry_msgs/PoseStamped) - 선택된 물체의 로봇 베이스 좌표
  /selected_object_pose            (geometry_msgs/PoseStamped) - GUI가 선택한 물체의 좌표
  /detected_objects                (std_msgs/String)           - 검출 물체 목록(JSON)
  /detection_debug_image           (sensor_msgs/Image)         - 디버그 시각화
"""

import json

import rclpy
from rclpy.node import Node
import rclpy.duration
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


class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector')

        # ── 파라미터 선언 ────────────────────────────────────────────────
        # 토픽/프레임 이름을 파라미터로 빼 두면 launch 나 yaml 에서 쉽게 바꿀 수 있다.
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('use_yolo', True)
        self.declare_parameter('yolo_model', 'yolov8s.pt')
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('target_classes', ['bottle', 'cup', 'bowl'])
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('min_depth_m', 0.15)
        self.declare_parameter('max_depth_m', 1.5)
        self.declare_parameter('depth_sample_radius', 5)
        self.declare_parameter('depth_center_ratio', 0.6)
        self.declare_parameter('depth_outlier_mad_scale', 2.5)
        self.declare_parameter('selected_object_topic', '/selected_object_label')

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
        self.selected_object_label = ''
        self.last_logged_selected_label = None

        # ── 카메라 내부 파라미터 (camera_info 수신 전까지 None) ─────────
        self.intrinsics = None

        # ── YOLO 모델 로드 ───────────────────────────────────────────────
        self.model = None
        if self.use_yolo:
            self._load_yolo()

        # ── TF2 ─────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── CvBridge ────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── 이미지 버퍼 ─────────────────────────────────────────────────
        self.latest_cv_color = None
        self.latest_cv_depth_mm = None

        # ── 구독 ────────────────────────────────────────────────────────
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
            queue_size=10,
            slop=0.1,
        )
        self.ts.registerCallback(self._cb_synced_camera)
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

        self.get_logger().info('컬러/뎁스/카메라정보 토픽 동기화 대기 중...')
        self.get_logger().info('ObjectDetectorNode 시작')

    # ────────────────────────────────────────────────────────────────────
    # YOLO 로드
    # ────────────────────────────────────────────────────────────────────
    def _load_yolo(self):
        model_name = self.get_parameter('yolo_model').value
        try:
            from ultralytics import YOLO
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

    # ────────────────────────────────────────────────────────────────────
    # 콜백
    # ────────────────────────────────────────────────────────────────────
    def _cb_synced_camera(self, color_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        try:
            self.latest_cv_color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            self.latest_cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge 변환 오류: {e}', throttle_duration_sec=3.0)
            return

        if self.intrinsics is None:
            intr = rs.intrinsics()
            intr.width = info_msg.width
            intr.height = info_msg.height
            intr.ppx = info_msg.k[2]
            intr.ppy = info_msg.k[5]
            intr.fx = info_msg.k[0]
            intr.fy = info_msg.k[4]
            if info_msg.distortion_model in ('plumb_bob', 'rational_polynomial'):
                intr.model = rs.distortion.brown_conrady
            else:
                intr.model = rs.distortion.none
            intr.coeffs = list(info_msg.d)
            self.intrinsics = intr
            self.get_logger().info('카메라 내장 파라미터(Intrinsics) 수신 완료.')

        self._detect_and_publish()

    def _cb_selected_object(self, msg: String):
        # 빈 문자열이면 자동 선택 모드로 간주한다.
        self.selected_object_label = msg.data.strip()
        if self.selected_object_label != self.last_logged_selected_label:
            label_text = self.selected_object_label if self.selected_object_label else '자동 선택'
            self.get_logger().info(f'선택 물체 변경: {label_text}')
            self.last_logged_selected_label = self.selected_object_label

    # ────────────────────────────────────────────────────────────────────
    # 메인 검출 루프
    # ────────────────────────────────────────────────────────────────────
    def _detect_and_publish(self):
        if self.latest_cv_color is None or self.latest_cv_depth_mm is None:
            self.get_logger().warn('이미지 미수신 (color or depth None)', throttle_duration_sec=3.0)
            return
        if self.intrinsics is None:
            self.get_logger().warn('RealSense intrinsics 미수신', throttle_duration_sec=3.0)
            return

        color_img = self.latest_cv_color.copy()
        depth_img = self.latest_cv_depth_mm

        # ── 검출 ────────────────────────────────────────────────────────
        detections = (self._detect_yolo(color_img) if self.use_yolo and self.model
                      else self._detect_color(color_img))

        debug_img = color_img.copy()
        candidates = []

        for u, v, w, h, label, conf in detections:
            depth_m = self._estimate_depth_m(depth_img, u, v)
            if depth_m is None:
                continue

            # 시각화
            cv2.rectangle(debug_img,
                          (u - w // 2, v - h // 2),
                          (u + w // 2, v + h // 2),
                          (0, 255, 0), 2)
            cv2.putText(debug_img,
                        f'{label} {conf:.2f} | {depth_m:.3f}m',
                        (u - w // 2, v - h // 2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            pose_cam = self._pixel_to_camera_pose(u, v, depth_m)
            pose_base = self._transform_to_base(pose_cam)
            if pose_base is None:
                continue

            pos = pose_base.pose.position
            candidates.append({
                'label': label,
                'confidence': conf,
                'depth_m': depth_m,
                'pixel_u': u,
                'pixel_v': v,
                'pose': pose_base,
                'pose_dict': {
                    'x': pos.x,
                    'y': pos.y,
                    'z': pos.z,
                },
            })

        self._publish_detected_objects(candidates)

        self.pub_debug.publish(
            self.bridge.cv2_to_imgmsg(debug_img, 'bgr8'))

        selected = self._choose_target(candidates)
        if selected is None:
            return

        pose_base = selected['pose']
        pos = pose_base.pose.position
        self.pub_pose.publish(pose_base)
        self.pub_selected_pose.publish(pose_base)
        self.get_logger().info(
            f'[{selected["label"]}] 로봇베이스 좌표: '
            f'x={pos.x:.3f} y={pos.y:.3f} z={pos.z:.3f} m'
        )

    def _estimate_depth_m(self, depth_img: np.ndarray, u: int, v: int):
        # bbox 중심 근처 depth를 모아 outlier를 제거한 뒤 대표값을 사용한다.
        r = max(1, int(self.depth_r))
        h_img, w_img = depth_img.shape[:2]
        x0 = max(0, u - r)
        x1 = min(w_img, u + r + 1)
        y0 = max(0, v - r)
        y1 = min(h_img, v + r + 1)
        roi = depth_img[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        yy, xx = np.indices(roi.shape)
        center_y = v - y0
        center_x = u - x0
        dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        max_dist = max(1.0, float(r) * max(0.1, self.depth_center_ratio))

        valid_mask = roi > 0
        valid_mask &= dist <= max_dist
        samples = roi[valid_mask].astype(np.float32) * self.depth_scale
        if samples.size == 0:
            return None

        samples = samples[(samples >= self.min_depth) & (samples <= self.max_depth)]
        if samples.size == 0:
            return None

        median = float(np.median(samples))
        abs_dev = np.abs(samples - median)
        mad = float(np.median(abs_dev))

        if mad > 0.0:
            filtered = samples[abs_dev <= self.depth_outlier_mad_scale * mad]
            if filtered.size > 0:
                samples = filtered

        return float(np.median(samples))

    # ────────────────────────────────────────────────────────────────────
    # 픽셀 + depth → 카메라 프레임 PoseStamped
    # ────────────────────────────────────────────────────────────────────
    def _pixel_to_camera_pose(self, u: int, v: int, depth_m: float) -> PoseStamped:
        X, Y, Z = rs.rs2_deproject_pixel_to_point(
            self.intrinsics,
            [float(u), float(v)],
            float(depth_m),
        )

        ps = PoseStamped()
        ps.header.frame_id = self.camera_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = X
        ps.pose.position.y = Y
        ps.pose.position.z = Z
        ps.pose.orientation.w = 1.0
        return ps

    # ────────────────────────────────────────────────────────────────────
    # 카메라 프레임 → 로봇 베이스 프레임 변환
    # ────────────────────────────────────────────────────────────────────
    def _transform_to_base(self, pose_cam: PoseStamped):
        try:
            # TF tree 에 등록된 정적/동적 변환을 이용해 camera_frame → base_link 로 바꾼다.
            pose_base = self.tf_buffer.transform(
                pose_cam,
                self.robot_base_frame,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            return pose_base
        except Exception as e:
            self.get_logger().warn(f'TF 변환 실패: {e}')
            return None

    def _publish_detected_objects(self, candidates: list):
        # GUI 에서 쉽게 읽을 수 있도록 JSON 문자열로 물체 목록을 보낸다.
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

    def _choose_target(self, candidates: list):
        # 선택한 라벨이 있으면 그 라벨 중 가장 가까운 물체를 사용한다.
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

    # ────────────────────────────────────────────────────────────────────
    # YOLO 검출
    # ────────────────────────────────────────────────────────────────────
    def _detect_yolo(self, img: np.ndarray) -> list:
        # 결과는 (중심 u, 중심 v, 폭, 높이, 라벨, confidence) 형식으로 통일한다.
        results = self.model(img, conf=self.conf_thresh, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                if self.target_classes and label not in self.target_classes:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                u = (x1 + x2) // 2
                v = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                detections.append((u, v, w, h, label, conf))
        return detections

    # ────────────────────────────────────────────────────────────────────
    # 색상 기반 검출 (YOLO 없을 때 fallback – 빨간 물체 검출)
    # ────────────────────────────────────────────────────────────────────
    def _detect_color(self, img: np.ndarray) -> list:
        # 데모용 단순 fallback 이므로 빨간색 물체만 찾는다.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 빨간색: 두 범위 합산
        m1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 100, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 800:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            u = x + w // 2
            v = y + h // 2
            detections.append((u, v, w, h, 'red_object', 1.0))
        return detections


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
