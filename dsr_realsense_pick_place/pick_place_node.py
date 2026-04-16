"""
pick_place_node.py
------------------
Doosan E0509 Pick & Place 상태머신 노드.

상태 전이 흐름:
  IDLE → DETECTING → PRE_PICK → PICK → LIFT → MOVE_TO_PLACE → PLACE → POST_PLACE → HOME → IDLE (반복)
  → ERROR : 예외 발생 시 수동 복구 대기

구독:
  /selected_object_pose  (geometry_msgs/PoseStamped)

발행:
  /pick_place_state        (std_msgs/String)
  /gripper/rh12_stroke_cmd (std_msgs/Int32)

Doosan 서비스 클라이언트 (namespace: /dsr01/):
  motion/move_joint        (dsr_msgs2/MoveJoint)
  motion/move_line         (dsr_msgs2/MoveLine)
  gripper/serial_send_data (dsr_msgs2/SerialSendData)

그리퍼: ROBOTIS RH-P12-Rn (Modbus RTU over serial)
"""

import threading
import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger

from dsr_msgs2.srv import MoveJoint, MoveLine, SerialSendData


# ── 상태 정의 ───────────────────────────────────────────────────────────
class State(Enum):
    """Pick & Place 작업 단계를 나타내는 상태 열거형.

    IDLE          : 초기 상태. 시작 시 홈으로 이동한 뒤 DETECTING으로 전환.
    DETECTING     : /selected_object_pose 토픽에서 유효한 타겟 좌표를 기다리는 상태.
    PRE_PICK      : 물체 위 pre_pick_z_offset 높이까지 이동. 그리퍼 미리 열기.
    PICK          : 저속(50mm/s)으로 pick_z_offset 높이까지 하강 후 그리퍼 닫기.
    LIFT          : 파지 후 PRE_PICK 높이까지 다시 상승.
    MOVE_TO_PLACE : place_position 상단(pre_place_z_offset)으로 수평 이동.
    PLACE         : 저속으로 place_position까지 하강 후 그리퍼 열기.
    POST_PLACE    : 그리퍼 오픈 후 place 위 안전 높이로 복귀.
    HOME          : 홈 관절 각도로 복귀. 다음 사이클 준비.
    ERROR         : 예외 발생 시 진입. 2초 간격으로 대기하며 수동 복구 안내.
    """
    IDLE          = auto()
    DETECTING     = auto()
    PRE_PICK      = auto()
    PICK          = auto()
    LIFT          = auto()
    MOVE_TO_PLACE = auto()
    PLACE         = auto()
    POST_PLACE    = auto()
    HOME          = auto()
    ERROR         = auto()


class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')

        # ── 파라미터 ────────────────────────────────────────────────────
        self.declare_parameter('robot_namespace',             'dsr01')
        self.declare_parameter('joint_vel',                   30.0)
        self.declare_parameter('joint_acc',                   60.0)
        self.declare_parameter('cart_vel',                    100.0)
        self.declare_parameter('cart_acc',                    200.0)
        self.declare_parameter('home_joints',                 [0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        self.declare_parameter('gripper_wait_sec',            0.8)
        self.declare_parameter('rh12_serial_service',         '/dsr01/gripper/serial_send_data')
        self.declare_parameter('rh12_bridge_topic',           '/gripper/rh12_stroke_cmd')
        self.declare_parameter('rh12_allow_missing_service',  True)
        self.declare_parameter('rh12_slave_id',               1)
        self.declare_parameter('rh12_torque_enable_register', 256)
        self.declare_parameter('rh12_goal_current_register',  275)
        self.declare_parameter('rh12_stroke_register',        282)
        self.declare_parameter('rh12_goal_current',           400)
        self.declare_parameter('rh12_open_stroke',            700)
        self.declare_parameter('rh12_close_stroke',           0)
        self.declare_parameter('rh12_port',                   1)
        self.declare_parameter('rh12_init_wait_sec',          0.1)
        self.declare_parameter('pre_pick_z_offset',           0.12)
        self.declare_parameter('pick_z_offset',               0.005)
        self.declare_parameter('grasp_rpy',                   [0.0, 180.0, 0.0])
        self.declare_parameter('place_position',              [0.4, -0.3, 0.1])
        self.declare_parameter('pre_place_z_offset',          0.15)
        self.declare_parameter('place_rpy',                   [0.0, 180.0, 0.0])
        self.declare_parameter('workspace_x_min',             0.15)
        self.declare_parameter('workspace_x_max',             0.80)
        self.declare_parameter('workspace_y_min',            -0.60)
        self.declare_parameter('workspace_y_max',             0.60)
        self.declare_parameter('workspace_z_min',             0.0)
        self.declare_parameter('workspace_z_max',             0.60)
        self.declare_parameter('target_pose_topic',           '/selected_object_pose')

        ns = self.get_parameter('robot_namespace').value
        self.jvel         = self.get_parameter('joint_vel').value
        self.jacc         = self.get_parameter('joint_acc').value
        self.cvel         = self.get_parameter('cart_vel').value
        self.cacc         = self.get_parameter('cart_acc').value
        self.home_joints  = self.get_parameter('home_joints').value
        self.gripper_wait = self.get_parameter('gripper_wait_sec').value
        self.rh12_serial_service        = self.get_parameter('rh12_serial_service').value
        self.rh12_bridge_topic          = self.get_parameter('rh12_bridge_topic').value
        self.rh12_allow_missing_service = self.get_parameter('rh12_allow_missing_service').value
        self.rh12_slave_id              = self.get_parameter('rh12_slave_id').value
        self.rh12_torque_enable_register= self.get_parameter('rh12_torque_enable_register').value
        self.rh12_goal_current_register = self.get_parameter('rh12_goal_current_register').value
        self.rh12_stroke_register       = self.get_parameter('rh12_stroke_register').value
        self.rh12_goal_current          = self.get_parameter('rh12_goal_current').value
        self.rh12_open_stroke           = self.get_parameter('rh12_open_stroke').value
        self.rh12_close_stroke          = self.get_parameter('rh12_close_stroke').value
        self.rh12_port                  = self.get_parameter('rh12_port').value
        self.rh12_init_wait             = self.get_parameter('rh12_init_wait_sec').value
        self.pre_pick_dz  = self.get_parameter('pre_pick_z_offset').value
        self.pick_dz      = self.get_parameter('pick_z_offset').value
        self.grasp_rpy    = self.get_parameter('grasp_rpy').value
        self.place_pos    = self.get_parameter('place_position').value
        self.pre_place_dz = self.get_parameter('pre_place_z_offset').value
        self.place_rpy    = self.get_parameter('place_rpy').value
        self.ws = {
            'x': (self.get_parameter('workspace_x_min').value,
                  self.get_parameter('workspace_x_max').value),
            'y': (self.get_parameter('workspace_y_min').value,
                  self.get_parameter('workspace_y_max').value),
            'z': (self.get_parameter('workspace_z_min').value,
                  self.get_parameter('workspace_z_max').value),
        }

        # ── 서비스 클라이언트 ────────────────────────────────────────────
        # ns=''이면 '/dsr01'을 기본으로 사용해 '/motion/move_joint' 오경로 방지
        prefix = f'/{ns}' if ns else '/dsr01'

        self.cli_movej         = self.create_client(MoveJoint,     f'{prefix}/motion/move_joint')
        self.cli_movel         = self.create_client(MoveLine,      f'{prefix}/motion/move_line')
        self.cli_serial_send   = self.create_client(SerialSendData, self.rh12_serial_service)
        self.cli_gripper_open  = self.create_client(Trigger, '/gripper/open')
        self.cli_gripper_close = self.create_client(Trigger, '/gripper/close')

        # robot_mode 서비스는 spin() 시작 전 __init__ 에서 미리 create_client
        from dsr_msgs2.srv import SetRobotMode
        self.cli_set_mode = self.create_client(SetRobotMode, f'{prefix}/system/set_robot_mode')

        self._wait_for_services()

        # ── 상태 변수 ───────────────────────────────────────────────────
        # state/target_pose는 상태머신 스레드와 ROS 콜백 스레드가 동시에 접근하므로 Lock 사용
        self.state       = State.IDLE
        self.state_lock  = threading.Lock()
        self.target_pose: PoseStamped | None = None
        self.rh12_initialized = False
        # rh12_lock: Open/Close 거의 동시 호출 시 Modbus 프레임 중복 발송 방지
        self.rh12_lock   = threading.Lock()

        # ── 퍼블리셔 / 구독 ─────────────────────────────────────────────
        self.pub_state       = self.create_publisher(String, '/pick_place_state', 10)
        self.pub_rh12_stroke = self.create_publisher(Int32,  self.rh12_bridge_topic, 10)
        self.create_subscription(
            PoseStamped,
            self.get_parameter('target_pose_topic').value,
            self._cb_pose, 10)

        # ── 상태머신 스레드 ─────────────────────────────────────────────
        # daemon=True: 메인 스레드(rclpy.spin) 종료 시 자동으로 함께 종료
        self.sm_thread = threading.Thread(target=self._state_machine_loop, daemon=True)
        self.sm_thread.start()

        self.get_logger().info('PickPlaceNode 시작 — 상태: IDLE')

    # ────────────────────────────────────────────────────────────────────
    # 서비스 대기 + robot_mode 설정
    # ────────────────────────────────────────────────────────────────────
    def _wait_for_services(self):
        required = [
            (self.cli_movej, 'move_joint'),
            (self.cli_movel, 'move_line'),
            (self.cli_gripper_open, 'gripper/open'),
            (self.cli_gripper_close, 'gripper/close'),
        ]
        if not self.rh12_allow_missing_service:
            required.append((self.cli_serial_send, self.rh12_serial_service))

        for cli, name in required:
            self.get_logger().info(f'서비스 대기 중: {name} ...')
            while not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f'{name} 없음, 재시도...')

        self.get_logger().info('컨트롤러 초기화 대기 중 (3초)...')
        time.sleep(3.0)
        self._set_robot_mode_auto()
        self.get_logger().info('모든 서비스 준비 완료')

    def _set_robot_mode_auto(self):
        """robot_mode=1(AUTO) 설정.

        _call_service() 폴링 방식으로 통일해 spin()과 충돌 없이 동작.
        별도 SingleThreadedExecutor를 생성하면 이미 spin() 중인 노드를
        두 번째 executor에 add_node하면서 충돌이 발생하므로 폴링 방식만 사용.
        """
        from dsr_msgs2.srv import SetRobotMode

        if not self.cli_set_mode.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('set_robot_mode 서비스 없음, 건너뜀')
            return

        req = SetRobotMode.Request()
        req.robot_mode = 1  # DR_MODE_AUTO
        try:
            self._call_service(self.cli_set_mode, req, 'set_robot_mode')
            self.get_logger().info('robot_mode=1 설정 완료 ✅')
        except Exception as e:
            self.get_logger().warn(f'robot_mode 설정 실패: {e}')

    # ────────────────────────────────────────────────────────────────────
    # 콜백: 검출 포즈 수신
    # ────────────────────────────────────────────────────────────────────
    def _cb_pose(self, msg: PoseStamped):
        with self.state_lock:
            # DETECTING 상태일 때만 새 타겟을 수신해 다음 단계로 넘어간다
            if self.state == State.DETECTING:
                pos = msg.pose.position
                if self._in_workspace(pos.x, pos.y, pos.z):
                    self.target_pose = msg
                    self.state = State.PRE_PICK
                    self.get_logger().info(
                        f'목표 설정: x={pos.x:.3f} y={pos.y:.3f} z={pos.z:.3f}')
                else:
                    self.get_logger().warn(
                        f'작업 공간 밖 무시: x={pos.x:.3f} y={pos.y:.3f} z={pos.z:.3f}')

    def _in_workspace(self, x, y, z) -> bool:
        """작업 가능 영역 검증. 영역 밖 좌표는 안전을 위해 무시."""
        return (self.ws['x'][0] <= x <= self.ws['x'][1] and
                self.ws['y'][0] <= y <= self.ws['y'][1] and
                self.ws['z'][0] <= z <= self.ws['z'][1])

    # ────────────────────────────────────────────────────────────────────
    # 상태머신 루프 (별도 스레드)
    # ────────────────────────────────────────────────────────────────────
    def _state_machine_loop(self):
        while rclpy.ok():
            with self.state_lock:
                current = self.state

            # 현재 상태를 토픽으로 발행 → 외부 모니터링 용이
            self._publish_state(current.name)

            try:
                if current == State.IDLE:
                    self.get_logger().info('홈으로 이동 후 DETECTING 대기...')
                    self._go_home()
                    self._set_state(State.DETECTING)

                elif current == State.DETECTING:
                    time.sleep(0.1)  # 포즈 콜백 대기 (CPU 점유 최소화)

                elif current == State.PRE_PICK:
                    # 물체 위 안전 높이까지 먼저 접근. 그리퍼를 미리 열어 충돌 예방.
                    pose = self.target_pose
                    self.get_logger().info('Pre-Pick 위치로 이동')
                    self._gripper_open()
                    self._move_to_cart(
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z + self.pre_pick_dz,
                        self.grasp_rpy)
                    self._set_state(State.PICK)

                elif current == State.PICK:
                    # 충돌 위험이 가장 큰 구간 → 저속(50mm/s) 접근
                    pose = self.target_pose
                    self.get_logger().info('Pick 위치로 하강')
                    self._move_to_cart(
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z + self.pick_dz,
                        self.grasp_rpy, vel=50.0, acc=100.0)
                    self._gripper_close()
                    self._set_state(State.LIFT)

                elif current == State.LIFT:
                    # 파지 후 위로 올라와 주변 장애물 간섭 최소화
                    pose = self.target_pose
                    self.get_logger().info('물체 들어올리기')
                    self._move_to_cart(
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z + self.pre_pick_dz,
                        self.grasp_rpy)
                    self._set_state(State.MOVE_TO_PLACE)

                elif current == State.MOVE_TO_PLACE:
                    # Place 위치 상단으로 수평 이동 후 최종 하강
                    self.get_logger().info('Place 위치로 이동')
                    px, py, pz = self.place_pos
                    self._move_to_cart(px, py, pz + self.pre_place_dz, self.place_rpy)
                    self._set_state(State.PLACE)

                elif current == State.PLACE:
                    px, py, pz = self.place_pos
                    self.get_logger().info('물체 내려놓기')
                    self._move_to_cart(px, py, pz, self.place_rpy, vel=50.0, acc=100.0)
                    self._gripper_open()
                    self._set_state(State.POST_PLACE)

                elif current == State.POST_PLACE:
                    px, py, pz = self.place_pos
                    self._move_to_cart(px, py, pz + self.pre_place_dz, self.place_rpy)
                    self.get_logger().info('Pick & Place 완료!')
                    self._set_state(State.HOME)

                elif current == State.HOME:
                    self._go_home()
                    self._set_state(State.IDLE)

                elif current == State.ERROR:
                    self.get_logger().error('오류 발생. 수동 복구 필요.')
                    time.sleep(2.0)

            except Exception as e:
                self.get_logger().error(f'상태머신 예외: {e}')
                self._set_state(State.ERROR)

    def _set_state(self, s: State):
        with self.state_lock:
            self.state = s
        self.get_logger().info(f'→ 상태 전환: {s.name}')

    def _publish_state(self, name: str):
        msg = String()
        msg.data = name
        self.pub_state.publish(msg)

    # ────────────────────────────────────────────────────────────────────
    # 로봇 모션 헬퍼
    # ────────────────────────────────────────────────────────────────────
    def _go_home(self):
        """홈 관절 각도로 이동.

        MoveJoint 파라미터:
          pos       : 목표 관절 각도 6개 (deg)
          vel       : 관절 속도 (deg/s)
          acc       : 관절 가속도 (deg/s²)
          time      : 0.0 → vel/acc 기반 자동 계산
          radius    : 0.0 → 블렌딩 없음
          mode      : 0 = MOVE_MODE_ABSOLUTE
          blendType : 0 (Doosan 필드명 camelCase)
          syncType  : 0 = 동기 블로킹 (Doosan 필드명 camelCase)
        """
        req = MoveJoint.Request()
        req.pos       = [float(v) for v in self.home_joints]
        req.vel       = self.jvel
        req.acc       = self.jacc
        req.time      = 0.0
        req.radius    = 0.0
        req.mode      = 0        # MOVE_MODE_ABSOLUTE
        req.blend_type = 0        # ← Doosan 공식 camelCase
        req.sync_type  = 0        # ← Doosan 공식 camelCase (동기 블로킹)
        self._call_service(self.cli_movej, req, 'move_joint(home)', timeout=30.0)

    def _move_to_cart(self, x, y, z, rpy, vel=None, acc=None):
        """Cartesian 직선 이동 (로봇 베이스 프레임 기준).

        인자:
          x, y, z : 목표 위치 (m 단위 → 내부에서 mm 변환)
          rpy     : [rx, ry, rz] 방향 (deg)
          vel     : 선속도(mm/s). None이면 cart_vel 파라미터 사용.
          acc     : 선가속도(mm/s²). None이면 cart_acc 파라미터 사용.

        Doosan MoveLine 서비스 단위:
          위치: mm (ROS 관례 m → x*1000 변환 필수)
          vel : [선속도 mm/s, 각속도 deg/s]
          acc : [선가속도 mm/s², 각가속도 deg/s²]
          ref : 0 = DR_BASE (로봇 베이스 좌표계)
          mode: 0 = DR_MV_MOD_ABS (절대 좌표)
          blendType / syncType : Doosan 공식 camelCase 필드명
        """
        req = MoveLine.Request()
        req.pos       = [x * 1000.0, y * 1000.0, z * 1000.0,
                         float(rpy[0]), float(rpy[1]), float(rpy[2])]
        req.vel       = [vel if vel else self.cvel, 30.0]   # [선속도, 각속도]
        req.acc       = [acc if acc else self.cacc, 60.0]   # [선가속도, 각가속도]
        req.time      = 0.0
        req.radius    = 0.0
        req.ref       = 0        # DR_BASE
        req.mode      = 0        # DR_MV_MOD_ABS
        req.blend_type = 0        # ← Doosan 공식 camelCase
        req.sync_type  = 0        # ← Doosan 공식 camelCase (동기 블로킹)
        self._call_service(self.cli_movel, req,
                           f'move_line({x:.3f},{y:.3f},{z:.3f})', timeout=30.0)

    def _call_service(self, cli, req, name: str, timeout: float = 15.0):
        """ROS 2 서비스를 call_async() + 폴링 방식으로 호출.

        spin()은 메인 스레드, 상태머신은 daemon 스레드에서 실행되므로
        call_sync()를 사용할 수 없다.
        call_async()로 요청 후 future.done()을 0.05초 간격으로 폴링하며
        spin()이 future를 완료 상태로 만들기를 기다린다.

        timeout 기본값 15초, 모션 서비스는 30초로 호출 (장거리 이동 대비).
        타임아웃 초과 시 RuntimeError → 상태머신 ERROR 진입.
        """
        future   = cli.call_async(req)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)

        if not future.done():
            future.cancel()
            raise RuntimeError(f'{name}: 타임아웃 ({timeout:.0f}s)')
        res = future.result()
        if res is None:
            raise RuntimeError(f'{name}: 응답 없음')
        if hasattr(res, 'success') and not res.success:
            raise RuntimeError(f'{name}: success=False')
        return res

    # ────────────────────────────────────────────────────────────────────
    # 그리퍼 제어
    # ────────────────────────────────────────────────────────────────────
    def _gripper_open(self):
        """gripper_node의 /gripper/open 서비스를 호출해 그리퍼를 연다."""
        self.get_logger().info('그리퍼 열기')
        self._call_service(self.cli_gripper_open, Trigger.Request(), 'gripper/open')
        time.sleep(self.gripper_wait)

    def _gripper_close(self):
        """gripper_node의 /gripper/close 서비스를 호출해 그리퍼를 닫는다."""
        self.get_logger().info('그리퍼 닫기')
        self._call_service(self.cli_gripper_close, Trigger.Request(), 'gripper/close')
        time.sleep(self.gripper_wait)

    def _rh12_move(self, stroke: int):
        """RH-P12-Rn 스트로크 직접 제어 (serial 서비스 또는 bridge 토픽).

        stroke 범위: 0 (완전 닫힘) ~ 700 (완전 열림)

        우선순위:
          1. SerialSendData 서비스가 준비된 경우 → Modbus FC16 직접 전송
          2. 서비스 없을 경우 → bridge 토픽(/gripper/rh12_stroke_cmd)으로 대체 발행

        payload는 lock 밖에서 미리 생성해 NameError 방지.
        """
        stroke = max(0, min(700, int(stroke)))
        # payload를 lock 밖에서 미리 생성 → serial 서비스/bridge 양쪽에서 재사용
        payload = self._modbus_fc16(
            self.rh12_slave_id, self.rh12_stroke_register, [stroke, 0])

        with self.rh12_lock:
            if self.cli_serial_send.service_is_ready():
                self._ensure_rh12_initialized()
                req = SerialSendData.Request()
                # dsr_msgs2 SerialSendData.data 타입이 string → latin-1로 raw bytes 전달
                req.data = payload.decode('latin-1')
                self.get_logger().info(
                    f'RH-P12-Rn serial 전송: stroke={stroke}, frame={payload.hex()}')
                try:
                    self._call_service(self.cli_serial_send, req,
                                       f'rh12_serial_send(stroke={stroke})')
                    return
                except Exception as e:
                    self.get_logger().warn(f'SerialSendData 오류: {e}')

        # serial 서비스 미준비 시 bridge 토픽으로 대체
        msg = Int32()
        msg.data = stroke
        self.pub_rh12_stroke.publish(msg)
        self.get_logger().warn(
            f'bridge 토픽으로 대체 발행: stroke={stroke}, '
            f'topic={self.rh12_bridge_topic}, frame={payload.hex()}')

    def _ensure_rh12_initialized(self):
        """RH-P12-Rn Torque Enable + Goal Current 최초 1회 초기화.

        초기화 순서 (Modbus FC06 Single Register Write):
          1. Torque Enable (reg 256, value=1)  : 모터 전원 활성화
          2. Goal Current  (reg 275, value=400): 파지력 전류 제한 설정
        rh12_init_wait(기본 0.1초): 레지스터 쓰기 후 그리퍼 내부 처리 시간 확보.
        """
        if self.rh12_initialized:
            return
        self._rh12_send_frame(
            self._modbus_fc06(self.rh12_slave_id,
                              self.rh12_torque_enable_register, 1),
            'rh12_torque_enable')
        time.sleep(self.rh12_init_wait)
        self._rh12_send_frame(
            self._modbus_fc06(self.rh12_slave_id,
                              self.rh12_goal_current_register,
                              self.rh12_goal_current),
            'rh12_goal_current')
        time.sleep(self.rh12_init_wait)
        self.rh12_initialized = True
        self.get_logger().info('RH-P12-Rn 초기화 완료')

    def _rh12_send_frame(self, payload: bytes, name: str):
        req = SerialSendData.Request()
        req.data = payload.decode('latin-1')
        self.get_logger().info(f'RH-P12-Rn frame 전송: {name}, frame={payload.hex()}')
        self._call_service(self.cli_serial_send, req, name)

    def _modbus_fc06(self, slave_id: int, register: int, value: int) -> bytes:
        """Modbus RTU FC06 (Write Single Register) 프레임 생성.

        프레임 구조 (8바이트):
          [Slave ID (1)] [0x06 (1)] [Register Addr (2, big-endian)]
          [Value (2, big-endian)] [CRC16 (2, little-endian)]

        예) slave=1, reg=256, val=1 → 01 06 01 00 00 01 [CRC_LO] [CRC_HI]
        """
        frame = bytearray([slave_id & 0xFF, 0x06])
        frame.extend(int(register).to_bytes(2, 'big'))
        frame.extend(int(value).to_bytes(2, 'big'))
        frame.extend(self._modbus_crc16(frame).to_bytes(2, 'little'))
        return bytes(frame)

    def _modbus_fc16(self, slave_id: int, start_register: int,
                     values: list[int]) -> bytes:
        """Modbus RTU FC16 (Write Multiple Registers) 프레임 생성.

        Doosan DART 예시의 modbus_fc16(282, 2, 2, [stroke, 0])와 동일 동작.

        프레임 구조:
          [Slave ID (1)] [0x10 (1)] [Start Reg (2, big-endian)]
          [Reg Count (2, big-endian)] [Byte Count (1)]
          [Value0 (2, big-endian)] ... [CRC16 (2, little-endian)]

        예) slave=1, start_reg=282, values=[700, 0] (열기):
          01 10 01 1A 00 02 04 02 BC 00 00 [CRC_LO] [CRC_HI]
        """
        rc         = len(values)
        byte_count = rc * 2
        frame = bytearray([slave_id & 0xFF, 0x10])
        frame.extend(start_register.to_bytes(2, 'big'))
        frame.extend(rc.to_bytes(2, 'big'))
        frame.append(byte_count & 0xFF)
        for v in values:
            frame.extend(int(v).to_bytes(2, 'big'))
        frame.extend(self._modbus_crc16(frame).to_bytes(2, 'little'))
        return bytes(frame)

    def _modbus_crc16(self, data: bytes) -> int:
        """Modbus RTU CRC-16/IBM 체크섬 계산.

        알고리즘: CRC-16/IBM (반사 다항식 0xA001 = 0x8005 반사)
          초기값 0xFFFF, 각 바이트 XOR 후 8회 비트 시프트.
        반환: 16비트 CRC (little-endian으로 프레임 뒤에 붙임)
        """
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        return crc & 0xFFFF


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()