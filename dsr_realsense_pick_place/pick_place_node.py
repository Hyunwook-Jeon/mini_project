"""
pick_place_node.py
------------------
Doosan E0509 Pick & Place 상태머신 노드.

상태 전이 흐름:
  IDLE
    → DETECTING      : 홈 이동 완료 후 물체 좌표 수신 대기
    → PRE_PICK        : 작업 영역 유효 검증 후 물체 위 안전 높이까지 이동
    → PICK            : 저속으로 파지 높이까지 하강 → 그리퍼 닫기
    → LIFT            : 파지 후 안전 높이로 상승
    → MOVE_TO_PLACE   : Place 위치 상단으로 이동
    → PLACE           : 저속으로 Place 높이까지 하강 → 그리퍼 열기
    → POST_PLACE      : Place 위치 상단으로 복귀
    → HOME            : 홈 포지션으로 이동
    → IDLE (반복)
    → ERROR          : 예외 발생 시 수동 복구 대기

구독:
  /selected_object_pose  (geometry_msgs/PoseStamped)  - object_detector가 발행하는 타겟 좌표

발행:
  /pick_place_state      (std_msgs/String)             - 현재 상태 이름 (모니터링용)
  /gripper/rh12_stroke_cmd (std_msgs/Int32)            - RH-P12-Rn 서비스 미사용 시 브리지 토픽

Doosan 서비스 클라이언트 (namespace: /dsr01/):
  motion/move_joint        (dsr_msgs2/MoveJoint)
  motion/move_line         (dsr_msgs2/MoveLine)
  gripper/serial_send_data (dsr_msgs2/SerialSendData)  [RH-P12-Rn 전용]

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

    각 상태의 역할:
      IDLE          : 초기 상태. 시작 시 홈으로 이동한 뒤 DETECTING으로 전환.
      DETECTING     : /selected_object_pose 토픽에서 유효한 타겟 좌표를 기다리는 상태.
                      포즈 콜백(_cb_pose)에서 작업 영역 검증 후 PRE_PICK으로 전환.
      PRE_PICK      : 물체 위 pre_pick_z_offset(기본 0.12m) 높이까지 이동.
                      그리퍼를 미리 열고 접근해 충돌을 예방.
      PICK          : 저속(50mm/s)으로 pick_z_offset(기본 0.005m) 높이까지 하강 후 그리퍼 닫기.
      LIFT          : 파지 후 PRE_PICK 높이까지 다시 상승. 주변 장애물 회피.
      MOVE_TO_PLACE : place_position 상단(pre_place_z_offset) 으로 수평 이동.
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
        # 파라미터는 "속도", "그리퍼", "작업영역"처럼 현장에서 자주 조정하는 값들이다.
        self.declare_parameter('robot_namespace', '')
        self.declare_parameter('joint_vel', 30.0)
        self.declare_parameter('joint_acc', 60.0)
        self.declare_parameter('cart_vel', 100.0)
        self.declare_parameter('cart_acc', 200.0)
        self.declare_parameter('home_joints', [0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        self.declare_parameter('gripper_wait_sec', 0.8)
        self.declare_parameter('rh12_serial_service', '/dsr01/gripper/serial_send_data')
        self.declare_parameter('rh12_bridge_topic', '/gripper/rh12_stroke_cmd')
        self.declare_parameter('rh12_allow_missing_service', True)
        self.declare_parameter('rh12_slave_id', 1)
        self.declare_parameter('rh12_torque_enable_register', 256)
        self.declare_parameter('rh12_goal_current_register', 275)
        self.declare_parameter('rh12_stroke_register', 282)
        self.declare_parameter('rh12_goal_current', 400)
        self.declare_parameter('rh12_open_stroke', 700)
        self.declare_parameter('rh12_close_stroke', 0)
        self.declare_parameter('rh12_port', 1)
        self.declare_parameter('rh12_init_wait_sec', 0.1)
        self.declare_parameter('pre_pick_z_offset', 0.12)
        self.declare_parameter('pick_z_offset', 0.005)
        self.declare_parameter('grasp_rpy', [0.0, 180.0, 0.0])
        self.declare_parameter('place_position', [0.4, -0.3, 0.1])
        self.declare_parameter('pre_place_z_offset', 0.15)
        self.declare_parameter('place_rpy', [0.0, 180.0, 0.0])
        self.declare_parameter('workspace_x_min', 0.15)
        self.declare_parameter('workspace_x_max', 0.80)
        self.declare_parameter('workspace_y_min', -0.60)
        self.declare_parameter('workspace_y_max', 0.60)
        self.declare_parameter('workspace_z_min', 0.0)
        self.declare_parameter('workspace_z_max', 0.60)
        self.declare_parameter('target_pose_topic', '/selected_object_pose')

        ns = self.get_parameter('robot_namespace').value
        # 이후 계산과 서비스 요청에서 바로 쓰도록 멤버 변수로 저장한다.
        self.jvel = self.get_parameter('joint_vel').value
        self.jacc = self.get_parameter('joint_acc').value
        self.cvel = self.get_parameter('cart_vel').value
        self.cacc = self.get_parameter('cart_acc').value
        self.home_joints = self.get_parameter('home_joints').value
        self.gripper_wait = self.get_parameter('gripper_wait_sec').value
        self.rh12_serial_service = self.get_parameter('rh12_serial_service').value
        self.rh12_bridge_topic = self.get_parameter('rh12_bridge_topic').value
        self.rh12_allow_missing_service = self.get_parameter('rh12_allow_missing_service').value
        self.rh12_slave_id = self.get_parameter('rh12_slave_id').value
        self.rh12_torque_enable_register = self.get_parameter('rh12_torque_enable_register').value
        self.rh12_goal_current_register = self.get_parameter('rh12_goal_current_register').value
        self.rh12_stroke_register = self.get_parameter('rh12_stroke_register').value
        self.rh12_goal_current = self.get_parameter('rh12_goal_current').value
        self.rh12_open_stroke = self.get_parameter('rh12_open_stroke').value
        self.rh12_close_stroke = self.get_parameter('rh12_close_stroke').value
        self.rh12_port = self.get_parameter('rh12_port').value
        self.rh12_init_wait = self.get_parameter('rh12_init_wait_sec').value
        self.pre_pick_dz = self.get_parameter('pre_pick_z_offset').value
        self.pick_dz = self.get_parameter('pick_z_offset').value
        self.grasp_rpy = self.get_parameter('grasp_rpy').value
        self.place_pos = self.get_parameter('place_position').value
        self.pre_place_dz = self.get_parameter('pre_place_z_offset').value
        self.place_rpy = self.get_parameter('place_rpy').value

        # 작업 공간 제한
        self.ws = {
            'x': (self.get_parameter('workspace_x_min').value,
                  self.get_parameter('workspace_x_max').value),
            'y': (self.get_parameter('workspace_y_min').value,
                  self.get_parameter('workspace_y_max').value),
            'z': (self.get_parameter('workspace_z_min').value,
                  self.get_parameter('workspace_z_max').value),
        }

        # ── Doosan 서비스 클라이언트 ────────────────────────────────────
        prefix = f'/{ns}' if ns else ''
        self.cli_movej = self.create_client(
            MoveJoint, f'{prefix}/motion/move_joint')
        self.cli_movel = self.create_client(
            MoveLine, f'{prefix}/motion/move_line')
        self.cli_serial_send = self.create_client(
            SerialSendData, self.rh12_serial_service)
        
        # gripper_node 서비스 클라이언트
        self.cli_gripper_open  = self.create_client(Trigger, '/gripper/open')
        self.cli_gripper_close = self.create_client(Trigger, '/gripper/close')
 
        self._wait_for_services()

        # ── 상태 변수 ───────────────────────────────────────────────────
        # 상태머신 스레드와 ROS 콜백 스레드가 state/target_pose를 동시에 접근하므로
        # threading.Lock()으로 보호한다. Lock 없이 접근하면 데이터 경쟁(race condition) 발생.
        self.state = State.IDLE
        self.state_lock = threading.Lock()
        # target_pose: 가장 최근에 수신한 타겟 위치. DETECTING 상태에서만 갱신.
        self.target_pose: PoseStamped | None = None
        # rh12_initialized: Torque Enable, Goal Current 초기화를 최초 1회만 수행하기 위한 플래그.
        self.rh12_initialized = False
        # rh12_lock: 그리퍼 Open/Close가 거의 동시에 호출될 경우 Modbus 프레임 중복 발송 방지.
        self.rh12_lock = threading.Lock()

        # ── 발행: 현재 상태 표시 ────────────────────────────────────────
        self.pub_state = self.create_publisher(String, '/pick_place_state', 10)
        # RH-P12-Rn 서비스 브리지가 아직 없을 때를 대비해 stroke 명령을 토픽으로도 내보낸다.
        self.pub_rh12_stroke = self.create_publisher(
            Int32, self.rh12_bridge_topic, 10)

        # ── 구독: 검출된 객체 포즈 ──────────────────────────────────────
        self.create_subscription(PoseStamped, self.get_parameter('target_pose_topic').value,
                                 self._cb_pose, 10)

        # ── 상태머신 스레드 ─────────────────────────────────────────────
        self.sm_thread = threading.Thread(target=self._state_machine_loop,
                                          daemon=True)
        self.sm_thread.start()

        self.get_logger().info('PickPlaceNode 시작 — 상태: IDLE')

    # ────────────────────────────────────────────────────────────────────
    # 서비스 대기
    # ────────────────────────────────────────────────────────────────────
    def _wait_for_services(self):
        required_services = [
            (self.cli_movej, 'move_joint'),
            (self.cli_movel, 'move_line'),
        ]

        if self.rh12_allow_missing_service:
            self.get_logger().warn(
                f'{self.rh12_serial_service} 서비스가 없어도 계속 진행합니다. '
                f'대신 {self.rh12_bridge_topic} 토픽으로 stroke 명령을 발행합니다.'
            )
        else:
            required_services.append((self.cli_serial_send, self.rh12_serial_service))

        for cli, name in required_services:
            self.get_logger().info(f'서비스 대기 중: {name} ...')
            while not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f'{name} 서비스 없음, 재시도...')
        self.get_logger().info('모든 Doosan 서비스 연결 완료')

    # ────────────────────────────────────────────────────────────────────
    # 콜백: 검출 포즈 수신
    # ────────────────────────────────────────────────────────────────────
    def _cb_pose(self, msg: PoseStamped):
        with self.state_lock:
            # 검출 대기 상태일 때만 새 타겟을 받아서 다음 단계로 넘어간다.
            if self.state == State.DETECTING:
                pos = msg.pose.position
                if self._in_workspace(pos.x, pos.y, pos.z):
                    self.target_pose = msg
                    self.state = State.PRE_PICK
                    self.get_logger().info(
                        f'목표 설정: x={pos.x:.3f} y={pos.y:.3f} z={pos.z:.3f}'
                    )
                else:
                    self.get_logger().warn(
                        f'작업 공간 밖 물체 무시: x={pos.x:.3f} y={pos.y:.3f} z={pos.z:.3f}'
                    )

    def _in_workspace(self, x, y, z) -> bool:
        # 작업 가능 영역 밖 좌표는 안전을 위해 무시한다.
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

            # 현재 상태를 토픽으로 내보내면 외부에서 모니터링하기 쉽다.
            self._publish_state(current.name)

            try:
                if current == State.IDLE:
                    self.get_logger().info('홈으로 이동 후 DETECTING 대기...')
                    self._go_home()
                    self._set_state(State.DETECTING)

                elif current == State.DETECTING:
                    time.sleep(0.1)  # 포즈 콜백 대기

                elif current == State.PRE_PICK:
                    # 물체 위 안전 높이까지 먼저 접근한다.
                    pose = self.target_pose
                    self.get_logger().info('Pre-Pick 위치로 이동')
                    self._gripper_open()
                    self._move_to_cart(
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z + self.pre_pick_dz,
                        self.grasp_rpy
                    )
                    self._set_state(State.PICK)

                elif current == State.PICK:
                    # 집는 순간은 가장 충돌 위험이 크므로 저속으로 접근한다.
                    pose = self.target_pose
                    self.get_logger().info('Pick 위치로 하강')
                    self._move_to_cart(
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z + self.pick_dz,
                        self.grasp_rpy,
                        vel=50.0, acc=100.0  # 느리게 접근
                    )
                    self._gripper_close()
                    self._set_state(State.LIFT)

                elif current == State.LIFT:
                    # 다시 위로 올라와 주변과의 간섭을 줄인다.
                    pose = self.target_pose
                    self.get_logger().info('물체 들어올리기')
                    self._move_to_cart(
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z + self.pre_pick_dz,
                        self.grasp_rpy
                    )
                    self._set_state(State.MOVE_TO_PLACE)

                elif current == State.MOVE_TO_PLACE:
                    # 내려놓기 위치 상단으로 먼저 이동한 후 최종 place 한다.
                    self.get_logger().info('Place 위치로 이동')
                    px, py, pz = self.place_pos
                    self._move_to_cart(
                        px, py, pz + self.pre_place_dz,
                        self.place_rpy
                    )
                    self._set_state(State.PLACE)

                elif current == State.PLACE:
                    px, py, pz = self.place_pos
                    self.get_logger().info('물체 내려놓기')
                    self._move_to_cart(px, py, pz, self.place_rpy,
                                       vel=50.0, acc=100.0)
                    self._gripper_open()
                    self._set_state(State.POST_PLACE)

                elif current == State.POST_PLACE:
                    px, py, pz = self.place_pos
                    self._move_to_cart(px, py, pz + self.pre_place_dz,
                                       self.place_rpy)
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
        """홈 관절 각도로 이동한다.

        홈 자세(기본: [0, 0, 90, 0, 90, 0] deg)는 작업 시작/종료 시 기준 자세다.
        로봇 앞쪽 시야를 확보하고 카메라가 작업 공간을 내려다볼 수 있는 자세.

        MoveJoint 파라미터 설명:
          pos        : 목표 관절 각도 6개 (deg) - home_joints 파라미터
          vel        : 관절 속도 (deg/s) - joint_vel 파라미터
          acc        : 관절 가속도 (deg/s²) - joint_acc 파라미터
          time       : 0.0 → vel/acc 기반 자동 계산 (시간 지정이 아님)
          radius     : 0.0 → 블렌딩 없음 (완전히 정지 후 다음 모션)
          mode       : 0 = MOVE_MODE_ABSOLUTE (절대 각도 기준)
          sync_type  : 0 = 동기 (모션 완료까지 서비스 콜 블로킹)
        """
        req = MoveJoint.Request()
        req.pos = [float(v) for v in self.home_joints]
        req.vel = self.jvel
        req.acc = self.jacc
        req.time = 0.0
        req.radius = 0.0
        req.mode = 0        # MOVE_MODE_ABSOLUTE: 절대 관절 각도 기준
        req.blend_type = 0
        req.sync_type = 0   # 동기(블로킹): 이동 완료 후 서비스가 응답 반환
        self._call_service(self.cli_movej, req, 'move_joint(home)')

    def _move_to_cart(self, x, y, z, rpy,
                      vel=None, acc=None):
        """Cartesian 직선 이동 (로봇 베이스 프레임 기준).

        인자:
          x, y, z  : 목표 위치 (m 단위, 내부에서 mm로 변환)
          rpy      : [rx, ry, rz] 방향 (deg 단위)
          vel      : 선속도(mm/s). None이면 cart_vel 파라미터 사용.
          acc      : 선가속도(mm/s²). None이면 cart_acc 파라미터 사용.

        Doosan MoveLine 서비스 단위:
          - 위치: mm (ROS 관례 m와 다름 → x*1000 변환 필수)
          - 방향: deg
          - vel: [선속도 mm/s, 각속도 deg/s]
          - acc: [선가속도 mm/s², 각가속도 deg/s²]

        PICK/PLACE 하강 시에는 vel=50 mm/s로 느리게 접근해 파지 충격을 줄인다.
        """
        req = MoveLine.Request()
        # m → mm 변환: object_detector는 m 단위, Doosan 서비스는 mm 단위
        req.pos = [x * 1000.0, y * 1000.0, z * 1000.0,
                   float(rpy[0]), float(rpy[1]), float(rpy[2])]
        req.vel = [vel if vel else self.cvel, 30.0]     # [선속도 mm/s, 각속도 deg/s]
        req.acc = [acc if acc else self.cacc, 60.0]
        req.time = 0.0
        req.radius = 0.0
        req.ref = 0         # DR_BASE: 로봇 베이스 좌표계 기준
        req.mode = 0        # DR_MV_MOD_ABS: 절대 좌표 기준 이동
        req.blend_type = 0
        req.sync_type = 0   # 동기(블로킹): 이동 완료 후 반환
        self._call_service(self.cli_movel, req, f'move_line({x:.3f},{y:.3f},{z:.3f})')

    def _call_service(self, cli, req, name: str):
        """ROS 2 서비스를 비동기로 호출하고 완료까지 블로킹 폴링한다.

        이 노드는 상태머신을 daemon 스레드에서 실행하고
        메인 스레드에서 rclpy.spin()이 돌고 있는 구조다.
        따라서 call_sync()를 쓸 수 없고 call_async() + 폴링으로 구현한다.
          - call_async(): 서비스 요청을 보내고 future를 즉시 반환
          - rclpy.spin()이 콜백을 처리하면서 future를 완료 상태로 만든다
          - 이 함수는 future.done()을 0.05초 간격으로 폴링하며 완료를 기다린다

        타임아웃 15초: Doosan 서비스는 일반적으로 모션 완료까지 수 초가 걸리므로
        여유 있게 설정. 타임아웃 초과 시 RuntimeError 발생 → 상태머신이 ERROR 진입.
        """
        future = cli.call_async(req)
        deadline = time.monotonic() + 15.0
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)   # 0.05초 간격 폴링 (CPU 점유 최소화)

        if not future.done():
            future.cancel()
            raise RuntimeError(f'{name}: 서비스 타임아웃')
        res = future.result()
        if res is None:
            raise RuntimeError(f'{name}: 응답 없음')
        # Doosan 서비스는 success 필드로 성공/실패를 반환하는 경우가 있음
        if hasattr(res, 'success') and not res.success:
            raise RuntimeError(f'{name}: 실패 응답 (success=False)')
        return res

    # ────────────────────────────────────────────────────────────────────
    # 그리퍼 제어
    # ────────────────────────────────────────────────────────────────────
    def _gripper_open(self):
        self.get_logger().info('그리퍼 열기')
        self._call_service(self.cli_gripper_open, Trigger.Request(), 'gripper/open')
        time.sleep(self.gripper_wait)

    def _gripper_close(self):
        self.get_logger().info('그리퍼 닫기')
        self._call_service(self.cli_gripper_close, Trigger.Request(), 'gripper/close')
        time.sleep(self.gripper_wait)

    def _rh12_move(self, stroke: int):
        stroke = max(0, min(700, int(stroke)))
        with self.rh12_lock:
            if self.cli_serial_send.service_is_ready():
                self._ensure_rh12_initialized()

                # RH-P12-Rn 은 예시 패키지와 같은 방식으로 FC16 stroke 명령을 보낸다.
                payload = self._modbus_fc16(
                    slave_id=self.rh12_slave_id,
                    start_register=self.rh12_stroke_register,
                    values=[stroke, 0],
                )
                req = SerialSendData.Request()
                # srv 타입이 string 이라 raw bytes 를 latin-1 문자열로 그대로 실어 보낸다.
                req.data = payload.decode('latin-1')
                self.get_logger().info(
                    f'RH-P12-Rn serial 서비스 전송: stroke={stroke}, frame={payload.hex()}'
                )
                try:
                    self._call_service(self.cli_serial_send, req,
                                       f'rh12_serial_send(stroke={stroke})')
                    return
                except Exception as e:
                    self.get_logger().warn(f'RH-P12-Rn SerialSendData 오류: {e}')

            payload = self._modbus_fc16(
                slave_id=self.rh12_slave_id,
                start_register=self.rh12_stroke_register,
                values=[stroke, 0],
            )

        msg = Int32()
        msg.data = stroke
        self.pub_rh12_stroke.publish(msg)
        self.get_logger().warn(
            f'RH-P12-Rn serial 서비스가 준비되지 않아 bridge 토픽으로 대체 발행: '
            f'stroke={stroke}, topic={self.rh12_bridge_topic}, frame={payload.hex()}'
        )

    def _ensure_rh12_initialized(self):
        """RH-P12-Rn 그리퍼의 Torque Enable과 Goal Current를 최초 1회 설정한다.

        RH-P12-Rn은 전원 인가 후 Torque Enable(reg 256)과 Goal Current(reg 275)를
        먼저 설정해야 Stroke 명령에 반응한다.

        초기화 순서 (Modbus FC06, Single Register Write):
          1. Torque Enable  (register 256, value=1)  : 모터 전원 활성화
          2. Goal Current   (register 275, value=400) : 최대 전류 제한 (단위: mA 또는 raw)
             → 너무 높으면 과전류, 너무 낮으면 파지력 부족
          3. Stroke 명령은 _rh12_move()에서 FC16으로 별도 전송

        rh12_init_wait(기본 0.1초): 레지스터 쓰기 후 그리퍼 내부 처리 시간 확보.
        """
        if self.rh12_initialized:
            return

        self._rh12_send_frame(
            self._modbus_fc06(
                slave_id=self.rh12_slave_id,
                register=self.rh12_torque_enable_register,   # 기본: 256
                value=1,                                      # 1=Enable
            ),
            'rh12_torque_enable',
        )
        time.sleep(self.rh12_init_wait)   # 레지스터 적용 대기
        self._rh12_send_frame(
            self._modbus_fc06(
                slave_id=self.rh12_slave_id,
                register=self.rh12_goal_current_register,    # 기본: 275
                value=self.rh12_goal_current,                # 기본: 400
            ),
            'rh12_goal_current',
        )
        time.sleep(self.rh12_init_wait)
        self.rh12_initialized = True
        self.get_logger().info('RH-P12-Rn 초기화 완료')

    def _rh12_send_frame(self, payload: bytes, name: str):
        req = SerialSendData.Request()
        req.data = payload.decode('latin-1')
        self.get_logger().info(f'RH-P12-Rn frame 전송: {name}, frame={payload.hex()}')
        self._call_service(self.cli_serial_send, req, name)

    def _modbus_fc06(self, slave_id: int, register: int, value: int) -> bytes:
        """Modbus RTU FC06 (Write Single Register) 프레임을 생성한다.

        FC06 프레임 구조 (8바이트):
          [Slave ID (1)] [0x06 (1)] [Register Addr (2, big-endian)]
          [Value (2, big-endian)] [CRC16 (2, little-endian)]

        예) slave=1, reg=256, val=1:
          01 06 01 00 00 01 [CRC_LO] [CRC_HI]
        """
        frame = bytearray()
        frame.append(slave_id & 0xFF)
        frame.append(0x06)   # FC06 = Write Single Register
        frame.extend(int(register).to_bytes(2, byteorder='big', signed=False))
        frame.extend(int(value).to_bytes(2, byteorder='big', signed=False))
        crc = self._modbus_crc16(frame)
        # CRC는 little-endian (Low byte 먼저)으로 프레임 뒤에 붙인다
        frame.extend(crc.to_bytes(2, byteorder='little', signed=False))
        return bytes(frame)

    def _modbus_fc16(self, slave_id: int, start_register: int, values: list[int]) -> bytes:
        """Modbus RTU FC16 (Write Multiple Registers) 프레임을 생성한다.

        Doosan DART 예시의 modbus_fc16(282, 2, 2, [stroke, 0]) 와 동일한 동작.

        FC16 프레임 구조:
          [Slave ID (1)] [0x10 (1)] [Start Reg (2, big-endian)]
          [Reg Count (2, big-endian)] [Byte Count (1)]
          [Value0 (2, big-endian)] [Value1 (2, big-endian)] ... [CRC16 (2, little-endian)]

        예) slave=1, start_reg=282, values=[700, 0] (열기):
          01 10 01 1A 00 02 04 02 BC 00 00 [CRC_LO] [CRC_HI]
          (register 282에 stroke=700, register 283에 0 기록)
        """
        register_count = len(values)       # 쓸 레지스터 개수
        byte_count = register_count * 2    # 데이터 바이트 수 (레지스터 1개 = 2바이트)
        frame = bytearray()
        frame.append(slave_id & 0xFF)
        frame.append(0x10)   # FC16 = Write Multiple Registers
        frame.extend(start_register.to_bytes(2, byteorder='big', signed=False))
        frame.extend(register_count.to_bytes(2, byteorder='big', signed=False))
        frame.append(byte_count & 0xFF)
        for value in values:
            frame.extend(int(value).to_bytes(2, byteorder='big', signed=False))
        crc = self._modbus_crc16(frame)
        frame.extend(crc.to_bytes(2, byteorder='little', signed=False))
        return bytes(frame)

    def _modbus_crc16(self, data: bytes) -> int:
        """Modbus RTU CRC-16/IBM 체크섬을 계산한다.

        알고리즘: CRC-16/IBM (반사 다항식 0xA001 = 0x8005 반사)
          1. CRC 초기값 0xFFFF
          2. 각 바이트와 XOR
          3. 8번 비트 시프트:
             - LSB=1이면 오른쪽 시프트 후 다항식 0xA001과 XOR
             - LSB=0이면 오른쪽 시프트만

        반환: 16비트 CRC 값 (little-endian으로 프레임 뒤에 붙임)
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001   # 다항식 XOR (반사값)
                else:
                    crc >>= 1
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