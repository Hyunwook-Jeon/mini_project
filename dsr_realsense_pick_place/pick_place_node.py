"""
pick_place_node.py
------------------
Doosan E0509 Pick & Place 상태머신 노드.

흐름:
  IDLE → DETECTING → PRE_PICK → PICK → LIFT → MOVE_TO_PLACE → PLACE → HOME → IDLE

구독:
  /selected_object_pose  (geometry_msgs/PoseStamped)

Doosan 서비스 (namespace: /dsr01/):
  motion/move_joint      (dsr_msgs2/MoveJoint)
  motion/move_line       (dsr_msgs2/MoveLine)
  io/set_ctrl_box_digital_output  (dsr_msgs2/SetCtrlBoxDigitalOutput)
"""

import threading
import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, String

from dsr_msgs2.srv import MoveJoint, MoveLine, SetCtrlBoxDigitalOutput, SetToolDigitalOutput, SerialSendData


# ── 상태 정의 ───────────────────────────────────────────────────────────
class State(Enum):
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
        self.declare_parameter('robot_namespace', 'dsr01')
        self.declare_parameter('joint_vel', 30.0)
        self.declare_parameter('joint_acc', 60.0)
        self.declare_parameter('cart_vel', 100.0)
        self.declare_parameter('cart_acc', 200.0)
        self.declare_parameter('home_joints', [0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        self.declare_parameter('gripper_type', 'robotis_rh_p12_rn')
        self.declare_parameter('gripper_open_io', 1)
        self.declare_parameter('gripper_close_io', 2)
        self.declare_parameter('gripper_wait_sec', 0.8)
        self.declare_parameter('tool_digital_open_io', 1)
        self.declare_parameter('tool_digital_close_io', 2)
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
        self.gripper_type = self.get_parameter('gripper_type').value
        self.io_open = self.get_parameter('gripper_open_io').value
        self.io_close = self.get_parameter('gripper_close_io').value
        self.gripper_wait = self.get_parameter('gripper_wait_sec').value
        self.tool_io_open = self.get_parameter('tool_digital_open_io').value
        self.tool_io_close = self.get_parameter('tool_digital_close_io').value
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
        self.cli_movej = self.create_client(
            MoveJoint, f'/{ns}/motion/move_joint')
        self.cli_movel = self.create_client(
            MoveLine, f'/{ns}/motion/move_line')
        self.cli_dout = self.create_client(
            SetCtrlBoxDigitalOutput,
            f'/{ns}/io/set_ctrl_box_digital_output')
        self.cli_tool_dout = self.create_client(
            SetToolDigitalOutput,
            f'/{ns}/io/set_tool_digital_output')
        self.cli_serial_send = self.create_client(
            SerialSendData, self.rh12_serial_service)

        self._wait_for_services()

        # ── 상태 변수 ───────────────────────────────────────────────────
        # 상태머신은 별도 스레드에서 돌기 때문에 lock 으로 상태 접근을 보호한다.
        self.state = State.IDLE
        self.state_lock = threading.Lock()
        self.target_pose: PoseStamped | None = None
        self.rh12_initialized = False
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
            (self.cli_dout, 'set_ctrl_box_digital_output'),
        ]

        if self.gripper_type == 'tool_digital':
            required_services.append((self.cli_tool_dout, 'set_tool_digital_output'))
        elif self.gripper_type == 'robotis_rh_p12_rn':
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
        # 홈 자세는 작업 시작/종료 시 기준 자세 역할을 한다.
        req = MoveJoint.Request()
        req.pos = [float(v) for v in self.home_joints]
        req.vel = self.jvel
        req.acc = self.jacc
        req.time = 0.0
        req.radius = 0.0
        req.mode = 0        # MOVE_MODE_ABSOLUTE
        req.blend_type = 0
        req.sync_type = 0   # 동기 (블로킹)
        self._call_service(self.cli_movej, req, 'move_joint(home)')

    def _move_to_cart(self, x, y, z, rpy,
                      vel=None, acc=None):
        """
        Cartesian 직선 이동 (로봇 베이스 프레임 기준, m → mm 변환).
        rpy: [rx, ry, rz] in degrees.
        """
        req = MoveLine.Request()
        # Doosan move_line: mm, deg 단위
        req.pos = [x * 1000.0, y * 1000.0, z * 1000.0,
                   float(rpy[0]), float(rpy[1]), float(rpy[2])]
        req.vel = [vel if vel else self.cvel, 30.0]     # [선속도 mm/s, 각속도 deg/s]
        req.acc = [acc if acc else self.cacc, 60.0]
        req.time = 0.0
        req.radius = 0.0
        req.ref = 0         # DR_BASE
        req.mode = 0        # DR_MV_MOD_ABS
        req.blend_type = 0
        req.sync_type = 0   # 동기
        self._call_service(self.cli_movel, req, f'move_line({x:.3f},{y:.3f},{z:.3f})')

    def _call_service(self, cli, req, name: str):
        future = cli.call_async(req)
        deadline = time.monotonic() + 15.0
        # 메인 스레드에서 rclpy.spin(node)이 이미 돌고 있으므로
        # 여기서는 future 완료만 폴링하면 된다.
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)

        if not future.done():
            future.cancel()
            raise RuntimeError(f'{name}: 서비스 타임아웃')
        res = future.result()
        if res is None:
            raise RuntimeError(f'{name}: 응답 없음')
        if hasattr(res, 'success') and not res.success:
            raise RuntimeError(f'{name}: 실패 응답 (success=False)')
        return res

    # ────────────────────────────────────────────────────────────────────
    # 그리퍼 제어
    # ────────────────────────────────────────────────────────────────────
    def _gripper_open(self):
        self.get_logger().info('그리퍼 열기')
        if self.gripper_type == 'digital_io':
            self._set_digital_output(self.io_open, active=True)
            self._set_digital_output(self.io_close, active=False)
        elif self.gripper_type == 'tool_digital':
            self._set_tool_digital_output(self.tool_io_open, active=True)
            self._set_tool_digital_output(self.tool_io_close, active=False)
        elif self.gripper_type == 'robotis_rh_p12_rn':
            self._rh12_move(self.rh12_open_stroke)
        else:
            self.get_logger().warn(f'지원하지 않는 gripper_type: {self.gripper_type}')
        time.sleep(self.gripper_wait)

    def _gripper_close(self):
        self.get_logger().info('그리퍼 닫기')
        if self.gripper_type == 'digital_io':
            self._set_digital_output(self.io_open, active=False)
            self._set_digital_output(self.io_close, active=True)
        elif self.gripper_type == 'tool_digital':
            self._set_tool_digital_output(self.tool_io_open, active=False)
            self._set_tool_digital_output(self.tool_io_close, active=True)
        elif self.gripper_type == 'robotis_rh_p12_rn':
            self._rh12_move(self.rh12_close_stroke)
        else:
            self.get_logger().warn(f'지원하지 않는 gripper_type: {self.gripper_type}')
        time.sleep(self.gripper_wait)

    def _set_digital_output(self, port: int, active: bool):
        # 컨트롤 박스 디지털 출력으로 그리퍼 open/close 신호를 보낸다.
        req = SetCtrlBoxDigitalOutput.Request()
        req.index = port
        req.value = 0 if active else 1  # 0=ON, 1=OFF (active low)
        try:
            self._call_service(self.cli_dout, req,
                               f'digital_output(port={port},val={active})')
        except Exception as e:
            self.get_logger().warn(f'Digital Output 오류: {e}')

    def _set_tool_digital_output(self, port: int, active: bool):
        # 툴 플랜지 디지털 출력으로 그리퍼 open/close 신호를 보낸다.
        req = SetToolDigitalOutput.Request()
        req.index = port
        req.value = 0 if active else 1  # 0=ON, 1=OFF (active low)
        try:
            self._call_service(self.cli_tool_dout, req,
                               f'tool_digital_output(port={port},val={active})')
        except Exception as e:
            self.get_logger().warn(f'Tool Digital Output 오류: {e}')

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
        if self.rh12_initialized:
            return

        # 강사님 예시처럼 torque enable, goal current를 먼저 넣고 stroke 명령을 보낸다.
        self._rh12_send_frame(
            self._modbus_fc06(
                slave_id=self.rh12_slave_id,
                register=self.rh12_torque_enable_register,
                value=1,
            ),
            'rh12_torque_enable',
        )
        time.sleep(self.rh12_init_wait)
        self._rh12_send_frame(
            self._modbus_fc06(
                slave_id=self.rh12_slave_id,
                register=self.rh12_goal_current_register,
                value=self.rh12_goal_current,
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
        frame = bytearray()
        frame.append(slave_id & 0xFF)
        frame.append(0x06)  # FC06 = Write Single Register
        frame.extend(int(register).to_bytes(2, byteorder='big', signed=False))
        frame.extend(int(value).to_bytes(2, byteorder='big', signed=False))
        crc = self._modbus_crc16(frame)
        frame.extend(crc.to_bytes(2, byteorder='little', signed=False))
        return bytes(frame)

    def _modbus_fc16(self, slave_id: int, start_register: int, values: list[int]) -> bytes:
        # DART의 modbus_fc16(282, 2, 2, [stroke, 0]) 형태를 ROS2에서 재현한다.
        register_count = len(values)
        byte_count = register_count * 2
        frame = bytearray()
        frame.append(slave_id & 0xFF)
        frame.append(0x10)  # FC16 = Write Multiple Registers
        frame.extend(start_register.to_bytes(2, byteorder='big', signed=False))
        frame.extend(register_count.to_bytes(2, byteorder='big', signed=False))
        frame.append(byte_count & 0xFF)
        for value in values:
            frame.extend(int(value).to_bytes(2, byteorder='big', signed=False))
        crc = self._modbus_crc16(frame)
        frame.extend(crc.to_bytes(2, byteorder='little', signed=False))
        return bytes(frame)

    def _modbus_crc16(self, data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
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
