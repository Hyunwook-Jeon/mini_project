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
import math
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger

from dsr_msgs2.srv import (
    MoveJoint, MoveLine, SerialSendData,
    MoveStop,
    ServoOff,
    GetRobotState, SetRobotSpeedMode, GetRobotSpeedMode,
    SetRobotControl,
    ReadDataRt,
)
from dsr_msgs2.msg import TorqueRtStream


class _MotionInterrupt(Exception):
    """긴급정지 또는 태스크 취소 요청 시 _call_service 내부에서 발생시키는 예외."""
    def __init__(self, mode: str):  # 'e_stop' | 'cancel'
        super().__init__(mode)
        self.mode = mode


# Doosan 로봇 하드웨어 상태 코드 → 표시 문자열
HW_STATE_NAMES = {
    0:  'INITIALIZING',
    1:  'STANDBY',
    2:  'MOVING',
    3:  'SAFE_OFF',
    4:  'TEACHING',
    5:  'SAFE_STOP',
    6:  'EMERGENCY_STOP',
    7:  'HOMING',
    8:  'RECOVERY',
    9:  'SAFE_STOP2',
    10: 'SAFE_OFF2',
    15: 'NOT_READY',
}


# ── 상태 정의 ───────────────────────────────────────────────────────────
class State(Enum):
    """Pick & Place 작업 단계를 나타내는 상태 열거형.

    IDLE           : 초기 상태. 시작 시 홈으로 이동한 뒤 DETECTING으로 전환.
    DETECTING      : /selected_object_pose 토픽에서 유효한 타겟 좌표를 기다리는 상태.
    PRE_PICK       : 물체 위 pre_pick_z_offset 높이까지 이동. 그리퍼 미리 열기.
    PICK           : 저속(50mm/s)으로 pick_z_offset 높이까지 하강 후 그리퍼 닫기.
    LIFT           : 파지 후 PRE_PICK 높이까지 다시 상승.
    MOVE_TO_PLACE  : place_position 상단(pre_place_z_offset)으로 수평 이동.
    PLACE          : 저속으로 place_position까지 하강 후 그리퍼 열기.
    POST_PLACE     : 그리퍼 오픈 후 place 위 안전 높이로 복귀.
    HOME           : 홈 관절 각도로 복귀. 다음 사이클 준비.
    ERROR          : 예외 발생 시 진입. 2초 간격으로 대기하며 수동 복구 안내.
    EMERGENCY_STOP : 긴급정지 발동. e_stop_reset 서비스로만 해제 가능.
    BACKDRIVE      : 역구동(중력보상) 모드. safety_normal 서비스로만 해제 가능.
    """
    IDLE           = auto()
    DETECTING      = auto()
    PRE_PICK       = auto()
    PICK           = auto()
    LIFT           = auto()
    MOVE_TO_PLACE  = auto()
    PLACE          = auto()
    POST_PLACE     = auto()
    HOME           = auto()
    ERROR          = auto()
    EMERGENCY_STOP = auto()
    BACKDRIVE      = auto()


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
        self.declare_parameter('pre_pick_z_offset',           0.14)
        self.declare_parameter('pick_z_offset',               0.015)
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
        self.declare_parameter('robot_base_frame',            'base_link')
        self.declare_parameter('target_pose_topic',           '/selected_object_pose')
        self.declare_parameter('selected_object_topic',       '/selected_object_label')
        self.declare_parameter('use_target_pose_yaw',         True)
        self.declare_parameter('grasp_yaw_offset_deg',        0.0)

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
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.use_target_pose_yaw = self.get_parameter('use_target_pose_yaw').value
        self.grasp_yaw_offset_deg = self.get_parameter('grasp_yaw_offset_deg').value
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

        # ── 안전 모드 관련 서비스 클라이언트 ────────────────────────────
        self.cli_move_stop       = self.create_client(MoveStop,          f'{prefix}/motion/move_stop')
        self.cli_servo_off       = self.create_client(ServoOff,          f'{prefix}/system/servo_off')
        self.cli_get_robot_state = self.create_client(GetRobotState,     f'{prefix}/system/get_robot_state')
        self.cli_set_speed_mode  = self.create_client(SetRobotSpeedMode, f'{prefix}/system/set_robot_speed_mode')
        self.cli_get_speed_mode  = self.create_client(GetRobotSpeedMode, f'{prefix}/system/get_robot_speed_mode')
        self.cli_set_robot_ctrl  = self.create_client(SetRobotControl,   f'{prefix}/system/set_robot_control')
        self.cli_read_data_rt    = self.create_client(ReadDataRt,         f'{prefix}/realtime/read_data_rt')

        self._wait_for_services()

        # ── 상태 변수 ───────────────────────────────────────────────────
        # state/target_pose는 상태머신 스레드와 ROS 콜백 스레드가 동시에 접근하므로 Lock 사용
        self.state       = State.IDLE
        self.state_lock  = threading.Lock()
        self.target_pose: PoseStamped | None = None
        self.pick_requested = False
        self.pending_command: str | None = None
        self.rh12_initialized = False
        # rh12_lock: Open/Close 거의 동시 호출 시 Modbus 프레임 중복 발송 방지
        self.rh12_lock   = threading.Lock()
        # 긴급정지 / 태스크 취소용 이벤트
        # _stop_event가 set되면 _call_service가 즉시 _MotionInterrupt를 발생시킨다.
        self._stop_event = threading.Event()
        self._stop_mode  = 'e_stop'  # 'e_stop' | 'cancel'
        # 하드웨어 상태 캐시 (GUI 표시용)
        self._hw_state_cache: int = -1   # -1 = unknown
        self._speed_mode_cache: int = 0  # 0 = NORMAL
        # 역구동(중력보상) 제어 스레드
        self._backdrive_active  = threading.Event()
        self._backdrive_thread: threading.Thread | None = None

        # ── 퍼블리셔 / 구독 ─────────────────────────────────────────────
        self.pub_state       = self.create_publisher(String,          '/pick_place_state', 10)
        self.pub_rh12_stroke = self.create_publisher(Int32,           self.rh12_bridge_topic, 10)
        self.pub_hw_state    = self.create_publisher(Int32,           '/robot_hw_state', 10)
        self.pub_speed_mode  = self.create_publisher(Int32,           '/robot_speed_mode', 10)
        self.pub_torque_rt   = self.create_publisher(TorqueRtStream,  f'{prefix}/torque_rt_stream', 10)
        self.pub_selected    = self.create_publisher(
            String,
            self.get_parameter('selected_object_topic').value,
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter('target_pose_topic').value,
            self._cb_pose, 10)
        self.create_service(Trigger, '/pick_place/run_once',       self._srv_run_once)
        self.create_service(Trigger, '/pick_place/go_home',        self._srv_go_home)
        self.create_service(Trigger, '/pick_place/e_stop',         self._srv_e_stop)
        self.create_service(Trigger, '/pick_place/cancel',         self._srv_cancel)
        self.create_service(Trigger, '/pick_place/e_stop_reset',    self._srv_e_stop_reset)
        self.create_service(Trigger, '/pick_place/speed_normal',    self._srv_speed_normal)
        self.create_service(Trigger, '/pick_place/speed_reduced',   self._srv_speed_reduced)
        self.create_service(Trigger, '/pick_place/servo_off',       self._srv_servo_off)
        self.create_service(Trigger, '/pick_place/servo_on',        self._srv_servo_on)
        self.create_service(Trigger, '/pick_place/safety_normal',   self._srv_safety_normal)
        self.create_service(Trigger, '/pick_place/safety_backdrive', self._srv_safety_backdrive)

        # 500ms 마다 하드웨어 상태 폴링 → GUI 토픽으로 발행
        self.create_timer(0.5, self._poll_hw_state)

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
            if self.state == State.DETECTING and self.pick_requested:
                frame_id = msg.header.frame_id.strip()
                if frame_id and frame_id != self.robot_base_frame:
                    self.get_logger().warn(
                        f'프레임 불일치 무시: expected={self.robot_base_frame}, '
                        f'got={frame_id}'
                    )
                    return
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
            command = self._pop_pending_command()
            if command is not None:
                try:
                    self._execute_manual_command(command)
                except _MotionInterrupt as mi:
                    self._stop_event.clear()
                    if mi.mode == 'e_stop':
                        self._set_state(State.EMERGENCY_STOP)
                    else:
                        self._finish_cycle()
                except Exception as e:
                    self.get_logger().error(f'수동 명령 예외({command}): {e}')
                    self._set_state(State.ERROR)
                continue

            with self.state_lock:
                current = self.state

            # 현재 상태를 토픽으로 발행 → 외부 모니터링 용이
            self._publish_state(current.name)

            try:
                if current == State.IDLE:
                    time.sleep(0.1)

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
                        self._grasp_rpy_for_pose(pose))
                    self._set_state(State.PICK)

                elif current == State.PICK:
                    # 충돌 위험이 가장 큰 구간 → 저속(50mm/s) 접근
                    pose = self.target_pose
                    self.get_logger().info('Pick 위치로 하강')
                    self._move_to_cart(
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z + self.pick_dz,
                        self._grasp_rpy_for_pose(pose), vel=50.0, acc=100.0)
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
                        self._grasp_rpy_for_pose(pose))
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
                    self._finish_cycle()

                elif current == State.ERROR:
                    self.get_logger().error('오류 발생. 수동 복구 필요.')
                    time.sleep(2.0)

                elif current == State.EMERGENCY_STOP:
                    self.get_logger().error(
                        '긴급정지 상태. /pick_place/e_stop_reset 서비스로 해제하세요.')
                    time.sleep(1.0)

                elif current == State.BACKDRIVE:
                    time.sleep(0.5)  # 역구동 루프는 별도 스레드 — 여기서는 대기만

            except _MotionInterrupt as mi:
                self._stop_event.clear()
                if mi.mode == 'e_stop':
                    self.get_logger().error('긴급정지 발동! 하드웨어 모션 정지 중...')
                    try:
                        self._hw_move_stop(stop_mode=0)  # DR_QSTOP_STO
                    except Exception as e2:
                        self.get_logger().warn(f'하드웨어 정지 실패 (무시): {e2}')
                    self._set_state(State.EMERGENCY_STOP)
                else:
                    self.get_logger().info('태스크 취소: 그리퍼 열고 홈으로 복귀 중...')
                    try:
                        self._gripper_open()
                        self._go_home()
                    except Exception as e2:
                        self.get_logger().warn(f'취소 복귀 중 오류 (무시): {e2}')
                    self._finish_cycle()

            except Exception as e:
                self.get_logger().error(f'상태머신 예외: {e}')
                self._set_state(State.ERROR)

    def _set_state(self, s: State):
        with self.state_lock:
            self.state = s
        self.get_logger().info(f'→ 상태 전환: {s.name}')

    def _enqueue_command(self, command: str) -> bool:
        with self.state_lock:
            if self.pending_command is not None:
                return False
            self.pending_command = command
        return True

    def _pop_pending_command(self) -> str | None:
        with self.state_lock:
            command = self.pending_command
            self.pending_command = None
        return command

    def _execute_manual_command(self, command: str):
        if command == 'run_once':
            self.get_logger().info('1회 Pick & Place 요청 수신')
            self._clear_target()
            self.pick_requested = True
            self._set_state(State.HOME)
            self._go_home()
            self._set_state(State.DETECTING)
            return

        if command == 'go_home':
            self.get_logger().info('수동 홈 이동 요청 수신')
            self.pick_requested = False
            self._clear_target()
            self._clear_selected_label()
            self._set_state(State.HOME)
            self._go_home()
            self._set_state(State.IDLE)
            return

        raise RuntimeError(f'알 수 없는 명령: {command}')

    def _clear_target(self):
        with self.state_lock:
            self.target_pose = None

    def _finish_cycle(self):
        self.pick_requested = False
        self._clear_target()
        self._clear_selected_label()
        self._set_state(State.IDLE)

    def _clear_selected_label(self):
        msg = String()
        msg.data = ''
        self.pub_selected.publish(msg)

    def _srv_run_once(self, _, res: Trigger.Response):
        with self.state_lock:
            busy = self.state != State.IDLE or self.pending_command is not None
        if busy:
            res.success = False
            res.message = '현재 작업 중이어서 1회 실행을 시작할 수 없습니다.'
            return res
        if not self._enqueue_command('run_once'):
            res.success = False
            res.message = '대기 중인 명령이 있습니다.'
            return res
        res.success = True
        res.message = '1회 Pick & Place 실행을 예약했습니다.'
        return res

    def _srv_go_home(self, _, res: Trigger.Response):
        with self.state_lock:
            busy_state = self.state not in (State.IDLE, State.DETECTING, State.ERROR)
            command_pending = self.pending_command is not None
        if busy_state or command_pending:
            res.success = False
            res.message = '현재 모션 수행 중이어서 홈 이동을 예약할 수 없습니다.'
            return res
        if not self._enqueue_command('go_home'):
            res.success = False
            res.message = '대기 중인 명령이 있습니다.'
            return res
        res.success = True
        res.message = '홈 이동을 예약했습니다.'
        return res

    def _srv_e_stop(self, _, res: Trigger.Response):
        """긴급정지: 현재 모션을 즉시 중단하고 EMERGENCY_STOP 상태로 진입.

        _stop_event를 set해 _call_service 폴링 루프를 깨고
        동시에 state를 직접 EMERGENCY_STOP으로 전환하여
        상태머신이 대기(sleep) 중일 때도 즉시 인지하게 한다.
        """
        self._stop_mode = 'e_stop'
        self._stop_event.set()
        with self.state_lock:
            self.state = State.EMERGENCY_STOP
            self.pick_requested = False
            self.pending_command = None
        self.get_logger().error('⛔ 긴급정지 발동!')
        res.success = True
        res.message = '긴급정지 발동. /pick_place/e_stop_reset 서비스로 해제하세요.'
        return res

    def _srv_cancel(self, _, res: Trigger.Response):
        """태스크 취소: 현재 모션 완료 후 그리퍼를 열고 홈으로 복귀.

        IDLE 또는 EMERGENCY_STOP 상태에서는 취소할 태스크가 없으므로 거절한다.
        """
        with self.state_lock:
            current = self.state
        if current in (State.IDLE, State.EMERGENCY_STOP):
            res.success = False
            res.message = '취소할 진행 중인 태스크가 없습니다.'
            return res
        self._stop_mode = 'cancel'
        self._stop_event.set()
        res.success = True
        res.message = '태스크 취소 요청. 현재 모션 완료 후 그리퍼 열고 홈으로 복귀합니다.'
        return res

    def _srv_e_stop_reset(self, _, res: Trigger.Response):
        """긴급정지 해제: EMERGENCY_STOP 상태에서 IDLE로 복귀."""
        with self.state_lock:
            if self.state != State.EMERGENCY_STOP:
                res.success = False
                res.message = '긴급정지 상태가 아닙니다.'
                return res
            self._stop_event.clear()
            self.pick_requested = False
            self.target_pose = None
            self.state = State.IDLE
        self.get_logger().info('✅ 긴급정지 해제. IDLE 상태로 복귀.')
        res.success = True
        res.message = '긴급정지 해제 완료. IDLE 상태입니다.'
        return res

    # ────────────────────────────────────────────────────────────────────
    # 안전 모드 서비스 핸들러
    # ────────────────────────────────────────────────────────────────────
    def _srv_speed_normal(self, _, res: Trigger.Response):
        """속도 모드: 정상 속도(SPEED_NORMAL_MODE=0)로 전환."""
        return self._set_speed_mode(0, res, '정상 속도')

    def _srv_speed_reduced(self, _, res: Trigger.Response):
        """속도 모드: 감속 모드(SPEED_REDUCED_MODE=1)로 전환.

        협업 작업 시 인간이 작업 공간에 접근할 때 사용.
        로봇이 설정된 안전 속도 이하로 동작한다.
        """
        return self._set_speed_mode(1, res, '감속 모드')

    def _set_speed_mode(self, mode: int, res: Trigger.Response, label: str):
        if not self.cli_set_speed_mode.service_is_ready():
            res.success = False
            res.message = 'set_robot_speed_mode 서비스 미연결.'
            return res
        req = SetRobotSpeedMode.Request()
        req.speed_mode = mode
        future = self.cli_set_speed_mode.call_async(req)

        def _done(f):
            try:
                result = f.result()
                self._speed_mode_cache = mode if result.success else self._speed_mode_cache
                self.get_logger().info(f'속도 모드 → {label}: success={result.success}')
            except Exception as e:
                self.get_logger().warn(f'set_speed_mode 콜백 오류: {e}')

        future.add_done_callback(_done)
        res.success = True
        res.message = f'{label} 전환 요청됨.'
        return res

    def _srv_servo_off(self, _, res: Trigger.Response):
        """서보 OFF: 모든 관절 모터 전원 차단 (SAFE TORQUE OFF).

        stop_type=0 (STOP_TYPE_QUICK_STO): 빠른 정지 후 STO.
        실행 후 STATE_SAFE_OFF(3)로 전환된다.
        복구는 servo_on (/pick_place/servo_on) 서비스로 가능하다.
        """
        if not self.cli_servo_off.service_is_ready():
            res.success = False
            res.message = 'servo_off 서비스 미연결.'
            return res
        req = ServoOff.Request()
        req.stop_type = 0  # STOP_TYPE_QUICK_STO

        def _done(f):
            try:
                r = f.result()
                self.get_logger().warn(f'servo_off 응답: success={r.success}')
            except Exception as e:
                self.get_logger().error(f'servo_off 콜백 오류: {e}')

        self.cli_servo_off.call_async(req).add_done_callback(_done)
        # pick_place 상태도 EMERGENCY_STOP으로 전환 (태스크 재개 방지)
        with self.state_lock:
            self.state = State.EMERGENCY_STOP
            self.pick_requested = False
        res.success = True
        res.message = '서보 OFF 요청됨. servo_on으로 재기동하세요.'
        return res

    def _srv_servo_on(self, _, res: Trigger.Response):
        """서보 ON: 서보 OFF 후 관절 모터를 다시 켠다.

        SetRobotControl(CONTROL_SERVO_ON = CONTROL_RESET_SAFET_OFF = 3) 호출.
        로봇 하드웨어 상태가 STATE_SAFE_OFF(3) 또는 STATE_SAFE_OFF2(10)일 때만 유효하다.
        성공 시 STATE_STANDBY(1)로 복귀하고 pick_place 상태도 IDLE로 전환된다.
        """
        if not self.cli_set_robot_ctrl.service_is_ready():
            res.success = False
            res.message = 'set_robot_control 서비스 미연결.'
            return res
        req = SetRobotControl.Request()
        req.robot_control = 3  # CONTROL_SERVO_ON = CONTROL_RESET_SAFET_OFF

        def _done(f):
            try:
                r = f.result()
                if r.success:
                    self.get_logger().info('서보 ON 완료 → STANDBY')
                    with self.state_lock:
                        self._stop_event.clear()
                        self.state = State.IDLE
                else:
                    self.get_logger().warn('서보 ON 실패 (로봇 상태 확인 필요)')
            except Exception as e:
                self.get_logger().error(f'servo_on 콜백 오류: {e}')

        self.cli_set_robot_ctrl.call_async(req).add_done_callback(_done)
        res.success = True
        res.message = '서보 ON 요청됨. 응답 후 IDLE 상태로 복귀합니다.'
        return res

    # ── Doosan 로봇 모드 전환 ─────────────────────────────────────────────────
    def _srv_safety_normal(self, _, res: Trigger.Response):
        """정상 운전 모드 복귀: 역구동 중력보상 루프 정지 후 AUTONOMOUS 모드 설정."""
        # 역구동 루프 중단
        self._backdrive_active.clear()
        # 픽앤플레이스 상태도 IDLE로 초기화
        with self.state_lock:
            self._stop_event.clear()
            self.pick_requested = False
            if self.state in (State.BACKDRIVE, State.EMERGENCY_STOP):
                self.state = State.IDLE
        return self._set_robot_mode(1, res, '정상 운전 (AUTONOMOUS)')

    def _srv_safety_backdrive(self, _, res: Trigger.Response):
        """역구동 모드: 중력보상 토크를 실시간으로 스트리밍해 로봇을 자유롭게 이동시킨다.

        SetRobotMode/SetSafetyMode로는 실제 역구동이 동작하지 않는다.
        실제 구현: ReadDataRt로 gravity_torque를 읽어 TorqueRtStream 토픽으로
        지속 발행 → 컨트롤러가 중력을 보상해 외력으로 자유롭게 가이딩 가능.
        (dsr_realtime_control/GravityCompensation 예제와 동일 방식)
        """
        # 진행 중인 모션 중단
        self._stop_mode = 'e_stop'
        self._stop_event.set()
        with self.state_lock:
            self.state = State.BACKDRIVE
            self.pick_requested = False
            self.pending_command = None
        self._stop_event.clear()  # 역구동 루프 내부 polling 방해 방지

        # ReadDataRt 서비스 가용 여부 확인
        if not self.cli_read_data_rt.service_is_ready():
            res.success = False
            res.message = 'realtime/read_data_rt 서비스 미연결. 역구동 불가.'
            return res

        # 중력보상 루프 시작
        self._backdrive_active.set()
        if self._backdrive_thread is None or not self._backdrive_thread.is_alive():
            self._backdrive_thread = threading.Thread(
                target=self._backdrive_loop, daemon=True)
            self._backdrive_thread.start()

        res.success = True
        res.message = '역구동 시작. 중력보상 토크 스트리밍 중. 정상운전 버튼으로 해제하세요.'
        return res

    def _backdrive_loop(self):
        """중력보상(역구동) 루프: gravity_torque를 읽어 TorqueRtStream으로 발행.

        ReadDataRt 서비스로 6축 gravity_torque를 읽고,
        해당 토크값을 그대로 TorqueRtStream 토픽으로 발행한다.
        이렇게 하면 로봇이 중력을 스스로 보상하여 외력으로 자유롭게 이동 가능.
        100Hz(10ms)로 동작. _backdrive_active.clear() 시 종료.
        """
        self.get_logger().info('역구동 루프 시작 (중력보상 토크 스트리밍)')
        gravity = [0.0] * 6  # 마지막으로 읽은 gravity_torque 캐시

        while self._backdrive_active.is_set() and rclpy.ok():
            # ── gravity_torque 읽기 ──────────────────────────────────────
            if self.cli_read_data_rt.service_is_ready():
                future = self.cli_read_data_rt.call_async(ReadDataRt.Request())
                deadline = time.monotonic() + 0.2
                while rclpy.ok() and not future.done():
                    if not self._backdrive_active.is_set():
                        future.cancel()
                        break
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.002)

                if future.done():
                    try:
                        r = future.result()
                        if r is not None:
                            gravity = list(r.data.gravity_torque)
                    except Exception:
                        pass

            # ── 중력 토크 그대로 발행 ────────────────────────────────────
            msg = TorqueRtStream()
            msg.tor = gravity
            msg.time = 0.0
            self.pub_torque_rt.publish(msg)

            time.sleep(0.01)  # 100 Hz

        self.get_logger().info('역구동 루프 종료')

    def _set_robot_mode(self, mode: int, res: Trigger.Response, label: str):
        """SetRobotMode 서비스 공통 호출 헬퍼."""
        if not self.cli_set_mode.service_is_ready():
            res.success = False
            res.message = 'set_robot_mode 서비스 미연결.'
            return res
        from dsr_msgs2.srv import SetRobotMode
        req = SetRobotMode.Request()
        req.robot_mode = mode

        def _done(f):
            try:
                r = f.result()
                if r.success:
                    self.get_logger().info(f'로봇 모드 → {label} 전환 성공')
                else:
                    self.get_logger().warn(f'로봇 모드 → {label} 전환 거절 (현재 상태 확인 필요)')
            except Exception as e:
                self.get_logger().error(f'set_robot_mode 콜백 오류: {e}')

        self.cli_set_mode.call_async(req).add_done_callback(_done)
        res.success = True
        res.message = f'로봇 모드 → {label} 전환 요청됨.'
        return res

    def _hw_move_stop(self, stop_mode: int = 0):
        """Doosan 컨트롤러에 직접 모션 정지 명령.

        _stop_event 해제 후 호출해야 한다(이미 clear 된 상태).
        stop_mode 0 = DR_QSTOP_STO (빠른 정지 + STO)
        """
        if not self.cli_move_stop.service_is_ready():
            self.get_logger().warn('move_stop 서비스 미연결, 하드웨어 정지 건너뜀')
            return
        req = MoveStop.Request()
        req.stop_mode = stop_mode
        self._call_service(self.cli_move_stop, req, f'move_stop(mode={stop_mode})', timeout=5.0)

    def _poll_hw_state(self):
        """500ms 주기로 로봇 하드웨어 상태와 속도 모드를 폴링해 토픽으로 발행.

        서비스가 준비되지 않았거나 호출 실패 시 조용히 무시한다.
        GUI는 이 토픽을 구독해 실시간 상태를 표시한다.
        """
        if self.cli_get_robot_state.service_is_ready():
            f = self.cli_get_robot_state.call_async(GetRobotState.Request())

            def _state_cb(fut):
                try:
                    r = fut.result()
                    if r.success:
                        self._hw_state_cache = int(r.robot_state)
                        msg = Int32()
                        msg.data = self._hw_state_cache
                        self.pub_hw_state.publish(msg)
                except Exception:
                    pass

            f.add_done_callback(_state_cb)

        if self.cli_get_speed_mode.service_is_ready():
            f = self.cli_get_speed_mode.call_async(GetRobotSpeedMode.Request())

            def _speed_cb(fut):
                try:
                    r = fut.result()
                    if r.success:
                        self._speed_mode_cache = int(r.speed_mode)
                        msg = Int32()
                        msg.data = self._speed_mode_cache
                        self.pub_speed_mode.publish(msg)
                except Exception:
                    pass

            f.add_done_callback(_speed_cb)

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

    def _grasp_rpy_for_pose(self, pose: PoseStamped):
        rpy = [float(v) for v in self.grasp_rpy]
        if not self.use_target_pose_yaw:
            return rpy

        yaw_deg = self._yaw_deg_from_pose(pose)
        if yaw_deg is None:
            return rpy

        rpy[2] = self._wrap_deg(rpy[2] + yaw_deg + float(self.grasp_yaw_offset_deg))
        self.get_logger().info(
            f'그리퍼 yaw 적용: target={yaw_deg:+.1f} deg, cmd_rz={rpy[2]:+.1f} deg'
        )
        return rpy

    def _yaw_deg_from_pose(self, pose: PoseStamped) -> float | None:
        qx = float(pose.pose.orientation.x)
        qy = float(pose.pose.orientation.y)
        qz = float(pose.pose.orientation.z)
        qw = float(pose.pose.orientation.w)
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-6:
            return None
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.degrees(math.atan2(siny_cosp, cosy_cosp))

    def _wrap_deg(self, angle_deg: float) -> float:
        return ((float(angle_deg) + 180.0) % 360.0) - 180.0

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
            if self._stop_event.is_set():
                future.cancel()
                raise _MotionInterrupt(self._stop_mode)
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
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
