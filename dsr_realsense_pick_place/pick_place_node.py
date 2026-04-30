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

Doosan 서비스 클라이언트 (namespace: /dsr01/):
  motion/move_joint        (dsr_msgs2/MoveJoint)
  motion/move_line         (dsr_msgs2/MoveLine)

그리퍼: /gripper/open, /gripper/close 서비스 경유
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
    MoveJoint, MoveLine,
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
    """Pick & Place 작업 단계를 나타내는 상태 열거형."""
    IDLE           = 'IDLE'
    INITIALIZING   = 'INITIALIZING' # 로봇 하드웨어 준비 중
    DETECTING      = 'DETECTING'
    PRE_PICK       = 'PRE_PICK'
    PICK           = 'PICK'
    LIFT           = 'LIFT'
    MOVE_TO_PLACE  = 'MOVE_TO_PLACE'
    PLACE          = 'PLACE'
    POST_PLACE     = 'POST_PLACE'
    HOME           = 'HOME'
    ERROR          = 'ERROR'
    EMERGENCY_STOP = 'EMERGENCY_STOP'
    BACKDRIVE      = 'BACKDRIVE'


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

        # ── 상태 변수 ───────────────────────────────────────────────────
        # state/target_pose는 상태머신 스레드와 ROS 콜백 스레드가 동시에 접근하므로 Lock 사용
        self.state       = State.IDLE
        self.state_lock  = threading.Lock()
        self.target_pose: PoseStamped | None = None
        self.pick_requested = False
        self.pending_command: str | None = None
        # 긴급정지 / 태스크 취소용 이벤트
        # _stop_event가 set되면 _call_service가 즉시 _MotionInterrupt를 발생시킨다.
        self._stop_event = threading.Event()
        self._stop_mode  = 'e_stop'  # 'e_stop' | 'cancel'
        self._missing_startup_services: set[str] = set()
        self._robot_mode_auto_ready = False
        self._robot_mode_requesting = False
        self._robot_mode_last_attempt = 0.0
        # 하드웨어 상태 캐시 (GUI 표시용)
        self._hw_state_cache: int = -1   # -1 = unknown
        self._speed_mode_cache: int = 0  # 0 = NORMAL
        # 역구동(중력보상) 제어 스레드
        self._backdrive_active  = threading.Event()
        self._backdrive_thread: threading.Thread | None = None

        # ── 퍼블리셔 / 구독 ─────────────────────────────────────────────
        self.pub_state       = self.create_publisher(String,          '/pick_place_state', 10)
        self.pub_hw_state    = self.create_publisher(Int32,           '/robot_hw_state', 10)
        self.pub_speed_mode  = self.create_publisher(Int32,           '/robot_speed_mode', 10)
        self.pub_torque_rt   = self.create_publisher(TorqueRtStream,  f'{prefix}/torque_rt_stream', 10)
        self.pub_selected    = self.create_publisher(
            String,
            self.get_parameter('selected_object_topic').value,
            10,
        )
        self.pub_heartbeat = self.create_publisher(String, '/system/heartbeat', 10)
        self.create_timer(1.0, self._publish_heartbeat)
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

        # 1초마다 하드웨어 상태 폴링 → GUI 토픽으로 발행
        self.create_timer(1.0, self._poll_hw_state)

        # ── 서비스 대기 ─────────────────────────────────────────────
        self._wait_for_services()

        # ── 상태머신 스레드 ─────────────────────────────────────────────
        # daemon=True: 메인 스레드(rclpy.spin) 종료 시 자동으로 함께 종료
        self.sm_thread = threading.Thread(target=self._state_machine_loop, daemon=True)
        self.sm_thread.start()

        self.detecting_start_time = 0.0

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

        for cli, name in required:
            self.get_logger().info(f'서비스 대기 중: {name} ...')
            max_retries = 30
            for attempt in range(max_retries):
                if cli.wait_for_service(timeout_sec=2.0):
                    break
                self.get_logger().warn(f'{name} 없음 ({attempt + 1}/{max_retries})')
            else:
                self._missing_startup_services.add(name)
                self.get_logger().error(f'{name} 연결 실패 — 서비스 없이 계속 진행')

        self.get_logger().info('기본 서비스 확인 완료. 하드웨어 동기화는 상태머신 스레드에서 진행합니다.')

    def _set_robot_mode_auto(self):
        """robot_mode=1(AUTO) 설정.
        call_async를 사용하여 타이머 콜백을 블로킹하지 않음.
        """
        if self._robot_mode_auto_ready or self._robot_mode_requesting:
            return

        now = time.monotonic()
        if now - self._robot_mode_last_attempt < 1.0:
            return
        self._robot_mode_last_attempt = now

        from dsr_msgs2.srv import SetRobotMode

        if not self.cli_set_mode.service_is_ready():
            self.get_logger().warn('set_robot_mode 서비스 대기 중...', throttle_duration_sec=2.0)
            return

        req = SetRobotMode.Request()
        req.robot_mode = 1  # DR_MODE_AUTO
        self._robot_mode_requesting = True

        def _done(f):
            self._robot_mode_requesting = False
            try:
                r = f.result()
                if r.success:
                    self._robot_mode_auto_ready = True
                    self.get_logger().info('robot_mode=1 설정 완료 ✅')
                    # 모드 설정 성공 시 서보 ON 추가 시도
                    self._auto_servo_on()
                else:
                    self.get_logger().warn('robot_mode=1 설정 거절됨')
            except Exception as e:
                self.get_logger().warn(f'robot_mode 설정 실패: {e}')

        self.cli_set_mode.call_async(req).add_done_callback(_done)

    def _auto_servo_on(self):
        """초기화 시 서보를 자동으로 켭니다."""
        if not self.cli_set_robot_ctrl.service_is_ready():
            return

        req = SetRobotControl.Request()
        req.robot_control = 3  # CONTROL_SERVO_ON

        def _servo_done(f):
            try:
                r = f.result()
                if r.success:
                    self.get_logger().info('초기 서보 ON 완료 ✅')
                else:
                    self.get_logger().info('서보가 이미 켜져 있거나 켤 수 없는 상태입니다.')
            except Exception:
                pass

        self.cli_set_robot_ctrl.call_async(req).add_done_callback(_servo_done)

    # ────────────────────────────────────────────────────────────────────
    # 콜백: 검출 포즈 수신
    # ────────────────────────────────────────────────────────────────────
    def _cb_pose(self, msg: PoseStamped):
        with self.state_lock:
            # DETECTING 상태일 때만 새 타겟을 수신해 다음 단계로 넘어간다
            if self.state == State.DETECTING and self.pick_requested:
                frame_id = msg.header.frame_id.strip()
                if not frame_id or frame_id != self.robot_base_frame:
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
        # ── 하드웨어 초기화 및 동기화 ────────────────────────────
        self.get_logger().info('🤖 로봇 하드웨어 동기화 시작...')
        self._set_state(State.INITIALIZING)

        # rclpy.spin()이 시작된 후이므로 _poll_hw_state가 정상 동작함
        start_time = time.monotonic()
        while time.monotonic() - start_time < 20.0 and rclpy.ok():
            self._set_robot_mode_auto()
            if self._hw_state_cache == 1:
                self.get_logger().info('✅ 로봇 준비 완료 (STANDBY)')
                break

            if self._hw_state_cache in (5, 6, 15): # 5:SAFE_STOP, 6:E_STOP, 15:NOT_READY
                self.get_logger().warn(f'🚨 로봇 이상 감지(상태:{self._hw_state_cache})! 자동 복구를 시도합니다...')
                self._srv_e_stop_reset(None, Trigger.Response())

            if self._hw_state_cache == 3:
                self._auto_servo_on()

            time.sleep(1.0)
            self.get_logger().info(f'하드웨어 준비 대기 중... (상태: {HW_STATE_NAMES.get(self._hw_state_cache, "UNKNOWN")})')
            self._publish_state(State.INITIALIZING.value)
        else:
            if self._hw_state_cache != 1:
                self.get_logger().error('❌ 로봇 하드웨어 준비 실패.')
                self._set_state(State.ERROR)
                # 에러 상태에서도 루프는 계속 돌려 수동 복구 대기

        if self.state == State.INITIALIZING:
            self._set_state(State.IDLE)

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
            self._publish_state(current.name if isinstance(current, State) else str(current))

            try:
                if current == State.IDLE:
                    time.sleep(0.1)

                elif current == State.DETECTING:
                    time.sleep(0.1)  # 포즈 콜백 대기 (CPU 점유 최소화)

                    if hasattr(self, 'detecting_start_time') and self.detecting_start_time > 0:
                        if time.monotonic() - self.detecting_start_time > 10.0:
                            self.get_logger().error('타겟 좌표 수신 타임아웃 (10초 초과). 카메라 연결 또는 검출 실패. IDLE 상태로 복귀합니다.')
                            self._finish_cycle()
                            continue

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
                    self.get_logger().warn(
                        '긴급정지 상태. /pick_place/e_stop_reset 서비스로 해제하세요.',
                        throttle_duration_sec=5.0,
                    )
                    time.sleep(0.2)

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
                elif mi.mode == 'backdrive':
                    self.get_logger().info('역구동 전환: 진행 중 모션 중단 완료')
                    self._set_state(State.BACKDRIVE)
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
        self._publish_state(s.value)
        self.get_logger().info(f'→ 상태 전환: {s.value}')

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
            if not self._ensure_robot_mode_auto_ready(timeout=5.0):
                raise RuntimeError('robot_mode=AUTO 준비 전입니다. 잠시 후 다시 시도하세요.')
            self._clear_target()
            self.pick_requested = True
            self._set_state(State.HOME)
            self._go_home()
            self.detecting_start_time = time.monotonic()
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

    def _ensure_robot_mode_auto_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            self._set_robot_mode_auto()
            if self._robot_mode_auto_ready:
                return True
            time.sleep(0.1)
        return self._robot_mode_auto_ready

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
        self._stop_mode = 'e_stop'
        self._stop_event.set()
        with self.state_lock:
            self.state = State.EMERGENCY_STOP
            self.pick_requested = False
            self.pending_command = None
            self.target_pose = None
        self._clear_selected_label()
        self.get_logger().error('⛔ 긴급정지 발동!')
        res.success = True
        res.message = '긴급정지 발동. /pick_place/e_stop_reset 서비스로 해제하세요.'
        return res

    def _srv_cancel(self, _, res: Trigger.Response):
        with self.state_lock:
            current = self.state
            if current in (State.IDLE, State.EMERGENCY_STOP):
                res.success = False
                res.message = '취소할 진행 중인 태스크가 없습니다.'
                return res
            self.pick_requested = False
            self.pending_command = None
            self.target_pose = None
        self._stop_mode = 'cancel'
        self._stop_event.set()
        self._clear_selected_label()
        res.success = True
        res.message = '태스크 취소 요청. 현재 모션 완료 후 그리퍼 열고 홈으로 복귀합니다.'
        return res

    def _srv_e_stop_reset(self, _, res: Trigger.Response):
        """실제 하드웨어 알람 리셋 후 상태를 IDLE로 복구합니다."""
        with self.state_lock:
            self._stop_event.clear()
            self.pending_command = None
            self.pick_requested = False
            self.target_pose = None

        if self.cli_set_robot_ctrl.service_is_ready():
            req = SetRobotControl.Request()
            req.robot_control = 1 # 1: CONTROL_RESET_ALARM

            def _reset_done(future):
                try:
                    result = future.result()
                    if result.success:
                        self.get_logger().info('하드웨어 알람 리셋 요청 성공')
                    else:
                        self.get_logger().warn('하드웨어 알람 리셋 요청 거절')
                except Exception as e:
                    self.get_logger().error(f'하드웨어 리셋 응답 오류: {e}')

            self.cli_set_robot_ctrl.call_async(req).add_done_callback(_reset_done)
        else:
            self.get_logger().warn('set_robot_control 서비스 미연결. 앱 상태만 복구합니다.')

        with self.state_lock:
            self.state = State.IDLE

        self._publish_state(State.IDLE.value)
        self.get_logger().info('✅ 알람 리셋 요청됨. IDLE 상태로 복귀.')
        res.success = True
        res.message = '하드웨어 알람 리셋 요청 및 상태 복구 완료.'
        return res

    def _srv_speed_normal(self, _, res: Trigger.Response):
        return self._set_speed_mode(0, res, '정상 속도')

    def _srv_speed_reduced(self, _, res: Trigger.Response):
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
        if not self.cli_servo_off.service_is_ready():
            res.success = False
            res.message = 'servo_off 서비스 미연결.'
            return res
        req = ServoOff.Request()
        req.stop_type = 0

        def _done(f):
            try:
                r = f.result()
                self.get_logger().warn(f'servo_off 응답: success={r.success}')
            except Exception as e:
                self.get_logger().error(f'servo_off 콜백 오류: {e}')

        self.cli_servo_off.call_async(req).add_done_callback(_done)
        with self.state_lock:
            self.state = State.EMERGENCY_STOP
            self.pick_requested = False
        res.success = True
        res.message = '서보 OFF 요청됨. servo_on으로 재기동하세요.'
        return res

    def _srv_servo_on(self, _, res: Trigger.Response):
        if not self.cli_set_robot_ctrl.service_is_ready():
            res.success = False
            res.message = 'set_robot_control 서비스 미연결.'
            return res
        req = SetRobotControl.Request()
        req.robot_control = 3

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

    def _srv_safety_normal(self, _, res: Trigger.Response):
        self._backdrive_active.clear()
        with self.state_lock:
            self._stop_event.clear()
            self.pick_requested = False
            if self.state in (State.BACKDRIVE, State.EMERGENCY_STOP):
                self.state = State.IDLE
        return self._set_robot_mode(1, res, '정상 운전 (AUTONOMOUS)')

    def _srv_safety_backdrive(self, _, res: Trigger.Response):
        if not self.cli_read_data_rt.service_is_ready():
            res.success = False
            res.message = 'realtime/read_data_rt 서비스 미연결. 역구동 불가.'
            return res

        try:
            self._hw_move_stop(stop_mode=0)
        except Exception as e:
            self.get_logger().warn(f'역구동 전환 전 move_stop 실패 (계속 진행): {e}')

        self._stop_mode = 'backdrive'
        self._stop_event.set()
        with self.state_lock:
            self.state = State.BACKDRIVE
            self.pick_requested = False
            self.pending_command = None

        self._backdrive_active.set()
        if self._backdrive_thread is None or not self._backdrive_thread.is_alive():
            self._backdrive_thread = threading.Thread(
                target=self._backdrive_loop, daemon=True)
            self._backdrive_thread.start()

        res.success = True
        res.message = '역구동 시작. 중력보상 토크 스트리밍 중. 정상운전 버튼으로 해제하세요.'
        return res

    def _backdrive_loop(self):
        self.get_logger().info('역구동 루프 시작 (중력보상 토크 스트리밍)')
        gravity = [0.0] * 6
        while self._backdrive_active.is_set() and rclpy.ok():
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
            msg = TorqueRtStream()
            msg.tor = gravity
            msg.time = 0.0
            self.pub_torque_rt.publish(msg)
            time.sleep(0.01)
        self.get_logger().info('역구동 루프 종료')

    def _set_robot_mode(self, mode: int, res: Trigger.Response, label: str):
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
        if not self.cli_move_stop.service_is_ready():
            return
        req = MoveStop.Request()
        req.stop_mode = stop_mode
        self._call_service(self.cli_move_stop, req, f'move_stop(mode={stop_mode})', timeout=5.0)

    def _poll_hw_state(self):
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

    def _go_home(self):
        req = MoveJoint.Request()
        req.pos       = [float(v) for v in self.home_joints]
        req.vel       = self.jvel
        req.acc       = self.jacc
        req.time      = 0.0
        req.radius    = 0.0
        req.mode      = 0
        req.blend_type = 0
        req.sync_type  = 0
        self._call_service(self.cli_movej, req, 'move_joint(home)', timeout=30.0)

    def _move_to_cart(self, x, y, z, rpy, vel=None, acc=None):
        req = MoveLine.Request()
        req.pos       = [x * 1000.0, y * 1000.0, z * 1000.0,
                         float(rpy[0]), float(rpy[1]), float(rpy[2])]
        req.vel       = [vel if vel else self.cvel, 30.0]
        req.acc       = [acc if acc else self.cacc, 60.0]
        req.time      = 0.0
        req.radius    = 0.0
        req.ref       = 0
        req.mode      = 0
        req.blend_type = 0
        req.sync_type  = 0
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
        if not cli.service_is_ready():
            raise RuntimeError(f'{name}: 서비스 미연결')
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

    def _call_service_with_retry(self, cli, req, name: str, timeout: float = 10.0, max_retries: int = 3):
        """서비스 호출 실패 시 일정 시간 대기 후 재시도합니다."""
        for attempt in range(1, max_retries + 1):
            try:
                return self._call_service(cli, req, name, timeout)
            except RuntimeError as e:
                # _MotionInterrupt(긴급정지/취소)는 RuntimeError 상속이 아니므로 여기서 잡히지 않고 그대로 상위로 전파됨
                if attempt == max_retries:
                    self.get_logger().error(f'{name} 최종 실패 ({max_retries}회 재시도 초과): {e}')
                    raise
                self.get_logger().warn(f'{name} 실패 ({e}). {attempt}/{max_retries} 재시도 중... (1초 대기)')
                time.sleep(1.0)

    def _gripper_open(self):
        self.get_logger().info('그리퍼 열기')
        self._call_service_with_retry(self.cli_gripper_open, Trigger.Request(), 'gripper/open', timeout=20.0)
        time.sleep(self.gripper_wait)

    def _gripper_close(self):
        """Trigger 서비스를 통해 그리퍼를 닫습니다."""
        self.get_logger().info('그리퍼 닫기')
        self._call_service_with_retry(self.cli_gripper_close, Trigger.Request(), 'gripper/close', timeout=20.0)
        time.sleep(self.gripper_wait)

    def _publish_heartbeat(self):
        msg = String()
        msg.data = 'pick_place_node'
        self.pub_heartbeat.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    from rclpy.executors import MultiThreadedExecutor
    node = PickPlaceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
