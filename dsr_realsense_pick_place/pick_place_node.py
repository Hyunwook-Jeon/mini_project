# 로봇 상태 머신 제어 및 이송 중 낙하 비상정지를 수행하는 제어 노드
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
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, String, Bool
from dsr_gripper_tcp_interfaces.msg import GripperState
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType, SetParametersResult
from std_srvs.srv import Trigger

from dsr_msgs2.srv import (
    MoveJoint, MoveLine,
    MoveStop,
    ServoOff,
    GetRobotState, SetRobotSpeedMode, GetRobotSpeedMode,
    SetRobotControl,
    ReadDataRt,
    GetLastAlarm,
)
from dsr_msgs2.msg import TorqueRtStream


class _MotionInterrupt(Exception):
    """긴급정지 또는 태스크 취소 요청 시 _call_service 내부에서 발생시키는 예외."""
    def __init__(self, mode: str):  # 'e_stop' | 'cancel' | 'object_lost'
        super().__init__(mode)
        self.mode = mode


class _Unreachable(Exception):
    """movel이 NOT REACHABLE(도달 불가)로 거부됐을 때 발생 — ERROR가 아닌 '물체 건너뛰기' 신호.
    잡기 전이면 그리퍼를 일절 안 건드리고 HOME으로(도달불가 후 그리퍼 close → status3 cascade 회피)."""


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
        self.declare_parameter('reach_radius_max',            0.65)   # 평면 reach 하드 차단(movel을 범위밖에 안 보냄 → status3 source 차단)
        # TCP Z 절대 하한 (base_link 기준, m). 모든 직교 이동이 이 값보다 낮게 내려가지 못하도록
        # _move_to_cart에서 강제 클램프한다. 검출 오차·잘못된 place 좌표 등 경로와 무관하게 작동.
        # 기본 0.0 = base_link 평면(현재 동작과 동일). 실제 테이블/안전 높이를 알면 그 값으로 올린다.
        self.declare_parameter('min_safe_z',                  0.0)
        self.declare_parameter('gripper_close_len',           0.145)  # RH-P12-RN close 시 flange→손끝(m). min_safe_z를 flange로 환산.
        self.declare_parameter('robot_base_frame',            'base_link')
        self.declare_parameter('target_pose_topic',           '/selected_object_pose')
        self.declare_parameter('selected_object_topic',       '/selected_object_label')
        self.declare_parameter('use_target_pose_yaw',         True)
        self.declare_parameter('grasp_yaw_offset_deg',        0.0)
        self.declare_parameter('max_grip_pos',                700)
        # grasp_min_pos: LIFT 후 파지 판정 하한. 이하면 close 명령이 안 먹어 그리퍼가
        # 거의 열린 상태(안 닫힘) → 파지 실패. (max_grip_pos 초과 = 빈손 완전닫힘)
        self.declare_parameter('grasp_min_pos',               50)
        # 낙하 감지 절대임계(mA): present_current 크기가 이보다 낮으면 물체 빠짐으로 본다.
        # "전류 0 근처 = 낙하"는 물체·모션 무관한 보편 신호라 절대임계가 견고하다. 가벼운 물체
        # 유지전류(~17mA) 아래, 빠졌을 때(~0mA) 위. 디바운스로 짧은 진동 흡수. (상대 baseline
        # 방식은 모션 전류 변동(정적31↔동적150mA)에 취약해 폐기함.)
        self.declare_parameter('object_lost_current_threshold', 10)
        # 낙하 감지 debounce: 연속 N프레임 조건 지속 시에만 낙하 판정
        self.declare_parameter('object_lost_debounce_frames',  5)
        # 물체별 파지 전류(강도) — 클래스명↔전류 1:1 매핑 + 미인식 기본값 + clamp 범위
        self.declare_parameter('grip_current_default',         300)
        self.declare_parameter('grip_class_names',             [''])
        self.declare_parameter('grip_class_currents',          [0])
        self.declare_parameter('grip_current_min',             80)
        self.declare_parameter('grip_current_max',             500)
        # 격리 토글: false면 _apply_grip_current()를 완전 우회. 이전 세션 close 동작과 동일하게 되돌림.
        # close 실패가 우리 신규 코드 때문인지 격리 검증용. true(default)로 두면 신규 코드 정상 작동.
        self.declare_parameter('enable_dynamic_grip_current',  True)

        ns = self.get_parameter('robot_namespace').value
        self.jvel         = self.get_parameter('joint_vel').value
        self.jacc         = self.get_parameter('joint_acc').value
        self.cvel         = self.get_parameter('cart_vel').value
        self.cacc         = self.get_parameter('cart_acc').value
        self.home_joints  = self.get_parameter('home_joints').value
        self.gripper_wait = self.get_parameter('gripper_wait_sec').value
        self.pre_pick_dz  = self.get_parameter('pre_pick_z_offset').value
        self.pick_dz      = self.get_parameter('pick_z_offset').value
        self.min_safe_z   = float(self.get_parameter('min_safe_z').value)
        self.gripper_close_len = float(self.get_parameter('gripper_close_len').value)
        self.grasp_rpy    = self.get_parameter('grasp_rpy').value
        self.place_pos    = self.get_parameter('place_position').value
        self.pre_place_dz = self.get_parameter('pre_place_z_offset').value
        self.place_rpy    = self.get_parameter('place_rpy').value
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.use_target_pose_yaw = self.get_parameter('use_target_pose_yaw').value
        self.grasp_yaw_offset_deg = self.get_parameter('grasp_yaw_offset_deg').value
        self.max_grip_pos = self.get_parameter('max_grip_pos').value
        self.grasp_min_pos = self.get_parameter('grasp_min_pos').value
        self.object_lost_current_threshold = self.get_parameter(
            'object_lost_current_threshold').value
        self.object_lost_debounce_frames = self.get_parameter(
            'object_lost_debounce_frames').value

        # 물체별 파지 전류 맵 구성 (names ↔ currents 1:1). 길이 불일치 시 안전하게 무시.
        self.grip_current_default = int(self.get_parameter('grip_current_default').value)
        self.grip_current_min = int(self.get_parameter('grip_current_min').value)
        self.grip_current_max = int(self.get_parameter('grip_current_max').value)
        self.grip_current_map = {}
        self._rebuild_grip_current_map(
            list(self.get_parameter('grip_class_names').value),
            list(self.get_parameter('grip_class_currents').value))
        # 파지 직전 갱신되는 현재 대상 물체 클래스 (/selected_object_class 구독)
        self._target_object_class = ''
        # 사용자가 GUI에서 클릭한 라벨 (/selected_object_label 구독) — pose race 방지 검증용
        # 예: 사용자가 "doll_2" 클릭 → 발행되는 pose의 class가 "doll"이어야 채택
        self._selected_object_label = ''
        # 격리 토글 — false면 close 직전 SetParameters 호출 안 함 (이전 동작과 동일)
        self.enable_dynamic_grip_current = bool(
            self.get_parameter('enable_dynamic_grip_current').value)
        if not self.enable_dynamic_grip_current:
            self.get_logger().warn(
                '⚠️ 격리 모드: enable_dynamic_grip_current=False → close 직전 강도 적용 우회. '
                '그리퍼는 gripper_node의 기본 close_current로 동작 (yaml/config 값).')
        self.get_logger().info(
            f'물체별 파지 전류 맵: {self.grip_current_map} (기본 {self.grip_current_default}mA, '
            f'clamp {self.grip_current_min}~{self.grip_current_max})')
        # 안내: 이 맵의 키 = "정답(known) 클래스"와 같이 다뤄야 라벨이 일관됨.
        # object_detector의 known_classes 파라미터도 동일 값으로 유지하세요.
        self.get_logger().info(
            f'  ↑ 위 키 = known(정답) 클래스 집합 → object_detector.known_classes도 '
            f'{sorted(self.grip_current_map.keys()) if self.grip_current_map else "(없음)"} '
            '와 같이 두면 라벨↔강도 일치')
        # GUI에서 grip_* 파라미터를 바꾸면 맵을 라이브로 다시 만든다.
        self.add_on_set_parameters_callback(self._on_set_parameters)

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
        # 알람 조회 — movel/movej 후 NOT REACHABLE 등 controller alarm 즉시 감지용.
        # 알람 무시하고 PICK으로 진행하던 cascade(그리퍼 status3 까지 죽는 흐름) 방지.
        self.cli_get_last_alarm = self.create_client(GetLastAlarm, f'{prefix}/system/get_last_alarm')
        # 마지막으로 본 알람의 (level, group, index) — None이면 첫 baseline (raise 안 함)
        self._last_alarm_signature = None
        self.cli_gripper_open  = self.create_client(Trigger, '/gripper/open')
        self.cli_gripper_close = self.create_client(Trigger, '/gripper/close')
        self.cli_gripper_hold_transport = self.create_client(Trigger, '/gripper/hold_transport')
        self.cli_gripper_hold_idle = self.create_client(Trigger, '/gripper/hold_idle')
        # 그리퍼 런타임 리셋(재초기화) — 에러 복구 시 그리퍼 stuck을 함께 푼다.
        self.cli_gripper_reinit = self.create_client(Trigger, '/gripper_service/reinitialize')
        # 긴급정지(EMO) 시 그리퍼 토크를 끊기 위한 클라이언트. /gripper/stop = torque OFF.
        # 그리퍼는 Doosan EMO 회로와 분리된 별도 Modbus 장치라 EMO 때 직접 꺼줘야 한다.
        self.cli_gripper_stop = self.create_client(Trigger, '/gripper/stop')
        # 물체별 파지 강도: gripper_node의 close_current 파라미터를 close 직전 동적으로 변경한다.
        # gripper_node에 close_current 런타임 변경 콜백이 이미 있어 별도 인터페이스가 필요 없다.
        self.cli_set_grip_current = self.create_client(
            SetParameters, '/rh_p12_rna_gripper/set_parameters')

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
        self._recovering = False   # recover_to_home 워커 스레드 중복 실행 가드
        # 긴급정지 / 태스크 취소용 이벤트
        # _stop_event가 set되면 _call_service가 즉시 _MotionInterrupt를 발생시킨다.
        self._stop_event = threading.Event()
        self._stop_mode  = 'e_stop'  # 'e_stop' | 'cancel' | 'object_lost'
        self._missing_startup_services: set[str] = set()
        self._robot_mode_auto_ready = False
        self._robot_mode_requesting = False
        self._robot_mode_last_attempt = 0.0
        # 하드웨어 상태 캐시 (GUI 표시용)
        self._hw_state_cache: int = -1   # -1 = unknown
        self._speed_mode_cache: int = 0  # 0 = NORMAL
        # 물리 EMO(하드웨어 E-STOP) 1회 처리 래치 — hw=6 지속 동안 토크 OFF/전이 반복 방지.
        self._hw_estop_latched = False
        self._object_lost_triggered = False
        self._object_lost_debounce_count = 0  # 낙하 조건 연속 프레임 카운터
        # status3(STATUS_IO_ERROR) false-positive 차단용 — 직전 GripperState.status 캐시.
        # 0이 아니면 그리퍼 통신 일시 장애 상태이므로 낙하 판정을 보류한다.
        self._gripper_last_status: int = 0
        self._gripper_last_pos: float = 0.0  # 최근 그리퍼 위치(present_position) — LIFT 후 파지 판정용
        # 그리퍼 준비 상태(=torque_enabled). False면 새 픽 사이클 거절 — 에러 해제 직후 reinit
        # 진행 중일 때 run_once가 들어와 TCP race가 나는 시나리오 차단.
        self._gripper_ready: bool = False
        # 역구동(중력보상) 제어 스레드
        self._backdrive_active  = threading.Event()
        self._backdrive_thread: threading.Thread | None = None

        # ── 퍼블리셔 / 구독 ─────────────────────────────────────────────
        self.pub_state       = self.create_publisher(String,          '/pick_place_state', 10)
        self.pub_error       = self.create_publisher(String,          '/pick_place_error', 10)
        self.pub_motion_active = self.create_publisher(Bool, '/gripper_service/motion_active', 10)
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
        self.create_subscription(JointState, '/gripper/state', self._cb_gripper_state, 10)
        # GripperState 직접 구독 — status 필드(STATUS_IO_ERROR=3 등)를 낙하 판정 게이트로 활용.
        self.create_subscription(GripperState, '/gripper_service/state', self._cb_gripper_status, 10)
        # 선택된 물체의 클래스명 — 파지 강도 결정에 사용 (object_detector가 좌표와 함께 발행)
        self.create_subscription(String, '/selected_object_class', self._cb_selected_class, 10)
        # 사용자가 GUI에서 클릭한 라벨 — pose race 방지 검증용 (사용자 선택과 일관된 pose만 채택)
        self.create_subscription(String, '/selected_object_label', self._cb_selected_label, 10)
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
        self.create_service(Trigger, '/pick_place/recover_to_home',  self._srv_recover_to_home)
        # ERROR 상태에서 로봇 이동 없이 알람 리셋 + 그리퍼 reinit만 하는 가벼운 복구 — recover_to_home과 별개.
        self.create_service(Trigger, '/pick_place/clear_error',      self._srv_clear_error)

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

                # ─── Race 방지: 사용자 선택과 발행된 pose의 class가 일관되는지 확인 ───
                # 시나리오: 사용자가 GUI에서 "doll_2" 클릭 직후 run_once →
                # detector가 옛 "pack_1" pose를 발행 중이면 pick_place가 그걸 채택 위험.
                # 라벨 "doll_2"의 prefix "doll"과 /selected_object_class로 받은 class를 비교.
                sel_label = self._selected_object_label
                cur_class = self._target_object_class
                if sel_label:  # 명시 선택 모드 (auto는 빈 문자열)
                    sel_prefix = sel_label.rsplit('_', 1)[0]
                    # known(doll/pack 등)이면 prefix가 class와 같아야, unknown_N이면 prefix="unknown"
                    expected_class = 'object' if sel_prefix == 'unknown' else sel_prefix
                    if cur_class and cur_class != expected_class:
                        self.get_logger().info(
                            f'race 무시: 선택={sel_label}(class={expected_class}) '
                            f'≠ 발행class={cur_class}. 새 pose 대기...'
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

    def _cb_selected_label(self, msg: String):
        # 사용자가 GUI에서 클릭한 라벨 추적. _cb_pose의 race 검증에 사용.
        self._selected_object_label = msg.data.strip()

    def _cb_selected_class(self, msg: String):
        # object_detector가 선택된 물체 좌표와 함께 발행하는 클래스명. 파지 강도 룩업에 쓴다.
        self._target_object_class = msg.data.strip()

    def _rebuild_grip_current_map(self, names, currents) -> bool:
        """클래스명↔전류 두 배열로 self.grip_current_map을 재구성한다.
        길이가 다르면 기존 맵을 유지하고 False를 반환(설정 거부 용)."""
        if len(names) != len(currents):
            self.get_logger().warn(
                f'grip_class_names({len(names)})와 grip_class_currents({len(currents)}) '
                '길이 불일치 — 맵 갱신 거부, 기존 값 유지')
            return False
        new_map = {}
        for n, c in zip(names, currents):
            n = str(n).strip()
            if n:
                new_map[n] = int(c)
        self.grip_current_map = new_map
        return True

    def _on_set_parameters(self, params):
        """GUI가 grip_* 파라미터 또는 격리 토글을 바꿀 때 라이브로 갱신한다.
        콜백 시점엔 get_parameter()가 아직 옛 값을 반환하므로 새 값은 params에서 읽는다."""
        # 동작 중 설정 변경 거부 — 안전·로봇 설정값(파지 강도, min_safe_z)은 IDLE에서만.
        # GUI도 IDLE 전용으로 막지만, ros2 param CLI 등 GUI 밖 경로까지 노드에서 차단한다.
        # close_current(그리퍼 노드 소유)는 여기 없음 — 파지 직전 동적 변경이라 게이트 대상 아님.
        GATED = {'grip_class_names', 'grip_class_currents', 'grip_current_default',
                 'grip_current_min', 'grip_current_max', 'min_safe_z'}
        if self.state != State.IDLE and any(p.name in GATED for p in params):
            blocked = sorted(p.name for p in params if p.name in GATED)
            self.get_logger().warn(
                f'설정 변경 거부({blocked}) — 현재 상태 {self.state.value}. IDLE에서만 변경 가능.')
            return SetParametersResult(
                successful=False, reason='동작 중에는 설정값 변경 불가 (IDLE 전용)')
        names = list(self.get_parameter('grip_class_names').value)
        currents = list(self.get_parameter('grip_class_currents').value)
        default = self.grip_current_default
        cmin, cmax = self.grip_current_min, self.grip_current_max
        # 격리 토글 라이브 변경 — 그리퍼 close 직전 _apply_grip_current 호출 여부
        new_toggle: bool | None = None
        # TCP Z 안전 하한 라이브 변경 — GUI에서 테이블 높이 입력 시 즉시 반영
        new_min_safe_z: float | None = None
        touched = False
        for p in params:
            if p.name == 'grip_class_names':
                names = list(p.value); touched = True
            elif p.name == 'grip_class_currents':
                currents = list(p.value); touched = True
            elif p.name == 'grip_current_default':
                default = int(p.value); touched = True
            elif p.name == 'grip_current_min':
                cmin = int(p.value); touched = True
            elif p.name == 'grip_current_max':
                cmax = int(p.value); touched = True
            elif p.name == 'enable_dynamic_grip_current':
                new_toggle = bool(p.value)
            elif p.name == 'min_safe_z':
                new_min_safe_z = float(p.value)

        if new_toggle is not None and new_toggle != self.enable_dynamic_grip_current:
            self.enable_dynamic_grip_current = new_toggle
            self.get_logger().info(
                f'🔧 격리 토글 변경: enable_dynamic_grip_current = {new_toggle} '
                f'→ 다음 close부터 {"동적 강도 적용" if new_toggle else "이전 동작(우회)"}'
            )

        if new_min_safe_z is not None:
            if new_min_safe_z < 0.0:
                return SetParametersResult(
                    successful=False, reason='min_safe_z는 0.0 이상이어야 함')
            self.min_safe_z = new_min_safe_z
            self.get_logger().info(
                f'🔧 TCP Z 안전 하한 변경: min_safe_z = {new_min_safe_z:.3f}m '
                f'→ 이후 모든 직교 이동이 이 높이로 클램프됨')

        if not touched:
            return SetParametersResult(successful=True)
        if not self._rebuild_grip_current_map(names, currents):
            return SetParametersResult(
                successful=False, reason='grip_class_names/currents 길이 불일치')
        self.grip_current_default = default
        self.grip_current_min = cmin
        self.grip_current_max = cmax
        self.get_logger().info(
            f'물체별 파지 전류 맵 갱신: {self.grip_current_map} (기본 {default}mA)')
        return SetParametersResult(successful=True)

    def _cb_gripper_status(self, msg: GripperState):
        # status는 낙하 판정 게이트, ready는 신규 픽 사이클 게이트로 사용.
        self._gripper_last_status = int(msg.status)
        self._gripper_last_pos = float(msg.present_position)
        self._gripper_ready = bool(msg.ready)

    def _cb_gripper_state(self, msg: JointState):
        if 'gripper_joint' not in msg.name:
            return

        idx = msg.name.index('gripper_joint')
        if idx >= len(msg.position):
            return

        pos = float(msg.position[idx])

        with self.state_lock:
            current_state = self.state
            already_triggered = self._object_lost_triggered

        # 들고 이동하는 상태(LIFT/MOVE_TO_PLACE)가 아니거나 이미 트리거됨 → 카운터 리셋.
        if already_triggered or current_state not in (State.LIFT, State.MOVE_TO_PLACE):
            self._object_lost_debounce_count = 0
            return

        # 통신 장애(status3 등)면 위치도 못 믿으니 낙하 판정 보류.
        if self._gripper_last_status != 0:
            return

        # 낙하 판정: 위치 기반. 물체를 쥐면 pos가 물체 두께(<max)에서 멈추고, 빠지면
        # 그리퍼가 완전닫힘(>max)으로 더 닫힌다(goal=1000 유지). 저전류 운영 시 전류 기반은
        # 정지구간 present_current가 낮아 상시 오탐이라 폐기하고 위치로 전환했다.
        object_lost_condition = (pos > self.max_grip_pos)

        if object_lost_condition:
            self._object_lost_debounce_count += 1
            if self._object_lost_debounce_count >= self.object_lost_debounce_frames:
                self.get_logger().error(
                    f'물체 탈조 낙하 감지 ({self._object_lost_debounce_count}프레임 지속): '
                    f'위치={pos:.1f} > max_grip_pos {self.max_grip_pos}')
                self._object_lost_debounce_count = 0
                self._trigger_object_lost_stop()
        else:
            self._object_lost_debounce_count = 0

    def _trigger_object_lost_stop(self):
        with self.state_lock:
            if self._object_lost_triggered:
                return
            # 상태 가드 — LIFT/MOVE_TO_PLACE에서만 _stop_event를 set한다.
            # _cb_gripper_state 진입 시점에 이미 상태 체크하지만, 그 사이 상태 전이가 일어났을 수 있음.
            # 다른 상태에서 set하면 인터럽트 잡힐 _call_service가 없어 event가 잔존 → 다음 사용자
            # 명령(run_once/go_home 등) 진입 시 stale _MotionInterrupt 발동 → 명령이 silently no-op.
            if self.state not in (State.LIFT, State.MOVE_TO_PLACE):
                self.get_logger().warn(
                    f'낙하 트리거 무시 — 현재 상태({getattr(self.state, "value", self.state)}) '
                    f'에서는 stop_event를 set하지 않음 (오염 방지).'
                )
                return
            self._object_lost_triggered = True
            self._stop_mode = 'object_lost'
            self.pick_requested = False
            self.pending_command = None
            self.target_pose = None

        if self.cli_move_stop.service_is_ready():
            req = MoveStop.Request()
            req.stop_mode = 1  # QUICK_STOP
            self._set_motion_active(True)  # move_stop도 컨트롤러 점유 → 폴링 조율(타임아웃 가드가 재개)
            self.cli_move_stop.call_async(req)
        else:
            self.get_logger().warn('move_stop 서비스 미연결. 인터럽트로만 모션을 중단합니다.')

        self._clear_selected_label()
        self._stop_event.set()
        self.get_logger().warn('낙하 감지: 모션 중단 후 태스크를 취소하고 홈으로 복귀합니다.')

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
                    elif mi.mode == 'object_lost':
                        self.get_logger().warn('낙하 감지: 태스크를 취소하고 홈으로 복귀합니다.')
                        self._finish_cycle()
                    else:
                        self._finish_cycle()
                except Exception as e:
                    self.get_logger().error(f'수동 명령 예외({command}): {e}')
                    self._publish_error(str(e))
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
                    # 파지 확정 판정 — 들어올린 후(중력 테스트)에 위치로 판단.
                    # close 순간의 지터·통신노이즈를 피해 안정된 시점에 판정한다.
                    grasp_pos = self._gripper_last_pos
                    if grasp_pos <= self.grasp_min_pos:
                        self.get_logger().error(
                            f'파지 실패 — 그리퍼 안 닫힘 (pos={grasp_pos:.0f} ≤ {self.grasp_min_pos}). HOME 복귀.')
                        self._set_state(State.HOME)
                    elif grasp_pos > self.max_grip_pos:
                        self.get_logger().error(
                            f'파지 실패 — 빈손 완전닫힘 (pos={grasp_pos:.0f} > {self.max_grip_pos}). HOME 복귀.')
                        self._set_state(State.HOME)
                    else:
                        self.get_logger().info(f'파지 확정 (pos={grasp_pos:.0f}).')
                        # 파지 확정 → 이송 전류로 낮춰 들고 이동 (발열·과압착 완화, self-locking이 유지)
                        self._call_service(self.cli_gripper_hold_transport, Trigger.Request(),
                                           'gripper/hold_transport', timeout=5.0)
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
                    self._gripper_torque_off()
                    try:
                        self._hw_move_stop(stop_mode=0)  # DR_QSTOP_STO
                    except Exception as e2:
                        self.get_logger().warn(f'하드웨어 정지 실패 (무시): {e2}')
                    self._set_state(State.EMERGENCY_STOP)
                elif mi.mode == 'object_lost':
                    self.get_logger().warn('낙하 감지: 태스크를 취소하고 홈으로 복귀합니다.')
                    self._safe_recover_to_home('낙하 복구')
                    self._finish_cycle()
                elif mi.mode == 'backdrive':
                    self.get_logger().info('역구동 전환: 진행 중 모션 중단 완료')
                    self._set_state(State.BACKDRIVE)
                else:
                    # Cancel = 그 자리에 우아하게 정지만. 그리퍼·HOME은 손대지 않는다.
                    # 그리퍼가 죽어 있어도 cancel은 항상 성공해야 하고, 잡은 물체는
                    # 떨어뜨리지 않고 유지(안전). HOME 복귀가 필요하면 사용자가 별도로 누름.
                    self.get_logger().info('태스크 취소: 모션 정지, 현 위치 유지 (HOME/그리퍼는 사용자 명령으로)')
                    self._finish_cycle()

            except _Unreachable as ue:
                self.get_logger().warn(f'🔴 도달 불가 — 물체 건너뜀: {ue}')
                self._publish_error(str(ue))
                with self.state_lock:
                    st = self.state
                if st in (State.LIFT, State.MOVE_TO_PLACE, State.PLACE):
                    # 이미 파지한 뒤 → 허공 낙하 방지 위해 그리퍼 안 건드리고 ERROR(수동 복구)
                    self.get_logger().warn('이미 파지 상태 — 안전하게 ERROR로 정지(물체 든 채).')
                    self._set_state(State.ERROR)
                else:
                    # 잡기 전(접근 중) → 그리퍼 일절 안 건드리고 HOME로 (도달불가 후 close → status3 cascade 회피)
                    self._set_state(State.HOME)

            except Exception as e:
                self.get_logger().error(f'상태머신 예외: {e}')
                self._publish_error(str(e))
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
            self._object_lost_triggered = False
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
            self._object_lost_triggered = False
            self._set_state(State.IDLE)
            return

        raise RuntimeError(f'알 수 없는 명령: {command}')

    def _ensure_robot_mode_auto_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            self._set_robot_mode_auto()
            if self._robot_mode_auto_ready:
                return True
            # 폴백: set_robot_mode 응답이 유실돼도(rmw "failed to send response" timeout) 로봇이
            # STANDBY(1)면 모드는 실제로 AUTO로 설정된 것(명령은 전달됨, 응답만 유실). 이 경우
            # 무한 차단을 막기 위해 ready로 본다. 안 그러면 멀쩡한 로봇에서 run_once가 ERROR.
            if self._hw_state_cache == 1:
                self._robot_mode_auto_ready = True
                self.get_logger().warn(
                    'robot_mode 응답 미수신이나 STANDBY 확인 → AUTO ready 처리.')
                return True
            time.sleep(0.1)
        return self._robot_mode_auto_ready

    def _clear_target(self):
        with self.state_lock:
            self.target_pose = None

    def _finish_cycle(self):
        self.pick_requested = False
        self._object_lost_triggered = False
        self._clear_target()
        self._clear_selected_label()
        self._set_state(State.IDLE)
        # idle 위치락 — 작업 완료 후 전류 튐/채터링 완화 (best-effort, 실패해도 무방)
        if self.cli_gripper_hold_idle.service_is_ready():
            try:
                self.cli_gripper_hold_idle.call_async(Trigger.Request())
            except Exception:
                pass

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
        # Fix C — 그리퍼 미준비 시 거절. 에러 해제 직후 reinit 중 PRE_PICK gripper/open이
        # 들어가 TCP race가 발생하던 시나리오(로그 45) 차단.
        if not self._gripper_ready:
            res.success = False
            res.message = '그리퍼 준비 미완료(reinit 중일 수 있음). 잠시 후 다시 시도하세요.'
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
        # 긴급정지 핵심 안전 동작 — 그리퍼 토크 OFF (잡고 있던 물체 안전하게 해제).
        self._gripper_torque_off()
        self.get_logger().error('⛔ 긴급정지 발동!')
        res.success = True
        res.message = '긴급정지 발동. /pick_place/e_stop_reset 서비스로 해제하세요.'
        return res

    def _srv_cancel(self, _, res: Trigger.Response):
        # cancel 허용 상태 — 실제 픽 사이클 + 단독 HOME 이동만. INITIALIZING/ERROR/BACKDRIVE는
        # 의미 없거나 다른 버튼(에러 해제, e-stop reset)이 담당하므로 제외.
        cancelable = (
            State.DETECTING, State.PRE_PICK, State.PICK, State.LIFT,
            State.MOVE_TO_PLACE, State.PLACE, State.POST_PLACE, State.HOME,
        )
        with self.state_lock:
            current = self.state
            if current not in cancelable:
                res.success = False
                res.message = (
                    f'현재 상태({current.value if hasattr(current, "value") else current})에서는 cancel 불가. '
                    f'픽 사이클 또는 HOME 이동 중에만 사용하세요.'
                )
                return res
            self.pick_requested = False
            self.pending_command = None
            self.target_pose = None
        self._stop_mode = 'cancel'
        # 인터럽트만으로는 컨트롤러에서 실행 중인 모션이 멈추지 않는다.
        # Soft Stop(감속 램프, 서보 유지) — 충격 최소화. QUICK_STOP(1)은 "콱" 멈춰서
        # 컨트롤러/그리퍼 통신에 충격을 주므로 cancel용으로 부적합. STO(0)는 서보 OFF라 위험.
        if self.cli_move_stop.service_is_ready():
            req = MoveStop.Request()
            req.stop_mode = 2  # DR_SSTOP (Soft Stop — 부드러운 감속, 서보 유지)
            self._set_motion_active(True)  # move_stop도 컨트롤러 점유 → 폴링 조율(타임아웃 가드가 재개)
            self.cli_move_stop.call_async(req)
        else:
            self.get_logger().warn('move_stop 서비스 미연결. 인터럽트로만 모션을 중단합니다.')
        self._stop_event.set()
        self._clear_selected_label()
        res.success = True
        res.message = '태스크 취소 요청. 부드럽게 감속 정지 후 현 위치 유지 (HOME/그리퍼는 별도 명령).'
        return res

    def _srv_e_stop_reset(self, _, res: Trigger.Response):
        """실제 하드웨어 알람 리셋 후 상태를 IDLE로 복구합니다."""
        with self.state_lock:
            self._stop_event.clear()
            self.pending_command = None
            self.pick_requested = False
            self.target_pose = None
            self._object_lost_triggered = False

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

    def _srv_clear_error(self, _, res: Trigger.Response):
        """에러 상태 해제 — 알람 리셋 + 그리퍼 reinit 후 IDLE 복귀. 로봇은 움직이지 않는다.
        recover_to_home과 달리 사용자가 상황 확인 후 수동으로 다음 액션을 결정하도록 함."""
        with self.state_lock:
            if self.state != State.ERROR:
                res.success = False
                res.message = 'ERROR 상태에서만 사용 가능합니다.'
                return res
            # cancel 등에서 잔존했을 수 있는 stop_event 정리.
            self._stop_event.clear()
            self.pending_command = None
            self.pick_requested = False
            self.target_pose = None
            self._object_lost_triggered = False
            # reinit 동안 /gripper_service/state가 옛 ready=True 캐시를 그대로 재발행하는 경우 대비,
            # 명시적으로 False로 떨어뜨려 Fix C(run_once 게이트)가 reinit 완료까지 reject 보장.
            self._gripper_ready = False

        # 1. 컨트롤러 알람 리셋 (async). 응답 결과는 로그로만 확인.
        if self.cli_set_robot_ctrl.service_is_ready():
            req = SetRobotControl.Request()
            req.robot_control = 1  # CONTROL_RESET_ALARM
            def _alarm_done(future):
                try:
                    r = future.result()
                    self.get_logger().info(f'에러 해제: 알람 리셋 응답 success={r.success}')
                except Exception as e:
                    self.get_logger().warn(f'에러 해제: 알람 리셋 응답 오류: {e}')
            self.cli_set_robot_ctrl.call_async(req).add_done_callback(_alarm_done)
        else:
            self.get_logger().warn('에러 해제: set_robot_control 서비스 미연결 — 알람 리셋 생략')

        # 2. 그리퍼 reinit (async). status3 latch가 있다면 풀어주는 용도.
        if self.cli_gripper_reinit.service_is_ready():
            def _reinit_done(future):
                try:
                    r = future.result()
                    self.get_logger().info(f'에러 해제: 그리퍼 reinit 응답 success={r.success}')
                except Exception as e:
                    self.get_logger().warn(f'에러 해제: 그리퍼 reinit 응답 오류: {e}')
            self.cli_gripper_reinit.call_async(Trigger.Request()).add_done_callback(_reinit_done)
        else:
            self.get_logger().warn('에러 해제: 그리퍼 reinit 서비스 미연결 — 생략')

        with self.state_lock:
            self.state = State.IDLE

        self._publish_state(State.IDLE.value)
        self.get_logger().info('에러 해제 요청 — IDLE 복귀. 로봇은 정지 상태 유지.')
        res.success = True
        res.message = (
            '에러 해제: 알람 리셋 + 그리퍼 reinit 요청 완료. '
            '그리퍼 ready 확인 후 수동으로 다음 명령을 내려주세요.'
        )
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

    def _srv_recover_to_home(self, _, res: Trigger.Response):
        with self.state_lock:
            if self.state != State.ERROR:
                res.success = False
                res.message = "로봇이 에러 상태가 아닙니다."
                return res
            if self._recovering:
                res.success = False
                res.message = "이미 복구 중입니다."
                return res
            self._recovering = True
        # 복구 시퀀스(sleep + 그리퍼 reinit 최대 90s + 홈 이동)를 서비스 콜백에서 직접
        # 돌리면 executor 콜백 스레드를 ~2분 점유하고 E-STOP로도 못 끊는다.
        # → 별도 스레드로 실행하고 서비스는 즉시 응답. GUI는 상태(IDLE 복귀)로 완료를 추적.
        threading.Thread(target=self._recover_to_home_worker, daemon=True).start()
        res.success = True
        res.message = "에러 복구 및 홈 복귀를 시작했습니다."
        return res

    def _recover_to_home_worker(self):
        try:
            self.get_logger().info("에러 복구 및 안전 복귀 시퀀스를 시작합니다.")

            # 1. 컨트롤러 에러 해제 및 상태 IDLE 복구
            self._srv_e_stop_reset(None, Trigger.Response())
            time.sleep(1.0)  # 알람 리셋 비동기 완료 대기

            # 2. 서보 ON 제어권 복구 (실패해도 홈 이동은 시도)
            try:
                ctrl_req = SetRobotControl.Request()
                ctrl_req.robot_control = 3  # CONTROL_SERVO_ON
                self._call_service(self.cli_set_robot_ctrl, ctrl_req, "set_robot_control",
                                   timeout=5.0)
            except Exception as e:
                self.get_logger().warn(f'서보 ON 요청 실패 (계속 진행): {e}')

            # 2.5 그리퍼 재초기화 — 그리퍼가 에러/무응답(status 3)으로 stuck일 수 있으므로
            #     열기 전에 먼저 복구한다(시리얼 recycle + 토크 재인가). 로봇 재부팅 불필요.
            reinit_attempted = False
            try:
                if self.cli_gripper_reinit.service_is_ready():
                    self.get_logger().info('그리퍼 재초기화 시도...')
                    # reinit 직전 stale True를 떨어뜨려 아래 ready 대기 루프가 신호만 인식하게 함.
                    self._gripper_ready = False
                    reinit_attempted = True
                    self._call_service(self.cli_gripper_reinit, Trigger.Request(),
                                       "gripper_reinit", timeout=90.0)
                else:
                    self.get_logger().warn('그리퍼 reinit 서비스 미연결 (건너뜀)')
            except Exception as e:
                self.get_logger().warn(f'그리퍼 재초기화 실패 (계속 진행): {e}')

            # reinit 응답 직후 /gripper_service/state가 ready=True로 갱신되기까지 한 박자 필요.
            # 이 대기 누락 시 다음 _gripper_open이 in-flight reinit과 충돌해 TCP race 발생 (로그 45/47).
            if reinit_attempted:
                wait_deadline = time.monotonic() + 10.0
                while not self._gripper_ready and time.monotonic() < wait_deadline:
                    time.sleep(0.1)
                if not self._gripper_ready:
                    self.get_logger().warn(
                        '그리퍼 reinit 후 ready=True 신호 미수신 (10s 초과). 그리퍼 열기는 건너뛰고 홈 복귀만 시도.'
                    )

            # 3. 그리퍼 완전 Open — ready 신호 받은 경우만 시도. 미수신이면 건너뛰고 홈 복귀.
            if self._gripper_ready:
                try:
                    self._gripper_open()
                except Exception as e:
                    self.get_logger().warn(f'그리퍼 열기 실패 (계속 진행): {e}')
            else:
                self.get_logger().warn('그리퍼 ready 미확인 — 열기 건너뜀, 홈 복귀만 진행.')

            # 4. 홈 위치로 안전 복귀 이동
            self._go_home()

            # 5. 상태 초기화
            with self.state_lock:
                self._object_lost_triggered = False
                self._object_lost_debounce_count = 0
                self.state = State.IDLE
            self.get_logger().info("에러 복구 및 홈으로 복귀 성공 완료")
        except Exception as e:
            self.get_logger().error(f'recover_to_home 실패: {e}')
            with self.state_lock:
                self.state = State.ERROR
        finally:
            self._recovering = False

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
        # move_stop도 컨트롤러를 점유 → 진행 중 그리퍼 통신과 경합(소켓 오염). 폴링 조율로 감싼다.
        self._set_motion_active(True)
        try:
            self._call_service(self.cli_move_stop, req, f'move_stop(mode={stop_mode})', timeout=5.0)
        finally:
            self._set_motion_active(False)

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
                        self._handle_hw_estop(self._hw_state_cache)
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

    def _handle_hw_estop(self, hw: int):
        """물리 EMO(하드웨어 E-STOP, hw=6) 감지 → 그리퍼 토크 OFF + EMERGENCY_STOP (1회).
        Doosan EMO는 로봇 서보만 끊고 그리퍼(별도 Modbus)는 안 건드리므로 여기서 직접 꺼준다.
        INITIALIZING은 기존 자동복구 로직(상태머신 진입부)이 담당하므로 제외, 래치로 중복 방지."""
        if hw != 6:
            self._hw_estop_latched = False
            return
        if self._hw_estop_latched or self.state in (State.INITIALIZING, State.EMERGENCY_STOP):
            return
        self._hw_estop_latched = True
        self.get_logger().error('⛔ 하드웨어 E-STOP(물리 EMO) 감지! 그리퍼 토크 OFF + 긴급정지.')
        self._gripper_torque_off()
        self._stop_mode = 'e_stop'
        self._stop_event.set()
        with self.state_lock:
            self.state = State.EMERGENCY_STOP
            self.pick_requested = False
            self.pending_command = None
            self.target_pose = None

    def _publish_state(self, name: str):
        msg = String()
        msg.data = name
        self.pub_state.publish(msg)

    def _classify_error(self, text: str) -> str:
        """에러 메시지를 카테고리로 분류 — GUI status 바에 '어떤 에러인지' 표기용."""
        t = (text or '').lower()
        if 'reachable' in t or '1206' in t:
            return '🔴 도달 불가 — 작업영역/IK 밖 (물체 위치 조정)'
        if 'singular' in t or '3205' in t or '3206' in t:
            return '🟠 특이점 영역 — 경로/자세 회피 필요'
        if 'status 3' in t or 'io_error' in t or 'io error' in t or 'status3' in t:
            return '🔴 그리퍼 통신 끊김 (status3) — 재시작 필요'
        if 'gripper/close' in t or '파지' in t or 'grasp' in t:
            return '🟠 파지 실패 — 빈손/안닫힘'
        if 'robot_mode' in t or 'auto 준비' in t or 'standby' in t:
            return '⏳ 로봇 모드 준비 전 — 잠시 후 재시도'
        if 'collision' in t or '충돌' in t:
            return '🔴 충돌 감지 — 정지'
        return f'⚠ 에러: {text[:60]}'

    def _publish_error(self, text: str):
        """ERROR 진입 시 분류된 에러 사유를 발행 (GUI status 바 표기)."""
        msg = String()
        msg.data = self._classify_error(text)
        self.pub_error.publish(msg)

    def _set_motion_active(self, active: bool):
        """모션(movel/movej/move_stop) 진행을 그리퍼 폴링에 알림 → 모션 중 폴링 skip(컨트롤러 경합·소켓 오염 회피)."""
        msg = Bool()
        msg.data = active
        self.pub_motion_active.publish(msg)

    def _check_motion_alarm(self, move_name: str) -> None:
        """movel/movej 호출 후 controller alarm 발생 여부 즉시 점검.
        새 알람이고 level >= 2(Error) 이면 RuntimeError 발생 → 상태머신이 ERROR로 전환.
        cascade(NOT REACHABLE 후 그리퍼 close 시도 → status3 → controller 깊은 fault) 차단."""
        if not self.cli_get_last_alarm.service_is_ready():
            return
        try:
            res = self._call_service(
                self.cli_get_last_alarm, GetLastAlarm.Request(),
                'get_last_alarm', timeout=2.0)
        except _MotionInterrupt:
            raise  # cancel/e-stop은 상위에서 처리
        except Exception as e:
            self.get_logger().debug(f'알람 조회 실패(무시): {e}')
            return
        a = getattr(res, 'log_alarm', None)
        if a is None:
            return
        sig = (int(a.level), int(a.group), int(a.index))
        prev = self._last_alarm_signature
        self._last_alarm_signature = sig
        if prev is None:
            # 첫 호출 — 기존 알람을 baseline으로만 잡고 raise 안 함
            return
        if sig == (0, 0, 0) or sig == prev:
            return  # 알람 없음 또는 이전과 동일(잔존)
        # 새 알람 발생
        params = ''
        try:
            params = ' / '.join(str(p) for p in a.param if p)
        except Exception:
            pass
        self.get_logger().error(
            f'🚨 모션 알람 감지 ({move_name}): level={a.level}, group={a.group}, '
            f'index={a.index} | {params}'
        )
        # level 2+(Error)면 상태머신으로 신호.
        if int(a.level) >= 2:
            # NOT REACHABLE(도달 불가, index 1206)은 ERROR가 아닌 '물체 건너뛰기'로 분기.
            # (잡기 전이면 그리퍼 안 건드리고 HOME → 도달불가 후 close 시도 status3 cascade 회피)
            if int(a.index) == 1206 or 'reachable' in params.lower():
                raise _Unreachable(f'{move_name} 도달 불가 (alarm {a.index})')
            raise RuntimeError(
                f'controller alarm {a.index} (level {a.level}) — {move_name} 실패'
            )

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
        self._set_motion_active(True)
        try:
            self._call_service(self.cli_movej, req, 'move_joint(home)', timeout=30.0)
            self._check_motion_alarm('movej(home)')
        finally:
            self._set_motion_active(False)

    def _move_to_cart(self, x, y, z, rpy, vel=None, acc=None):
        # ★ 범위 밖 좌표는 movel을 아예 보내지 않는다(source 차단). movel NOT REACHABLE 알람이
        # 컨트롤러를 교란해 그리퍼 RS-485 폴링을 죽이므로(status3), 평면 reach 밖이면 여기서 막는다.
        _r = math.hypot(x, y)
        _reach_max = self.get_parameter('reach_radius_max').value
        if _r > _reach_max:
            raise _Unreachable(
                f'범위 밖 차단 — 평면반경 {_r:.3f}m > reach {_reach_max:.3f}m (movel 미전송)')
        # TCP Z 절대 하한 강제. 모든 직교 이동이 이 함수를 거치므로 여기 한 곳에서 막으면
        # 검출 오차·잘못된 목표 좌표 등 어떤 경로로 들어온 값이든 하한 아래로는 못 내려간다.
        # min_safe_z = 그리퍼 close 손끝이 닿으면 안 되는 최저 Z(테이블/지그). calibration과 무관.
        # movel z는 flange 좌표(TCP 미설정)라, close 길이만큼 빼야 실제 손끝 높이가 된다.
        fingertip_z = z - self.gripper_close_len
        if fingertip_z < self.min_safe_z:
            new_z = self.min_safe_z + self.gripper_close_len
            self.get_logger().warn(
                f'Z 하한 클램프: 손끝 z={fingertip_z:.3f}m < min_safe_z={self.min_safe_z:.3f}m '
                f'(close 길이 {self.gripper_close_len:.3f}m). flange z {z:.3f}→{new_z:.3f}m로 올림.')
            z = new_z
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
        self._set_motion_active(True)
        try:
            self._call_service(self.cli_movel, req,
                               f'move_line({x:.3f},{y:.3f},{z:.3f})', timeout=30.0)
            self._check_motion_alarm(f'movel({x:.3f},{y:.3f},{z:.3f})')
        finally:
            self._set_motion_active(False)

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

    def _safe_recover_to_home(self, reason: str):
        """취소/낙하 복귀 전용 헬퍼 — 그리퍼 열기와 홈 복귀를 **독립적으로** 시도한다.
        - 한 단계 실패해도 다음 단계는 무조건 시도(그래서 go_home 호출이 빠지지 않음).
        - 복귀 도중 들어온 추가 _MotionInterrupt(중첩 cancel 등)도 무시 — 한 번 복귀가 시작됐으면 끝까지 간다.
        log 24번 분석: 그리퍼 status3 latch로 _gripper_open이 success=False → 재시도 중 추가 cancel → go_home 미호출로
        로봇이 그 자리에 멈추던 버그를 차단."""
        # 다음 단계 진입 전에 추가로 set된 _stop_event를 비운다(중첩 cancel 무시).
        self._stop_event.clear()
        try:
            self._gripper_open()
        except _MotionInterrupt:
            self.get_logger().warn(f'{reason}: 그리퍼 열기 도중 추가 cancel — 무시하고 다음 단계로')
        except Exception as e:
            self.get_logger().warn(f'{reason}: 그리퍼 열기 실패 (무시): {e}')
        self._stop_event.clear()
        try:
            self._go_home()
        except _MotionInterrupt:
            self.get_logger().warn(f'{reason}: 홈 복귀 도중 추가 cancel — 무시')
        except Exception as e:
            self.get_logger().warn(f'{reason}: 홈 복귀 실패 (무시): {e}')
        self._stop_event.clear()

    def _gripper_open(self):
        self.get_logger().info('그리퍼 열기')
        self._call_service_with_retry(self.cli_gripper_open, Trigger.Request(), 'gripper/open', timeout=20.0)
        time.sleep(self.gripper_wait)

    def _apply_grip_current(self):
        """선택된 물체 클래스에 맞는 close_current를 gripper_node에 설정한다.
        맵에 없으면 기본값(미인식 물체)을 쓰고, 안전 범위로 clamp한다.
        파라미터 설정 실패는 치명적이지 않으므로 경고만 남기고 진행(직전 전류 사용)."""
        cls = self._target_object_class
        current = self.grip_current_map.get(cls, self.grip_current_default)
        current = max(self.grip_current_min, min(self.grip_current_max, int(current)))
        src = '맵' if cls in self.grip_current_map else '기본값(미인식)'
        self.get_logger().info(f'파지 강도: [{cls or "미선택"}] → {current}mA ({src})')

        if not self.cli_set_grip_current.service_is_ready():
            self.get_logger().warn('close_current 파라미터 서비스 미연결 — 강도 변경 생략')
            return
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name='close_current',
            value=ParameterValue(type=ParameterType.PARAMETER_INTEGER,
                                 integer_value=current))]
        try:
            self._call_service(self.cli_set_grip_current, req, 'set_close_current', timeout=3.0)
        except _MotionInterrupt:
            raise
        except Exception as e:
            self.get_logger().warn(f'파지 강도 설정 실패(무시하고 진행): {e}')

    def _gripper_close(self):
        """Trigger 서비스를 통해 그리퍼를 닫습니다."""
        # 격리 토글: false면 _apply_grip_current() 우회 (이전 세션 close 동작 그대로).
        # 그리퍼 close 실패가 이 코드 때문인지 분리 검증용.
        if self.enable_dynamic_grip_current:
            self._apply_grip_current()
        else:
            self.get_logger().info('그리퍼 닫기 (격리: 동적 강도 우회, 기본 close_current 사용)')
        self.get_logger().info('그리퍼 닫기')
        self._call_service_with_retry(self.cli_gripper_close, Trigger.Request(), 'gripper/close', timeout=20.0)
        time.sleep(self.gripper_wait)

    def _gripper_torque_off(self):
        """긴급정지(EMO) 시 그리퍼 토크를 OFF한다.
        best-effort·비차단·예외무시 — EMO 응답을 절대 막지 않도록 call_async만 쓴다.
        그리퍼는 Doosan EMO 회로와 분리된 별도 Modbus 장치라 EMO 때 직접 꺼줘야 한다."""
        if self.cli_gripper_stop.service_is_ready():
            try:
                self.cli_gripper_stop.call_async(Trigger.Request())
                self.get_logger().warn('⛔ EMO → 그리퍼 토크 OFF 명령 전송')
            except Exception as e:
                self.get_logger().error(f'그리퍼 토크 OFF 실패(무시): {e}')
        else:
            self.get_logger().error('⛔ EMO인데 gripper/stop 서비스 미연결 — 그리퍼 토크 OFF 못 함!')

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
