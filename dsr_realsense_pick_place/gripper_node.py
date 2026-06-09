#!/usr/bin/env python3
# Doosan-E0509-ROBOTIS-RH-P12-RN-TCP-Bridge 패키지의 서비스를 래핑하는 ROS 2 그리퍼 제어 래퍼 노드.

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import JointState
from rcl_interfaces.msg import SetParametersResult

# TCP Bridge 패키지의 서비스 및 메시지 타입 임포트
from dsr_gripper_tcp_interfaces.srv import SetMotionProfile, SetPosition, SetTorque
from dsr_gripper_tcp_interfaces.msg import GripperState


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROS 2 Node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GripperNode(Node):

    def __init__(self):
        super().__init__('rh_p12_rna_gripper')
        cb = ReentrantCallbackGroup()

        # ── 파라미터 선언 ──────────────────────────────────────────────
        self.declare_parameter('robot_ns', 'dsr01')
        self.declare_parameter('svc_timeout', 10.0)
        self.declare_parameter('state_hz', 20.0)
        self.declare_parameter('open_current', 200)
        self.declare_parameter('close_current', 300)
        # 이송 전류 — 파지 후 들고 이동할 때 쓰는 낮은 전류. self-locking(기어비 1181:1)
        # 덕에 약한 전류로도 물체를 유지 → 발열·과압착 완화. (close_current로 물고 LIFT 후 전환)
        self.declare_parameter('transport_current', 150)
        # idle 위치락 전류 — 작업 완료(IDLE) 시 현재 위치를 goal로 락 + 이 낮은 전류로 유지.
        # Current-based 위치유지 미세토크(전류 튐/채터링)를 줄인다. self-locking이라 약해도 안 풀림.
        self.declare_parameter('idle_current', 50)
        self.declare_parameter('profile_velocity', 1500)
        self.declare_parameter('profile_acceleration', 1000)
        self.declare_parameter('stroke_open', 0)
        self.declare_parameter('stroke_close', 1000)
        self.declare_parameter('min_grip_pos', 500)
        self.declare_parameter('max_grip_pos', 700)
        # 파지 감지 임계 전류(mA) — close_current(전류 한계)와 분리한다.
        # _srv_close에서 present_current가 이 값 이상이면 물체 접촉(파지)으로 판단.
        # close_current(한계)보다 낮게 둬 포화 전에 접촉을 감지 → 빨리 락 → 갈아대기/status3 ↓.
        # 말랑한 물체는 전류가 한계까지 포화 안 돼서, 한계=감지 임계면 영영 감지 못 하던 문제 해소.
        self.declare_parameter('grasp_detect_current', 100)

        self._robot_ns = self.get_parameter('robot_ns').value
        self._timeout = self.get_parameter('svc_timeout').value
        
        self.open_current = self.get_parameter('open_current').value
        self.close_current = self.get_parameter('close_current').value
        self.transport_current = self.get_parameter('transport_current').value
        self.idle_current = self.get_parameter('idle_current').value
        self.profile_velocity = self.get_parameter('profile_velocity').value
        self.profile_acceleration = self.get_parameter('profile_acceleration').value
        self.stroke_open = self.get_parameter('stroke_open').value
        self.stroke_close = self.get_parameter('stroke_close').value
        self.min_grip_pos = self.get_parameter('min_grip_pos').value
        self.max_grip_pos = self.get_parameter('max_grip_pos').value
        self.grasp_detect_current = self.get_parameter('grasp_detect_current').value

        # 파라미터 동적 변경 콜백 등록
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # ── 서비스 클라이언트 (TCP Bridge 서비스 연동) ──────────────────
        # gripper_service 노드가 제공하는 서비스 호출
        self._cli_set_profile = self.create_client(
            SetMotionProfile, '/gripper_service/set_motion_profile',
            callback_group=cb)
        self._cli_set_position = self.create_client(
            SetPosition, '/gripper_service/set_position',
            callback_group=cb)
        self._cli_set_torque = self.create_client(
            SetTorque, '/gripper_service/set_torque',
            callback_group=cb)

        # ── 서브스크라이버 (TCP Bridge 상태 모니터링) ──────────────────
        self._sub_gripper_state = self.create_subscription(
            GripperState, '/gripper_service/state',
            self._cb_gripper_state, 10,
            callback_group=cb)

        # ── 퍼블리셔 (기존 /gripper/state 유지) ──────────────────────
        self._pub = self.create_publisher(JointState, '/gripper/state', 10)
        self.create_timer(
            1.0 / self.get_parameter('state_hz').value,
            self._pub_state, callback_group=cb)

        # ── 서비스 서버 (기존 서비스 유지) ───────────────────────────
        self.create_service(Trigger, '/gripper/open',
                            self._srv_open, callback_group=cb)
        self.create_service(Trigger, '/gripper/close',
                            self._srv_close, callback_group=cb)
        self.create_service(Trigger, '/gripper/stop',
                            self._srv_stop, callback_group=cb)
        self.create_service(Trigger, '/gripper/hold_transport',
                            self._srv_hold_transport, callback_group=cb)
        self.create_service(Trigger, '/gripper/hold_idle',
                            self._srv_hold_idle, callback_group=cb)
        self.create_service(SetBool, '/gripper/enable',
                            self._srv_enable, callback_group=cb)

        # ── 내부 상태 변수 ───────────────────────────────────────────
        self._last_state = None
        self._lock = threading.Lock()

        self.get_logger().info("그리퍼 래퍼 노드 기동 완료. TCP Bridge 서비스를 대기합니다.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 파라미터 업데이트 콜백
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _on_set_parameters(self, params):
        for param in params:
            if param.name == 'open_current':
                self.open_current = param.value
                self.get_logger().info(f"파라미터 변경: open_current -> {param.value}")
            elif param.name == 'close_current':
                self.close_current = param.value
                self.get_logger().info(f"파라미터 변경: close_current -> {param.value}")
            elif param.name == 'transport_current':
                self.transport_current = param.value
                self.get_logger().info(f"파라미터 변경: transport_current -> {param.value}")
            elif param.name == 'idle_current':
                self.idle_current = param.value
                self.get_logger().info(f"파라미터 변경: idle_current -> {param.value}")
            elif param.name == 'grasp_detect_current':
                self.grasp_detect_current = param.value
                self.get_logger().info(f"파라미터 변경: grasp_detect_current -> {param.value}")
            elif param.name == 'profile_velocity':
                self.profile_velocity = param.value
                self.get_logger().info(f"파라미터 변경: profile_velocity -> {param.value}")
            elif param.name == 'profile_acceleration':
                self.profile_acceleration = param.value
                self.get_logger().info(f"파라미터 변경: profile_acceleration -> {param.value}")
            elif param.name == 'stroke_open':
                self.stroke_open = param.value
                self.get_logger().info(f"파라미터 변경: stroke_open -> {param.value}")
            elif param.name == 'stroke_close':
                self.stroke_close = param.value
                self.get_logger().info(f"파라미터 변경: stroke_close -> {param.value}")
            elif param.name == 'min_grip_pos':
                self.min_grip_pos = param.value
                self.get_logger().info(f"파라미터 변경: min_grip_pos -> {param.value}")
            elif param.name == 'max_grip_pos':
                self.max_grip_pos = param.value
                self.get_logger().info(f"파라미터 변경: max_grip_pos -> {param.value}")
        return SetParametersResult(successful=True, reason='Parameters updated.')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TCP Bridge 상태 피드백 수신 콜백
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _cb_gripper_state(self, msg: GripperState):
        with self._lock:
            self._last_state = msg

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 상태 퍼블리시 콜백 (기존 JointState 토픽과 호환성 유지)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _pub_state(self):
        with self._lock:
            state = self._last_state

        if state is None:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['gripper_joint']
        # TCP Bridge의 0~1150 raw position값을 기존 GUI/노드가 그대로 받을 수 있게 float 리스트로 래핑
        msg.position = [float(state.present_position)]
        msg.velocity = [float(state.present_velocity)]
        # effort 값에 실시간 전류 피드백을 전달하여 GUI 등에서 모니터링 가능하게 호환 처리
        msg.effort = [float(state.present_current)]
        self._pub.publish(msg)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 비동기 서비스 호출 헬퍼
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _call_service(self, client, request, label: str):
        if not client.service_is_ready():
            self.get_logger().error(f"서비스 미연결: {label}")
            return None

        event = threading.Event()
        result = [None]

        def _done(future):
            result[0] = future
            event.set()

        future = client.call_async(request)
        future.add_done_callback(_done)

        if not event.wait(timeout=self._timeout):
            self.get_logger().error(f"타임아웃 ({self._timeout}s): {label}")
            return None

        try:
            return result[0].result()
        except Exception as e:
            self.get_logger().error(f"서비스 오류 [{label}]: {e}")
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 모션 제어 로직 구현
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _move(self, position: int, goal_current: int) -> tuple:
        # 1. 모션 프로파일(전류 한계, 속도, 가속도) 인가
        profile_req = SetMotionProfile.Request()
        profile_req.goal_current = goal_current
        profile_req.profile_velocity = self.profile_velocity
        profile_req.profile_acceleration = self.profile_acceleration
        
        self.get_logger().info(
            f"모션 프로파일 인가 요청: current={goal_current}mA, vel={self.profile_velocity}, acc={self.profile_acceleration}")
        
        profile_res = self._call_service(self._cli_set_profile, profile_req, "set_motion_profile")
        if not profile_res or not profile_res.success:
            return False, "모션 프로파일 설정 실패"

        # 2. 이동 명령 전송
        pos_req = SetPosition.Request()
        pos_req.position = position
        
        self.get_logger().info(f"이동 명령 전송: position={position}")
        pos_res = self._call_service(self._cli_set_position, pos_req, "set_position")
        if not pos_res or not pos_res.success:
            return False, "이동 명령 실행 실패"

        return True, "동작 완료"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 서비스 핸들러 구현
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _srv_open(self, _, res: Trigger.Response):
        # open은 무부하(여는) 방향이라 goal_current를 건드릴 이유가 없다.
        # _move처럼 set_motion_profile로 전류를 먼저 바꾸면, goal_position이 아직 닫힘값(이전)인
        # 채 전류만 바뀌어 set_position 전 짧게 close 방향으로 움직이는 '전류 침범'이 생긴다.
        # → 전류 프로파일은 건드리지 않고 position만 보내 시퀀스를 분리한다.
        pos_req = SetPosition.Request()
        pos_req.position = self.stroke_open
        pos_res = self._call_service(self._cli_set_position, pos_req, "set_position(open)")
        res.success = bool(pos_res and pos_res.success)
        res.message = "열기 완료" if (pos_res and pos_res.success) else "열기 실패"
        return res

    def _srv_hold_transport(self, _, res: Trigger.Response):
        # 이송 전류로 전환 — 위치 명령 없이 goal_current만 낮춘다(물체 문 채 전류만 변경).
        # apply_profile_settings가 토크를 끊지 않고 goal_current를 write하므로 물체를 안 놓친다.
        profile_req = SetMotionProfile.Request()
        profile_req.goal_current = self.transport_current
        profile_req.profile_velocity = self.profile_velocity
        profile_req.profile_acceleration = self.profile_acceleration
        profile_res = self._call_service(self._cli_set_profile, profile_req, "set_motion_profile(transport)")
        res.success = bool(profile_res and profile_res.success)
        res.message = f"이송 전류 전환 ({self.transport_current}mA)" if res.success else "이송 전류 전환 실패"
        return res

    def _srv_hold_idle(self, _, res: Trigger.Response):
        # idle 위치락 — 현재 위치를 goal로 명시(락) + idle_current(낮음)로 유지.
        # Current-based 위치유지 미세토크(전류 튐)를 완화. self-locking이라 약해도 위치 유지.
        with self._lock:
            state = self._last_state
        if state is None:
            res.success = False
            res.message = 'idle 위치락 실패 — 그리퍼 상태 미수신'
            return res
        pos = int(state.present_position)
        profile_req = SetMotionProfile.Request()
        profile_req.goal_current = self.idle_current
        profile_req.profile_velocity = self.profile_velocity
        profile_req.profile_acceleration = self.profile_acceleration
        self._call_service(self._cli_set_profile, profile_req, "set_motion_profile(idle)")
        pos_req = SetPosition.Request()
        pos_req.position = pos
        pos_res = self._call_service(self._cli_set_position, pos_req, "set_position(idle_hold)")
        res.success = bool(pos_res and pos_res.success)
        res.message = f'idle 위치락 (pos={pos}, {self.idle_current}mA)' if res.success else 'idle 위치락 실패'
        return res

    def _srv_close(self, _, res: Trigger.Response):
        # 닫기 명령만 전송한다. 파지 성공/실패 판정은 상위(pick_place)가 LIFT 후
        # 위치로 결정한다(설계안 v2). close 순간의 지터·통신노이즈에 휘둘리던
        # 2.5초 모니터·전류판정·call_async 위치락을 제거했다.
        ok, msg = self._move(self.stroke_close, self.close_current)
        res.success = ok
        res.message = "닫기 명령 전송" if ok else f"이동 명령 전송 실패 - {msg}"
        return res

    def _srv_stop(self, _, res: Trigger.Response):
        req = SetTorque.Request()
        req.enabled = False
        self.get_logger().info("토크 비활성화 요청")
        res_torque = self._call_service(self._cli_set_torque, req, "set_torque")
        res.success = bool(res_torque and res_torque.success)
        res.message = "토크 비활성화 완료" if res.success else "토크 비활성화 실패"
        return res

    def _srv_enable(self, req: SetBool.Request, res: SetBool.Response):
        torque_req = SetTorque.Request()
        torque_req.enabled = req.data
        label = f"토크 {'활성화' if req.data else '비활성화'}"
        self.get_logger().info(f"{label} 요청")
        res_torque = self._call_service(self._cli_set_torque, torque_req, "set_torque")
        res.success = bool(res_torque and res_torque.success)
        res.message = f"{label} 완료" if res.success else f"{label} 실패"
        return res

    def shutdown_safe(self, executor, timeout_sec: float = 2.0):
        """종료 시 토크를 끈다(best-effort).

        executor.spin()이 멈춘 뒤 destroy 직전에 호출되므로 call_async만으로는
        요청이 전송되지 않는다(future를 처리할 spin이 없음). 넘겨받은 executor로
        future를 직접 spin하여 동기적으로 전송한다.
        """
        try:
            if not self._cli_set_torque.service_is_ready():
                self.get_logger().warning('종료 — set_torque 서비스 미연결, 토크 OFF 생략')
                return
            req = SetTorque.Request()
            req.enabled = False
            future = self._cli_set_torque.call_async(req)
            executor.spin_until_future_complete(future, timeout_sec=timeout_sec)
            res = future.result() if future.done() else None
            if res is not None and res.success:
                self.get_logger().info('종료 — 토크 OFF 완료')
            else:
                self.get_logger().warning('종료 — 토크 OFF 응답 없음/실패(타임아웃)')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warning(f'종료 토크 OFF 실패: {e}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=4)
    node = None
    try:
        node = GripperNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None and rclpy.ok():
            node.shutdown_safe(executor)   # 종료 전 토크 OFF 동기 전송
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
