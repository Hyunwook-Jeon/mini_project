#!/usr/bin/env python3
"""
RH-P12-RN(A) Gripper ROS 2 Node  ─  v5  (DRL-only 방식)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v4 버그 수정
  ▸ [버그1] SerialSendData 제거
      flange_serial_open을 DrlStart로 실행하면 DRL 태스크가
      종료되는 순간 포트가 닫힘. 이후 SerialSendData 호출 시
      포트가 닫혀 있어 전송 실패.
  ▸ [버그2] SerialSendData 포트 불일치 가능성
      SerialSendData는 컨트롤러 박스 시리얼 포트를 대상으로
      할 수 있으며, Flange Serial에 도달하지 않을 수 있음.

해결 방식
  모든 그리퍼 명령을 DrlStart 하나로 처리.
  DRL 스크립트 안에 open → write → wait → close를 모두 포함.
  flange_serial_open 과 write 가 같은 DRL 태스크 안에 있으므로
  포트가 닫히지 않은 상태에서 패킷이 전송됨.

통신 경로
  gripper_node → DrlStart.code (DRL 스크립트 문자열)
              → dsr_control2 → DRFL API → TCP:컨트롤러
              → flange_serial_open / flange_serial_write
              → RS-485 → RH-P12-RN(A)
"""

import struct
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import JointState

from dsr_msgs2.srv import DrlStart


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Modbus RTU 패킷 빌더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ModbusRTU:
    @staticmethod
    def crc16(data: bytes) -> bytes:
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        return struct.pack('<H', crc)

    @classmethod
    def fc06(cls, slave_id: int, addr: int, value: int) -> bytes:
        """FC06: Write Single Register"""
        body = bytes([slave_id, 0x06]) + struct.pack('>HH', addr, value)
        return body + cls.crc16(body)

    @classmethod
    def fc16(cls, slave_id: int, start: int, values: list) -> bytes:
        """FC16: Write Multiple Registers"""
        n    = len(values)
        body = (bytes([slave_id, 0x10])
                + struct.pack('>HH', start, n)
                + bytes([n * 2]))
        for v in values:
            body += struct.pack('>H', v)
        return body + cls.crc16(body)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 레지스터 / 파라미터 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Reg:
    TORQUE_ENABLE = 256
    GOAL_CURRENT  = 275
    GOAL_POSITION = 282

class GP:
    SLAVE_ID     = 1
    STROKE_OPEN  = 0
    STROKE_CLOSE = 1000   # ★ 실측값으로 교체
    CUR_DEFAULT  = 400    # mA
    CUR_CUBE     = 300    # mA


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DRL 스크립트 빌더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_drl(packets: list[bytes], motion_wait: float = 0.0) -> str:
    """
    flange_serial_open → 패킷 전송 → (선택적 대기) → flange_serial_close
    를 하나의 DRL 스크립트 문자열로 반환합니다.

    packets      : 순서대로 전송할 Modbus RTU 바이트 패킷 리스트
    motion_wait  : 마지막 패킷 전송 후 그리퍼 동작 대기 시간 (초)
                   위치 명령 후 그리퍼가 이동을 완료할 때까지 필요
    """
    lines = [
        "flange_serial_open("
        "baudrate=57600, bytesize=DR_EIGHTBITS, "
        "parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)",
        "wait(0.2)",
    ]
    for i, pkt in enumerate(packets):
        # bytes → Python int 리스트 리터럴 → DRL flange_serial_write 인자
        lines.append(f"flange_serial_write(bytes({list(pkt)}))")
        # 패킷 사이 인터프레임 딜레이 (마지막 패킷 제외)
        if i < len(packets) - 1:
            lines.append("wait(0.1)")

    if motion_wait > 0:
        lines.append(f"wait({motion_wait})")

    lines.append("flange_serial_close()")
    return "\n".join(lines) + "\n"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROS 2 Node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GripperNode(Node):

    def __init__(self):
        super().__init__('rh_p12_rna_gripper')
        cb = ReentrantCallbackGroup()

        # ── 파라미터 ──────────────────────────────────────────────
        self.declare_parameter('robot_ns',     'dsr01')
        self.declare_parameter('svc_timeout',  10.0)   # DRL 실행 시간 포함
        self.declare_parameter('state_hz',     10.0)
        self.declare_parameter('init_current', GP.CUR_DEFAULT)
        self.declare_parameter('cube_current', GP.CUR_CUBE)
        self.declare_parameter('stroke_close', GP.STROKE_CLOSE)
        self.declare_parameter('motion_wait',  1.5)    # 위치 명령 후 대기 (초)

        ns              = self.get_parameter('robot_ns').value
        self._timeout   = self.get_parameter('svc_timeout').value
        self._cur_def   = self.get_parameter('init_current').value
        self._cur_cube  = self.get_parameter('cube_current').value
        self._st_close  = self.get_parameter('stroke_close').value
        self._mot_wait  = self.get_parameter('motion_wait').value

        # ── 서비스 클라이언트 (DrlStart 하나만 사용) ──────────────
        ns = self.get_parameter('robot_ns').value or 'dsr01'
        prefix = f'/{ns}'
        self._cli_drl = self.create_client(
            DrlStart, f'{prefix}/drl/drl_start',
            callback_group=cb)

        # ── 퍼블리셔 ──────────────────────────────────────────────
        self._pub = self.create_publisher(JointState, '/gripper/state', 10)
        self.create_timer(
            1.0 / self.get_parameter('state_hz').value,
            self._pub_state, callback_group=cb)

        # ── 서비스 서버 ───────────────────────────────────────────
        self.create_service(Trigger, '/gripper/open',
                            self._srv_open,   callback_group=cb)
        self.create_service(Trigger, '/gripper/close',
                            self._srv_close,  callback_group=cb)
        self.create_service(Trigger, '/gripper/stop',
                            self._srv_stop,   callback_group=cb)
        self.create_service(SetBool, '/gripper/enable',
                            self._srv_enable, callback_group=cb)

        # ── 내부 상태 ─────────────────────────────────────────────
        self._stroke = 0
        self._torque = False
        self._ready  = False

        # ── 초기화 타이머 (executor 기동 후 실행) ─────────────────
        self._init_timer = self.create_timer(
            0.5, self._init_once, callback_group=cb)

        self.get_logger().info("노드 생성 완료 — 초기화 대기 중 (0.5s)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # threading.Event 기반 서비스 호출
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _call_service(self, client, request, label: str):
        if not client.service_is_ready():
            self.get_logger().error(f"서비스 미연결: {label}")
            return None

        event  = threading.Event()
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
    # DRL 헬퍼
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _drl(self, code: str, label: str = "DrlStart") -> bool:
        """DRL 스크립트를 컨트롤러에 전송합니다."""
        self.get_logger().debug(f"[DRL] {label}\n{code}")
        req = DrlStart.Request()
        req.robot_system = 0
        req.code = code
        res = self._call_service(self._cli_drl, req, label)
        ok  = bool(res and res.success)
        if not ok:
            self.get_logger().error(f"[DRL 실패] {label}")
        return ok

    def _run_packets(self, packets: list[bytes],
                     motion_wait: float, label: str) -> bool:
        """패킷 리스트를 DRL 스크립트로 변환 후 실행합니다."""
        code = build_drl(packets, motion_wait)
        return self._drl(code, label)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 초기화 (one-shot 타이머 콜백)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _init_once(self):
        """executor 기동 후 한 번만 실행 — Torque Enable + Goal Current 초기값 설정"""
        self._init_timer.cancel()

        self.get_logger().info("그리퍼 초기화 시작...")

        # drl_start 서비스 연결 대기
        waited = 0.0
        while not self._cli_drl.service_is_ready() and waited < 30.0:
            time.sleep(0.5)
            waited += 0.5
        if not self._cli_drl.service_is_ready():
            self.get_logger().error("drl_start 서비스 연결 실패")
            return

        # DRL 스크립트: open → Torque Enable → Goal Current → close
        pkts = [
            ModbusRTU.fc06(GP.SLAVE_ID, Reg.TORQUE_ENABLE, 1),
            ModbusRTU.fc06(GP.SLAVE_ID, Reg.GOAL_CURRENT,  self._cur_def),
        ]
        if not self._run_packets(pkts, motion_wait=0.0, label="Init"):
            self.get_logger().error("초기화 실패")
            return

        self._torque = True
        self._ready  = True
        self.get_logger().info("초기화 완료 ✓  서비스 요청 수신 가능")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 이동
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _move(self, stroke: int, current: int) -> tuple:
        if not self._ready:
            return False, "초기화 미완료"
        if not self._torque:
            return False, "토크 OFF — /gripper/enable 먼저 호출"

        # DRL 스크립트: open → Goal Current → Goal Position → wait → close
        pkts = [
            ModbusRTU.fc06(GP.SLAVE_ID, Reg.GOAL_CURRENT,  current),
            ModbusRTU.fc16(GP.SLAVE_ID, Reg.GOAL_POSITION, [stroke, 0]),
        ]
        label = f"Move stroke={stroke} current={current}mA"
        ok = self._run_packets(pkts, motion_wait=self._mot_wait, label=label)

        if ok:
            self._stroke = stroke
        return ok, ("완료" if ok else "실패")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 상태 퍼블리시
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _pub_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name         = ['gripper_joint']
        msg.position     = [float(self._stroke)]
        msg.velocity     = [0.0]
        msg.effort       = [float(self._cur_def if self._torque else 0)]
        self._pub.publish(msg)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 서비스 핸들러
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _srv_open(self, _, res: Trigger.Response):
        res.success, res.message = self._move(GP.STROKE_OPEN, self._cur_def)
        return res

    def _srv_close(self, _, res: Trigger.Response):
        res.success, res.message = self._move(self._st_close, self._cur_cube)
        return res

    def _srv_stop(self, _, res: Trigger.Response):
        pkts = [ModbusRTU.fc06(GP.SLAVE_ID, Reg.TORQUE_ENABLE, 0)]
        res.success = self._run_packets(pkts, motion_wait=0.0, label="Torque OFF")
        res.message = "토크 OFF" if res.success else "실패"
        if res.success:
            self._torque = False
        return res

    def _srv_enable(self, req: SetBool.Request, res: SetBool.Response):
        val  = 1 if req.data else 0
        pkts = [ModbusRTU.fc06(GP.SLAVE_ID, Reg.TORQUE_ENABLE, val)]
        label = f"Torque {'ON' if req.data else 'OFF'}"
        res.success = self._run_packets(pkts, motion_wait=0.0, label=label)
        res.message = (label if res.success else "실패")
        if res.success:
            self._torque = bool(req.data)
        return res

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Motion Planning 노드 호출용 퍼블릭 메서드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def grip_cube(self) -> bool:
        ok, msg = self._move(self._st_close, self._cur_cube)
        self.get_logger().info(f"grip_cube → {msg}")
        return ok

    def release(self) -> bool:
        ok, msg = self._move(GP.STROKE_OPEN, self._cur_def)
        self.get_logger().info(f"release → {msg}")
        return ok

    def move_stroke(self, stroke: int, current: int | None = None) -> bool:
        ok, _ = self._move(stroke, current or self._cur_def)
        return ok

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 소멸자
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def destroy_node(self):
        self.get_logger().info("노드 종료 — 토크 OFF")
        pkts = [ModbusRTU.fc06(GP.SLAVE_ID, Reg.TORQUE_ENABLE, 0)]
        self._run_packets(pkts, motion_wait=0.0, label="shutdown")
        super().destroy_node()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        node = GripperNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()