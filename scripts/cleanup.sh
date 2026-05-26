#!/usr/bin/env bash
# cleanup.sh — Pick & Place 실행 전 좀비 프로세스 일괄 정리 스크립트
#
# 사용법:
#   bash src/mini_project/scripts/cleanup.sh          # 정리 후 종료
#   bash src/mini_project/scripts/cleanup.sh --wait   # 정리 후 DRCF 안정화 대기(90초)
#
# launch 파일 자동 실행:
#   pick_place.launch.py 의 pre_cleanup 인수를 true 로 설정하면 자동 호출된다.

set -euo pipefail

WAIT_MODE=false
for arg in "$@"; do
    [[ "$arg" == "--wait" ]] && WAIT_MODE=true
done

echo "[cleanup] ===== Pick & Place 사전 정리 시작 ====="

# ── 1. RealSense 관련 프로세스 ─────────────────────────────────────────────
REALSENSE_PATTERNS=(
    "realsense2_camera_node"
    "rs2_"
    "realsense"
)
echo "[cleanup] RealSense 프로세스 탐색 중..."
for pat in "${REALSENSE_PATTERNS[@]}"; do
    pids=$(pgrep -f "$pat" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "[cleanup]   [$pat] PID: $pids → kill -9"
        kill -9 $pids 2>/dev/null || true
    fi
done

# ── 2. ROS2 제어 관련 프로세스 ────────────────────────────────────────────
ROS2_PATTERNS=(
    "ros2_control_node"
    "pick_place_node"
    "gripper_service_node"
    "gripper_node"
    "object_detector"
    "gui_node"
    "dsr_controller2"
    "robot_state_publisher"
    "static_transform_publisher"
)
echo "[cleanup] ROS2 노드 프로세스 탐색 중..."
for pat in "${ROS2_PATTERNS[@]}"; do
    pids=$(pgrep -f "$pat" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "[cleanup]   [$pat] PID: $pids → kill -9"
        kill -9 $pids 2>/dev/null || true
    fi
done

# ── 3. 남은 Python ros2 프로세스 (포트 12345 바인딩 중인 것) ──────────────
echo "[cleanup] 포트 12345 점유 프로세스 탐색 중..."
port_pids=$(lsof -ti tcp:12345 2>/dev/null || true)
if [[ -n "$port_pids" ]]; then
    echo "[cleanup]   포트 12345 PID: $port_pids → kill -9"
    kill -9 $port_pids 2>/dev/null || true
fi

# ── 4. 정리 완료 대기 ────────────────────────────────────────────────────
echo "[cleanup] 프로세스 종료 대기 2초..."
sleep 2

# 남아 있는 프로세스 확인
remaining=""
for pat in "realsense2_camera_node" "ros2_control_node" "gripper_service_node"; do
    pids=$(pgrep -f "$pat" 2>/dev/null || true)
    [[ -n "$pids" ]] && remaining+="$pat($pids) "
done
if [[ -n "$remaining" ]]; then
    echo "[cleanup] ⚠️  아직 살아있는 프로세스: $remaining"
else
    echo "[cleanup] ✅ 모든 대상 프로세스 정리 완료."
fi

# ── 5. DRCF 안정화 대기 (--wait 옵션 시) ─────────────────────────────────
if [[ "$WAIT_MODE" == true ]]; then
    SETTLE_SEC=90
    echo "[cleanup] DRCF 안정화 대기 ${SETTLE_SEC}초 시작..."
    for i in $(seq $SETTLE_SEC -10 10); do
        echo "[cleanup]   남은 대기: ${i}초"
        sleep 10
    done
    echo "[cleanup] ✅ DRCF 안정화 대기 완료. 이제 launch 가능."
fi

echo "[cleanup] ===== 정리 완료 ====="
