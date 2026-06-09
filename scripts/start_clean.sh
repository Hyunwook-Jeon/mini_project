#!/usr/bin/env bash
# start_clean.sh — 잔재를 정리한 뒤 깨끗하게 launch한다 (테스트 반복 / status3 복구용).
#
# 매 launch마다 수동으로 kill하던 걸 한 방으로. 잔재가 있으면만 shutdown_nodes.sh로
# 정상 종료(DRCF/DRL/시리얼 해제)하고, 없으면 바로 launch한다.
#   사용:  bash start_clean.sh [host]      (host 기본 110.120.1.50)
#
# status3로 그리퍼가 죽었을 때도 이 스크립트로 재시작하면 된다
# (in-process 재초기화로는 RS-485를 못 잡으므로 클린 재시작이 유일한 복구다).

# NOTE: set -u는 쓰지 않는다 — ROS setup.bash가 미정의 변수를 참조해 -u와 충돌(launch 무산).
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${1:-110.120.1.50}"

echo "[start_clean] ===== 깨끗한 시작 ====="

# ── 1. 잔재가 있으면만 정상 종료 (없으면 빠르게 건너뜀) ──────────────────
if pgrep -f "ros2_control_node|gripper_service_node|pick_place_node|ros2 launch dsr_realsense" >/dev/null 2>&1; then
    echo "[start_clean] 기존 노드 감지 → 정상 종료(DRCF/DRL/시리얼 해제)..."
    bash "$SCRIPT_DIR/shutdown_nodes.sh" 2>/dev/null || true
    pkill -9 -f "ros2 launch dsr_realsense" 2>/dev/null || true
    sleep 2
else
    echo "[start_clean] 잔재 없음 — 바로 launch."
fi

# ── 2. 외부제어 포트 해제 확인 (점유 남아있으면 다음 시작이 막힘) ────────────
for i in 1 2 3; do
    if ss -tnp 2>/dev/null | grep -qE "20002|12345"; then
        echo "[start_clean] 포트(12345/20002) 아직 점유 — 대기 ($i/3)..."
        sleep 2
    else
        break
    fi
done

# ── 3. launch ────────────────────────────────────────────────────────────
echo "[start_clean] launch (host=$HOST)..."
source /opt/ros/humble/setup.bash
source "$HOME/kairos_ws/install/setup.bash"
export QT_QPA_PLATFORM=xcb
export DISPLAY="${DISPLAY:-:0}"
exec ros2 launch dsr_realsense_pick_place pick_place.launch.py mode:=real host:="$HOST"
