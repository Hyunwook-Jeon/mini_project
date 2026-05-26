#!/usr/bin/env bash
# shutdown_nodes.sh — GUI를 제외한 모든 ROS2 노드를 순서대로 정상 종료한다.
#
# SIGTERM 순서:
#   1. ros2_control_node (DRCF 연결 해제 — 가장 중요, 충분한 대기 필요)
#   2. gripper_service_node / gripper_node
#   3. pick_place_node / object_detector / realsense / static_tf
#
# 종료 후 재실행 시 DRCF authority가 정상 해제되어 있으므로
# joint 활성화가 즉시 성공한다.

set -euo pipefail

echo "[shutdown] ===== 노드 정상 종료 시작 ====="

# ── 1. ros2_control_node — DRCF 연결 해제를 위해 먼저, 충분한 대기 ───────
echo "[shutdown] [1/3] ros2_control_node SIGTERM (DRCF 해제)..."
pkill -TERM -f "ros2_control_node" 2>/dev/null || true
sleep 6   # DRCF가 연결 해제를 처리할 시간 충분히 확보

# ── 2. 그리퍼 관련 노드 ───────────────────────────────────────────────────
echo "[shutdown] [2/3] gripper 노드 SIGTERM..."
pkill -TERM -f "gripper_service_node" 2>/dev/null || true
pkill -TERM -f "gripper_node" 2>/dev/null || true
sleep 2

# ── 3. 나머지 노드 ────────────────────────────────────────────────────────
echo "[shutdown] [3/3] 나머지 노드 SIGTERM..."
pkill -TERM -f "pick_place_node" 2>/dev/null || true
pkill -TERM -f "object_detector" 2>/dev/null || true
pkill -TERM -f "realsense2_camera_node" 2>/dev/null || true
pkill -TERM -f "static_transform_publisher" 2>/dev/null || true
pkill -TERM -f "robot_state_publisher" 2>/dev/null || true
pkill -TERM -f "dsr_controller2" 2>/dev/null || true
sleep 2

# ── 4. 잔여 프로세스 SIGKILL (ros2_control_node 제외 — 이미 충분히 대기함) ─
echo "[shutdown] 잔여 프로세스 정리..."
pkill -KILL -f "gripper_service_node|gripper_node|pick_place_node|object_detector|realsense2_camera_node|robot_state_publisher|static_transform_publisher|dsr_controller2" 2>/dev/null || true
# ros2_control_node는 마지막에 SIGKILL (이미 6초 대기했으므로 DRCF는 해제됨)
pkill -KILL -f "ros2_control_node" 2>/dev/null || true

echo "[shutdown] ===== 종료 완료 ====="
