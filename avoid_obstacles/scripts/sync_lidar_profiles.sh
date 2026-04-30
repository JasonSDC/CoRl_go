#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/dacheng.shen/Desktop/omniverse/avoid_obstacles"
TARGET_DIR="/home/dacheng.shen/Desktop/omniverse/IsaacLab/source/data/sensors/lidar"

cp "${WORK_ROOT}/env/lidar/Unitree_L1_PM_approx.json" "${TARGET_DIR}/Unitree_L1_PM_approx.json"
cp "${WORK_ROOT}/env/lidar/Unitree_L1_RM_approx.json" "${TARGET_DIR}/Unitree_L1_RM_approx.json"

echo "[OK] Synced PM/RM lidar profiles to ${TARGET_DIR}"
