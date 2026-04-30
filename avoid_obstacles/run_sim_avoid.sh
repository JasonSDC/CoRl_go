#!/usr/bin/env bash
set -eo pipefail

WORK_ROOT="/home/dacheng.shen/Desktop/omniverse/avoid_obstacles"
OMNI_ROOT="/home/dacheng.shen/Desktop/omniverse"
PLAY_DIR="${WORK_ROOT}/play"

export GO2_ENABLE_ROS2=${GO2_ENABLE_ROS2:-0}

if [ "${GO2_ENABLE_ROS2}" = "1" ]; then
  source /opt/ros/jazzy/setup.bash
fi

unset PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate orbit
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONPATH="${CONDA_PREFIX}/lib/python3.10/site-packages"

ISAAC_SIM_ROOT=""
if [ -f "${OMNI_ROOT}/IsaacLab/_isaac_sim/setup_conda_env.sh" ]; then
  ISAAC_SIM_ROOT="${OMNI_ROOT}/IsaacLab/_isaac_sim"
elif [ -f "${HOME}/.local/share/ov/pkg/setup_conda_env.sh" ]; then
  ISAAC_SIM_ROOT="${HOME}/.local/share/ov/pkg"
else
  echo "[ERROR] Isaac Sim _isaac_sim not found." >&2
  exit 1
fi

# shellcheck source=/dev/null
. "${ISAAC_SIM_ROOT}/setup_conda_env.sh"

NVRTC_TESTLIBS="${ISAAC_SIM_ROOT}/kit/testlibs"
if [ -d "${NVRTC_TESTLIBS}" ]; then
  export LD_LIBRARY_PATH="${NVRTC_TESTLIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Keep IsaacLab lidar JSONs in sync with this workspace copy.
"${WORK_ROOT}/scripts/sync_lidar_profiles.sh"

GO2_HEADLESS="${GO2_HEADLESS:-1}"
export GO2_MULTI_GPU="${GO2_MULTI_GPU:-0}"
HEADLESS_ARGS=()
if [ "${GO2_HEADLESS}" = "1" ]; then
  HEADLESS_ARGS=(--headless)
fi

echo "[INFO] WORK_ROOT=${WORK_ROOT}"
echo "[INFO] GO2_HEADLESS=${GO2_HEADLESS} GO2_MULTI_GPU=${GO2_MULTI_GPU}"

cd "${PLAY_DIR}"
python main.py --robot_amount 1 --robot go2 --terrain flat "${HEADLESS_ARGS[@]}" "$@"
