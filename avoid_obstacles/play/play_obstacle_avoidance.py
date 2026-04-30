#!/usr/bin/env python3
"""Play hierarchical obstacle avoidance policy.

- High-level policy: predicts base velocity command (vx, vy, wz)
- Low-level policy: frozen gait policy, maps command-conditioned obs to joint actions
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from dataclasses import dataclass
import math
import random

import numpy as np
import torch
from tensordict import TensorDict

from omni.isaac.orbit.app import AppLauncher
import omni


parser = argparse.ArgumentParser(description="Play trained obstacle-avoidance high-level policy.")
LEGACY_WALK_ONLY_CHECKPOINT = (
    "/home/dacheng.shen/Desktop/omniverse/avoid_obstacles/weights/model_1500.pt"
)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "rough"])
parser.add_argument("--goal_radius_min", type=float, default=2.0)
parser.add_argument("--goal_radius_max", type=float, default=6.0)
parser.add_argument("--goal_threshold", type=float, default=0.6)
parser.add_argument(
    "--fixed_layout",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Fix spawn, gap, obstacles, and goal placement for repeatable play.",
)
parser.add_argument("--spawn_x", type=float, default=0.0, help="Spawn x offset in env frame.")
parser.add_argument("--spawn_y", type=float, default=0.0, help="Spawn y offset in env frame.")
parser.add_argument(
    "--goal_radius_fixed",
    type=float,
    default=3.6,
    help="Fixed goal radius from env origin when fixed_layout is enabled.",
)
parser.add_argument("--ring_obstacle_count", type=int, default=20)
parser.add_argument("--ring_radius", type=float, default=2.2)
parser.add_argument("--obstacle_size_min", type=float, default=0.25)
parser.add_argument("--obstacle_size_max", type=float, default=0.7)
parser.add_argument(
    "--obstacle_gap_width",
    type=float,
    default=0.93,  # ~3x Go2 width (0.31 m)
    help="Gap width in meters on the obstacle ring.",
)
parser.add_argument("--enable_lidar_debug", action="store_true", default=False)
parser.add_argument(
    "--enable_height_scan_debug",
    action="store_true",
    default=False,
    help="Show ray-caster (height_scan) debug rays in GUI.",
)
parser.add_argument("--rtx_lidar_profile", type=str, default="Unitree_L1_PM_approx")
parser.add_argument("--sensor_mode", type=str, default="rtx_lidar", choices=["rtx_lidar", "height_scan"])
parser.add_argument("--lidar_bins", type=int, default=24)
parser.add_argument("--lidar_range_max", type=float, default=20.0)
parser.add_argument(
    "--temporal_min_frames",
    type=int,
    default=1,
    help="Per-bin min over latest N frames (1 disables temporal min).",
)
parser.add_argument(
    "--startup_hold_steps",
    type=int,
    default=80,
    help="Initial steps per episode to hold zero command so robot starts still.",
)
parser.add_argument(
    "--print_lidar_interval",
    type=int,
    default=100,
    help="Print lidar sample values every N play steps (0 disables).",
)
parser.add_argument(
    "--lidar_sample_count",
    type=int,
    default=24,
    help="Deprecated: all 24 lidar bins are always printed.",
)
parser.add_argument(
    "--freeze_robot",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Lock robot base pose every step for stationary lidar inspection.",
)
parser.add_argument(
    "--enable_rtx_scan_debug",
    action="store_true",
    default=True,
    help="Draw RTX lidar scan rays from env_0 robot for visualization.",
)
parser.add_argument(
    "--low_level_checkpoint",
    type=str,
    default=LEGACY_WALK_ONLY_CHECKPOINT,
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stuck_speed_threshold", type=float, default=0.08)
parser.add_argument("--stuck_steps", type=int, default=100)
parser.add_argument(
    "--walk_only",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Bypass high-level policy and obstacles; run low-level walking only.",
)
parser.add_argument("--walk_cmd_x", type=float, default=1.0)
parser.add_argument("--walk_cmd_y", type=float, default=0.0)
parser.add_argument("--walk_cmd_z", type=float, default=0.0, help="Yaw rate command for low-level walking.")
parser.add_argument(
    "--legacy_joint_remap",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Remap joint-related obs/actions between Isaac Sim joint order and legacy IsaacGym order.",
)
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

torch.manual_seed(args_cli.seed)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


WORK_ROOT = "/home/dacheng.shen/Desktop/omniverse/avoid_obstacles"
PLAY_DIR = os.path.join(WORK_ROOT, "play")
CONFIG_DIR = os.path.join(WORK_ROOT, "configs")

if PLAY_DIR not in sys.path:
    sys.path.insert(0, PLAY_DIR)
if CONFIG_DIR not in sys.path:
    sys.path.insert(0, CONFIG_DIR)

# custom_rl_env expects `from omniverse_sim import args_cli`.
sys.modules["omniverse_sim"] = types.SimpleNamespace(args_cli=args_cli)

import gymnasium as gym
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

import custom_rl_env
from custom_rl_env import UnitreeGo2CustomEnvCfg
from agent_cfg import unitree_go2_agent_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.usd


def _adapt_legacy_agent_cfg(cfg: dict) -> dict:
    adapted = dict(cfg)
    adapted = {k: (v.copy() if isinstance(v, dict) else v) for k, v in adapted.items()}
    algorithm_cfg = adapted.setdefault("algorithm", {})
    algorithm_cfg.setdefault("class_name", "PPO")
    adapted.setdefault("obs_groups", {})

    if "policy" in adapted and ("actor" not in adapted or "critic" not in adapted):
        policy_cfg = adapted.pop("policy")
        activation = policy_cfg.get("activation", "elu")
        actor_hidden_dims = policy_cfg.get("actor_hidden_dims", [512, 256, 128])
        critic_hidden_dims = policy_cfg.get("critic_hidden_dims", [512, 256, 128])
        init_noise_std = policy_cfg.get("init_noise_std", 1.0)

        adapted["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": actor_hidden_dims,
            "activation": activation,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": init_noise_std,
            },
        }
        adapted["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": critic_hidden_dims,
            "activation": activation,
        }

    if not adapted.get("obs_groups"):
        adapted["obs_groups"] = {"actor": ["policy"], "critic": ["policy"]}

    return adapted


def _load_legacy_checkpoint_if_needed(ppo_runner: OnPolicyRunner, checkpoint_path: str) -> None:
    try:
        ppo_runner.load(checkpoint_path)
        return
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Standard checkpoint load failed: {exc}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint.get("model_state_dict")
    if model_state is None:
        raise RuntimeError("Unsupported checkpoint format: 'model_state_dict' missing.")

    actor_state = {}
    critic_state = {}
    for key, value in model_state.items():
        if key.startswith("actor."):
            actor_state[f"mlp.{key[len('actor.'):]}"] = value
        elif key.startswith("critic."):
            critic_state[f"mlp.{key[len('critic.'):]}"] = value
        elif key == "std":
            actor_state["distribution.std_param"] = value

    ppo_runner.alg.actor.load_state_dict(actor_state, strict=False)
    ppo_runner.alg.critic.load_state_dict(critic_state, strict=False)
    print("[INFO] Loaded legacy checkpoint weights with compatibility mapping.")


def _patch_env_observation_api(env) -> None:
    original_get_observations = env.get_observations
    original_step = env.step

    def _to_tensordict(obs_data):
        if isinstance(obs_data, TensorDict):
            return obs_data
        if isinstance(obs_data, tuple) and len(obs_data) > 0:
            obs_data = obs_data[0]
        if isinstance(obs_data, TensorDict):
            return obs_data
        if isinstance(obs_data, dict):
            sample = next(iter(obs_data.values()))
            return TensorDict(obs_data, batch_size=[sample.shape[0]], device=sample.device)
        if torch.is_tensor(obs_data):
            return TensorDict({"policy": obs_data}, batch_size=[obs_data.shape[0]], device=obs_data.device)
        raise TypeError(f"Unsupported observation type: {type(obs_data)}")

    def _compat_get_observations():
        return _to_tensordict(original_get_observations())

    def _compat_step(actions):
        obs, rewards, dones, extras = original_step(actions)
        return _to_tensordict(obs), rewards, dones, extras

    env.get_observations = _compat_get_observations
    env.step = _compat_step


def _patch_policy_obs_dim(env, policy_obs_dim: int) -> None:
    """Truncate 'policy' observation to a target dimension for low-level policy compatibility."""
    if policy_obs_dim <= 0:
        return
    original_get_observations = env.get_observations
    original_step = env.step

    def _truncate_obs_td(obs_td: TensorDict) -> TensorDict:
        if "policy" not in obs_td:
            return obs_td
        policy = obs_td["policy"]
        if not torch.is_tensor(policy) or policy.shape[-1] <= policy_obs_dim:
            return obs_td
        new_policy = policy[..., :policy_obs_dim]
        return TensorDict({"policy": new_policy}, batch_size=[new_policy.shape[0]], device=new_policy.device)

    def _compat_get_observations():
        return _truncate_obs_td(original_get_observations())

    def _compat_step(actions):
        obs, rewards, dones, extras = original_step(actions)
        return _truncate_obs_td(obs), rewards, dones, extras

    env.get_observations = _compat_get_observations
    env.step = _compat_step


def _infer_legacy_actor_input_dim(checkpoint_path: str) -> int | None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to inspect low-level checkpoint: {exc}")
        return None
    model_state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(model_state, dict):
        return None
    weight = model_state.get("actor.0.weight")
    if weight is None or not hasattr(weight, "shape") or len(weight.shape) != 2:
        return None
    return int(weight.shape[1])


def _apply_walk_spawn_pose(base_env, spawn_x: float = 0.0, spawn_y: float = 0.0, spawn_z: float = 0.5) -> None:
    """Force a deterministic upright spawn pose for low-level walk-only mode."""
    try:
        robot = base_env.env.scene["robot"]
        env_origins = base_env.env.scene.env_origins
        root = robot.data.root_state_w.clone()
        root[:, 0] = env_origins[:, 0] + float(spawn_x)
        root[:, 1] = env_origins[:, 1] + float(spawn_y)
        root[:, 2] = env_origins[:, 2] + float(spawn_z)
        # Face +X with zero initial velocities.
        root[:, 3] = 1.0
        root[:, 4] = 0.0
        root[:, 5] = 0.0
        root[:, 6] = 0.0
        root[:, 7:13] = 0.0
        env_ids = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
        if hasattr(robot, "write_root_state_to_sim"):
            robot.write_root_state_to_sim(root, env_ids=env_ids)
        elif hasattr(robot, "write_root_pose_to_sim"):
            robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
            if hasattr(robot, "write_root_velocity_to_sim"):
                robot.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)
        # Also reset joints to default stand state to avoid immediate post-reset collapse.
        default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = torch.zeros_like(default_joint_pos)
        if hasattr(robot, "write_joint_state_to_sim"):
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        elif hasattr(robot, "write_joint_position_to_sim"):
            robot.write_joint_position_to_sim(default_joint_pos, env_ids=env_ids)
            if hasattr(robot, "write_joint_velocity_to_sim"):
                robot.write_joint_velocity_to_sim(default_joint_vel, env_ids=env_ids)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to enforce walk-only spawn pose: {exc}")


def _adapt_obs_to_isaacgym_low_level(obs_td: TensorDict) -> TensorDict:
    """Match Isaac Sim policy obs to IsaacGym low-level observation scaling."""
    if "policy" not in obs_td:
        return obs_td
    policy = obs_td["policy"]
    if not torch.is_tensor(policy) or policy.shape[-1] < 48:
        return obs_td
    scaled = policy.clone()
    # LeggedGym scales: lin_vel=2.0, ang_vel=0.25, dof_vel=0.05, commands=[2.0, 2.0, 0.25]
    scaled[:, 0:3] *= 2.0
    scaled[:, 3:6] *= 0.25
    scaled[:, 9:11] *= 2.0
    scaled[:, 11:12] *= 0.25
    scaled[:, 24:36] *= 0.05
    scaled = torch.clamp(scaled, -100.0, 100.0)
    return TensorDict({"policy": scaled}, batch_size=[scaled.shape[0]], device=scaled.device)


LEGACY_GO2_JOINT_ORDER = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


def _build_legacy_joint_mapping(base_env) -> torch.Tensor | None:
    """Build index map from legacy IsaacGym joint order to Isaac Sim joint order."""
    if not args_cli.legacy_joint_remap:
        return None
    try:
        sim_joint_names = list(base_env.env.scene["robot"].joint_names)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to query robot joint names for remap: {exc}")
        return None

    name_to_sim_idx = {name: idx for idx, name in enumerate(sim_joint_names)}
    missing = [name for name in LEGACY_GO2_JOINT_ORDER if name not in name_to_sim_idx]
    if missing:
        print(f"[WARN] Legacy remap disabled. Missing joints in Isaac Sim model: {missing}")
        return None

    map_legacy_to_sim = torch.tensor(
        [name_to_sim_idx[name] for name in LEGACY_GO2_JOINT_ORDER],
        dtype=torch.long,
        device=base_env.device,
    )
    is_identity = torch.equal(
        map_legacy_to_sim,
        torch.arange(len(LEGACY_GO2_JOINT_ORDER), dtype=torch.long, device=base_env.device),
    )
    print(f"[INFO] Isaac Sim joint order: {sim_joint_names}")
    if is_identity:
        print("[INFO] Legacy joint remap resolved as identity (no reorder needed).")
    else:
        print(f"[INFO] Applying legacy joint remap (legacy->sim): {map_legacy_to_sim.tolist()}")
    return map_legacy_to_sim


def _remap_obs_to_legacy_joint_order(obs_td: TensorDict, map_legacy_to_sim: torch.Tensor | None) -> TensorDict:
    """Convert policy obs joint-related chunks from sim-order to legacy-order."""
    if map_legacy_to_sim is None or "policy" not in obs_td:
        return obs_td
    policy = obs_td["policy"]
    if not torch.is_tensor(policy) or policy.shape[-1] < 48:
        return obs_td

    remapped = policy.clone()
    # [12:24]=joint_pos, [24:36]=joint_vel, [36:48]=last_actions
    remapped[:, 12:24] = policy[:, 12:24].index_select(1, map_legacy_to_sim)
    remapped[:, 24:36] = policy[:, 24:36].index_select(1, map_legacy_to_sim)
    remapped[:, 36:48] = policy[:, 36:48].index_select(1, map_legacy_to_sim)
    return TensorDict({"policy": remapped}, batch_size=[remapped.shape[0]], device=remapped.device)


def _remap_actions_to_sim_joint_order(
    legacy_actions: torch.Tensor, map_legacy_to_sim: torch.Tensor | None
) -> torch.Tensor:
    """Convert low-level policy actions from legacy-order to sim-order."""
    if map_legacy_to_sim is None:
        return legacy_actions
    sim_actions = torch.zeros_like(legacy_actions)
    sim_actions[:, map_legacy_to_sim] = legacy_actions
    return sim_actions


def _align_env_cfg_for_isaacgym_low_level(env_cfg: UnitreeGo2CustomEnvCfg) -> None:
    """Align Isaac Sim env config to the IsaacGym low-level policy assumptions."""
    # Legacy low-level policy is sensitive to observation noise.
    env_cfg.observations.policy.enable_corruption = False

    # Keep action interpretation consistent with IsaacGym training.
    env_cfg.actions.joint_pos.scale = 0.25

    # Align initialization and motor gains to the GO2 gym config.
    try:
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.42)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to set legacy-compatible init height: {exc}")

    try:
        actuator = env_cfg.scene.robot.actuators.get("base_legs", None)
        if actuator is not None:
            actuator.stiffness = 20.0
            actuator.damping = 0.5
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to set legacy-compatible PD gains: {exc}")


@dataclass
class GoalConfig:
    radius_min: float = 2.0
    radius_max: float = 6.0
    threshold: float = 0.6


class HighLevelAvoidanceVecEnv(VecEnv):
    """Play-time high-level vecenv wrapper over frozen low-level policy."""

    def __init__(
        self,
        base_env: RslRlVecEnvWrapper,
        low_level_policy,
        goal_cfg: GoalConfig,
        legacy_joint_map: torch.Tensor | None = None,
    ):
        self.base_env = base_env
        self.low_level_policy = low_level_policy
        self.legacy_joint_map = legacy_joint_map
        self.goal_cfg = goal_cfg

        self.num_envs = base_env.num_envs
        self.device = base_env.device
        self.max_episode_length = base_env.max_episode_length
        self.num_actions = 3
        self.sensor_mode = args_cli.sensor_mode
        self.lidar_bins = int(args_cli.lidar_bins)
        self.lidar_range_max = float(args_cli.lidar_range_max)
        self.temporal_min_frames = max(1, int(args_cli.temporal_min_frames))
        self.num_obs = self.lidar_bins + 3
        self.num_privileged_obs = 0
        self.ring_obstacle_count = max(4, args_cli.ring_obstacle_count)
        self.ring_radius = max(0.8, args_cli.ring_radius)
        self.obstacle_gap_width = max(0.1, args_cli.obstacle_gap_width)
        self.obstacle_size_range = (args_cli.obstacle_size_min, args_cli.obstacle_size_max)
        self.fixed_layout = bool(args_cli.fixed_layout)
        self.spawn_xy = (float(args_cli.spawn_x), float(args_cli.spawn_y))
        self.spawn_z = 0.5
        self.goal_radius_fixed = float(args_cli.goal_radius_fixed)
        self.freeze_robot = bool(args_cli.freeze_robot)
        self._all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._gap_centers = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._gap_half_angle = min(math.pi * 0.8, (self.obstacle_gap_width / self.ring_radius) * 0.5)
        self._env_origins = self.base_env.env.scene.env_origins

        self.goal_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._rtx_lidar_sensors = []
        self._cached_lidar_scan = torch.full(
            (self.num_envs, self.lidar_bins), self.lidar_range_max, device=self.device, dtype=torch.float32
        )
        self._scan_history = torch.full(
            (self.num_envs, self.temporal_min_frames, self.lidar_bins),
            self.lidar_range_max,
            device=self.device,
            dtype=torch.float32,
        )
        self._scan_hist_idx = 0
        self._scan_hist_len = 0
        self._rtx_scan_draw = None
        self._stuck_counter = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._startup_hold_steps = max(0, int(args_cli.startup_hold_steps))
        # Important: create random obstacles before first sensor query.
        # Dynamic prim create/delete after contact views are built can trigger GPU view invalidation.
        self._spawn_random_obstacles_once()
        self._obs_td = self.base_env.get_observations()
        self._setup_rtx_lidar_if_needed()
        self._apply_fixed_spawn_pose(self._all_env_ids)
        self._high_obs, self._dist = self._build_high_obs(self._obs_td)
        self._high_obs_td = TensorDict({"policy": self._high_obs}, batch_size=[self.num_envs], device=self.device)
        self._resample_goals(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

        self.success_count = 0
        self.timeout_count = 0
        self.collision_count = 0
        self.episode_count = 0

    @property
    def cfg(self):
        return self.base_env.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.base_env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.base_env.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.base_env.seed(seed)

    def _robot_root_state(self) -> torch.Tensor:
        return self.base_env.env.scene["robot"].data.root_state_w

    def _resample_goals(self, mask: torch.Tensor) -> None:
        env_ids = mask.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        goal_r_min = max(self.goal_cfg.radius_min, self.ring_radius + 0.8)
        goal_r_max = max(goal_r_min + 0.1, self.goal_cfg.radius_max)
        if self.fixed_layout:
            theta = self._gap_centers[env_ids]
            radius_val = max(goal_r_min, min(goal_r_max, self.goal_radius_fixed))
            radius = torch.full((env_ids.numel(),), radius_val, device=self.device, dtype=torch.float32)
        else:
            gap_jitter = self._gap_half_angle * 0.6
            theta = self._gap_centers[env_ids] + (2.0 * torch.rand((env_ids.numel(),), device=self.device) - 1.0) * gap_jitter
            radius = goal_r_min + (goal_r_max - goal_r_min) * torch.rand((env_ids.numel(),), device=self.device)
        offsets = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), dim=-1)
        self.goal_xy[env_ids] = self._env_origins[env_ids, :2] + offsets

    def _apply_fixed_spawn_pose(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0 or not self.fixed_layout:
            return
        robot = self.base_env.env.scene["robot"]
        root = self._robot_root_state().clone()
        root[env_ids, 0] = self._env_origins[env_ids, 0] + self.spawn_xy[0]
        root[env_ids, 1] = self._env_origins[env_ids, 1] + self.spawn_xy[1]
        root[env_ids, 2] = self._env_origins[env_ids, 2] + self.spawn_z
        yaw = self._gap_centers[env_ids]
        root[env_ids, 3] = torch.cos(0.5 * yaw)
        root[env_ids, 4] = 0.0
        root[env_ids, 5] = 0.0
        root[env_ids, 6] = torch.sin(0.5 * yaw)
        root[env_ids, 7:13] = 0.0

        try:
            if hasattr(robot, "write_root_state_to_sim"):
                robot.write_root_state_to_sim(root[env_ids], env_ids=env_ids)
            elif hasattr(robot, "write_root_pose_to_sim"):
                robot.write_root_pose_to_sim(root[env_ids, :7], env_ids=env_ids)
                if hasattr(robot, "write_root_velocity_to_sim"):
                    robot.write_root_velocity_to_sim(root[env_ids, 7:13], env_ids=env_ids)
            self.base_env.env.sim.step()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to enforce fixed spawn pose: {exc}")

    def _setup_rtx_lidar_if_needed(self) -> None:
        if self.sensor_mode != "rtx_lidar":
            return
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_mgr.set_extension_enabled_immediate("omni.isaac.sensor", True)
        ext_mgr.set_extension_enabled_immediate("omni.sensors.nv.lidar", True)
        try:
            from omni.isaac.sensor import LidarRtx
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to import RTX lidar API: {exc}") from exc

        for env_id in range(self.num_envs):
            sensor = LidarRtx(
                f"/World/envs/env_{env_id}/Robot/base/lidar_sensor_hl",
                rotation_frequency=20,
                pulse_time=1,
                translation=(0.0, 0.0, 0.4),
                orientation=(1.0, 0.0, 0.0, 0.0),
                config_file_name=args_cli.rtx_lidar_profile,
            )
            sensor.add_point_cloud_data_to_frame()
            sensor.initialize()
            self._rtx_lidar_sensors.append(sensor)
        if args_cli.enable_rtx_scan_debug:
            try:
                ext_mgr.set_extension_enabled_immediate("omni.isaac.debug_draw", True)
                import omni.isaac.debug_draw._debug_draw as omni_debug_draw

                self._rtx_scan_draw = omni_debug_draw.acquire_debug_draw_interface()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to init RTX scan debug draw: {exc}")
        print(
            f"[INFO] High-level input uses RTX lidar profile={args_cli.rtx_lidar_profile}, "
            f"bins={self.lidar_bins}, temporal_min_frames={self.temporal_min_frames}"
        )

    def _reset_scan_history(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._scan_history.fill_(self.lidar_range_max)
            self._scan_hist_idx = 0
            self._scan_hist_len = 0
            return
        if env_ids.numel() == 0:
            return
        self._scan_history[env_ids] = self.lidar_range_max

    def _read_rtx_lidar_scan(self) -> torch.Tensor:
        if self.sensor_mode != "rtx_lidar":
            return self._cached_lidar_scan
        scans = torch.full((self.num_envs, self.lidar_bins), self.lidar_range_max, dtype=torch.float32, device=self.device)
        edges = np.linspace(-np.pi, np.pi, self.lidar_bins + 1)
        for env_id, sensor in enumerate(self._rtx_lidar_sensors):
            frame = sensor.get_current_frame()
            pc = frame.get("point_cloud_data", None)
            if pc is None or len(pc) == 0:
                scans[env_id] = self._cached_lidar_scan[env_id]
                continue
            pts = np.asarray(pc, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] < 2:
                scans[env_id] = self._cached_lidar_scan[env_id]
                continue
            x = pts[:, 0]
            y = pts[:, 1]
            ranges = np.sqrt(x * x + y * y)
            angles = np.arctan2(y, x)
            valid = np.isfinite(ranges) & np.isfinite(angles) & (ranges > 0.05) & (ranges < self.lidar_range_max)
            binned = np.full((self.lidar_bins,), self.lidar_range_max, dtype=np.float32)
            if np.any(valid):
                idx = np.digitize(angles[valid], edges) - 1
                idx = np.clip(idx, 0, self.lidar_bins - 1)
                np.minimum.at(binned, idx, ranges[valid])
            scans[env_id] = torch.from_numpy(binned).to(device=self.device)
        self._scan_history[:, self._scan_hist_idx, :] = scans
        self._scan_hist_idx = (self._scan_hist_idx + 1) % self.temporal_min_frames
        self._scan_hist_len = min(self._scan_hist_len + 1, self.temporal_min_frames)
        scans_temporal = self._scan_history[:, : self._scan_hist_len, :].amin(dim=1)
        self._cached_lidar_scan = scans_temporal
        return scans_temporal

    def _draw_rtx_scan_overlay(self, lidar_scan: torch.Tensor) -> None:
        if self._rtx_scan_draw is None or self.num_envs < 1:
            return
        root = self._robot_root_state()[0]
        base_pos = root[:3].detach().float().cpu()
        quat = root[3:7].detach().float().cpu()  # wxyz
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        angles = torch.linspace(-torch.pi, torch.pi, self.lidar_bins, device=lidar_scan.device)
        ranges = torch.clamp(lidar_scan[0], 0.05, self.lidar_range_max).detach().cpu()
        c = torch.cos(angles + yaw).cpu()
        s = torch.sin(angles + yaw).cpu()
        start = torch.tensor([base_pos[0], base_pos[1], base_pos[2] + 0.4], dtype=torch.float32)
        ends = torch.stack(
            (
                start[0] + ranges * c,
                start[1] + ranges * s,
                torch.full_like(ranges, start[2]),
            ),
            dim=-1,
        )
        starts = start.repeat(self.lidar_bins, 1)
        self._rtx_scan_draw.clear_lines()
        colors = [[0.2, 1.0, 0.2, 0.9]] * self.lidar_bins
        sizes = [1.0] * self.lidar_bins
        self._rtx_scan_draw.draw_lines(starts.tolist(), ends.tolist(), colors, sizes)

    def _build_high_obs(self, obs_td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        policy_obs = obs_td["policy"]
        if self.sensor_mode == "rtx_lidar":
            lidar_scan = self._read_rtx_lidar_scan()
            if args_cli.enable_rtx_scan_debug:
                self._draw_rtx_scan_overlay(lidar_scan)
        else:
            height_scan = policy_obs[:, -187:]
            if self.lidar_bins == 187:
                lidar_scan = height_scan
            else:
                lidar_scan = torch.nn.functional.interpolate(
                    height_scan.unsqueeze(1), size=self.lidar_bins, mode="linear", align_corners=False
                ).squeeze(1)

        root = self._robot_root_state()
        pos_xy = root[:, :2]
        quat = root[:, 3:7]  # wxyz
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        rel_world = self.goal_xy - pos_xy
        dx, dy = rel_world[:, 0], rel_world[:, 1]
        c, s = torch.cos(yaw), torch.sin(yaw)
        rel_body_x = c * dx + s * dy
        rel_body_y = -s * dx + c * dy
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)

        goal_obs = torch.stack((rel_body_x, rel_body_y, dist), dim=-1)
        high_obs = torch.cat((lidar_scan, goal_obs), dim=-1)
        return high_obs, dist

    def _reset_base_env_ids(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        base = self.base_env.env
        try:
            base.reset_idx(env_ids)
        except Exception:
            _ = base.reset()

    def _spawn_random_obstacles_once(self) -> None:
        stage = omni.usd.get_context().get_stage()
        env_origins = self.base_env.env.scene.env_origins
        size_min = min(self.obstacle_size_range)
        size_max = max(self.obstacle_size_range)
        size_fixed = 0.5 * (size_min + size_max)
        for env_id in range(self.num_envs):
            root_prim = f"/World/envs/env_{env_id}/RandomObstacles"
            if not stage.GetPrimAtPath(root_prim).IsValid():
                prim_utils.create_prim(root_prim, "Xform")

            center_x = float(env_origins[env_id, 0].item())
            center_y = float(env_origins[env_id, 1].item())
            # Keep one deterministic opening (+x) so the spawn faces toward the gap.
            gap_center = 0.0
            self._gap_centers[env_id] = gap_center
            gap_angle = self._gap_half_angle * 2.0
            arc_len = max(1e-6, 2.0 * math.pi - gap_angle)
            for i in range(self.ring_obstacle_count):
                size = size_fixed if self.fixed_layout else random.uniform(size_min, size_max)
                frac = (i + 0.5) / float(self.ring_obstacle_count)
                theta_local = self._gap_half_angle + frac * arc_len
                theta = theta_local + gap_center
                x = center_x + self.ring_radius * math.cos(theta)
                y = center_y + self.ring_radius * math.sin(theta)
                z = size * 0.5
                prim_path = f"{root_prim}/obs_{i}"
                if stage.GetPrimAtPath(prim_path).IsValid():
                    continue
                cube_cfg = sim_utils.CuboidCfg(
                    size=(size, size, size),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2), roughness=0.7),
                )
                cube_cfg.func(prim_path, cube_cfg, translation=(x, y, z))

    def metrics(self) -> dict:
        success_rate = (self.success_count / max(1, self.episode_count)) * 100.0
        collision_rate = (self.collision_count / max(1, self.episode_count)) * 100.0
        timeout_rate = (self.timeout_count / max(1, self.episode_count)) * 100.0
        mean_goal_dist = float(self._dist.mean().item())
        return {
            "episodes": self.episode_count,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "timeout_rate": timeout_rate,
            "mean_goal_dist": mean_goal_dist,
        }

    def lidar_samples_text(self, env_id: int = 0, count: int = 24) -> str:
        if self._high_obs is None or self._high_obs.shape[0] <= env_id:
            return "lidar unavailable"
        idx = list(range(self.lidar_bins))
        scan = self._high_obs[env_id, : self.lidar_bins].detach().float().cpu().numpy()
        parts = [f"bin{j}:{scan[j]:.2f}m" for j in idx]
        return ", ".join(parts)

    def lidar_bin_mapping_text(self) -> str:
        edges = np.linspace(-180.0, 180.0, self.lidar_bins + 1, dtype=np.float32)
        lines = []
        for i in range(self.lidar_bins):
            left = float(edges[i])
            right = float(edges[i + 1])
            center = 0.5 * (left + right)
            lines.append(
                f"bin{i:02d}: yaw[{left:+06.1f},{right:+06.1f}) deg, center {center:+06.1f} deg"
            )
        return "\n".join(lines)

    def get_observations(self):
        return self._high_obs_td

    def reset(self):
        self._reset_scan_history()
        self._obs_td = self.base_env.get_observations()
        self._apply_fixed_spawn_pose(self._all_env_ids)
        self._obs_td = self.base_env.get_observations()
        self._resample_goals(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self._high_obs, self._dist = self._build_high_obs(self._obs_td)
        self._high_obs_td = TensorDict({"policy": self._high_obs}, batch_size=[self.num_envs], device=self.device)
        return self._high_obs_td

    def step(self, actions: torch.Tensor):
        cmd = torch.zeros_like(actions)
        if not self.freeze_robot:
            cmd[:, 0] = torch.clamp(actions[:, 0], -1.2, 1.2)
            cmd[:, 1] = torch.clamp(actions[:, 1], -0.8, 0.8)
            cmd[:, 2] = torch.clamp(actions[:, 2], -1.8, 1.8)
        if self.freeze_robot:
            cmd[:] = 0.0
        elif self._startup_hold_steps > 0:
            hold_mask = self.base_env.episode_length_buf < self._startup_hold_steps
            cmd[hold_mask] = 0.0

        for i in range(self.num_envs):
            custom_rl_env.base_command[str(i)] = [float(cmd[i, 0]), float(cmd[i, 1]), float(cmd[i, 2])]

        with torch.inference_mode():
            low_obs = _adapt_obs_to_isaacgym_low_level(self._obs_td)
            low_obs = _remap_obs_to_legacy_joint_order(low_obs, self.legacy_joint_map)
            low_actions_legacy = self.low_level_policy(low_obs)
            low_actions = _remap_actions_to_sim_joint_order(low_actions_legacy, self.legacy_joint_map)

        self._obs_td, _, low_dones, extras = self.base_env.step(low_actions)
        if self.freeze_robot:
            self._apply_fixed_spawn_pose(self._all_env_ids)
            self._obs_td = self.base_env.get_observations()
        self._high_obs, dist = self._build_high_obs(self._obs_td)
        self._high_obs_td = TensorDict({"policy": self._high_obs}, batch_size=[self.num_envs], device=self.device)

        success = dist < self.goal_cfg.threshold
        base_speed = torch.linalg.norm(self._obs_td["policy"][:, :2], dim=-1)
        low_speed = base_speed < args_cli.stuck_speed_threshold
        far_from_goal = dist > (self.goal_cfg.threshold * 1.5)
        self._stuck_counter = torch.where(
            low_speed & far_from_goal,
            self._stuck_counter + 1,
            torch.zeros_like(self._stuck_counter),
        )
        stuck = self._stuck_counter >= args_cli.stuck_steps

        done = (low_dones.to(dtype=torch.bool) | success | stuck)
        if torch.any(done):
            done_ids = done.nonzero(as_tuple=False).flatten()
            timeout_mask = done & (self.base_env.episode_length_buf >= (self.max_episode_length - 1))
            collision_mask = done & (~success) & (~timeout_mask) & (~stuck)
            self.success_count += int(success[done].sum().item())
            self.timeout_count += int(timeout_mask.sum().item())
            self.collision_count += int(collision_mask.sum().item())
            self.episode_count += int(done.sum().item())
            self._resample_goals(done)
            self._reset_scan_history(done_ids)
            self._stuck_counter[done] = 0
            self._reset_base_env_ids(done_ids)
            self._apply_fixed_spawn_pose(done_ids)
            self._obs_td = self.base_env.get_observations()
            self._high_obs, self._dist = self._build_high_obs(self._obs_td)
            self._high_obs_td = TensorDict({"policy": self._high_obs}, batch_size=[self.num_envs], device=self.device)

        # Play mode doesn't need a training reward.
        reward = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        return self._high_obs_td, reward, done.to(dtype=torch.long), extras

    def close(self):
        return self.base_env.close()


def _enable_rtx_lidar_debug(num_envs: int, profile_name: str) -> None:
    """Attach RTX LiDAR pointcloud debug writer for visualization."""
    try:
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_mgr.set_extension_enabled_immediate("omni.isaac.sensor", True)
        ext_mgr.set_extension_enabled_immediate("omni.sensors.nv.lidar", True)
        import omni.replicator.core as rep
        from omni.isaac.sensor import LidarRtx

        for i in range(num_envs):
            lidar_sensor = LidarRtx(
                f"/World/envs/env_{i}/Robot/base/lidar_sensor",
                rotation_frequency=200,
                pulse_time=1,
                translation=(0.0, 0.0, 0.4),
                orientation=(1.0, 0.0, 0.0, 0.0),
                config_file_name=profile_name,
            )
            writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
            writer.attach([lidar_sensor.get_render_product_path()])
        print(f"[INFO] RTX LiDAR debug enabled with profile: {profile_name}")
    except Exception as e:
        # Keep play running even if RTX debug annotator is unavailable.
        print(f"[WARN] RTX LiDAR debug setup failed: {e}")
        print("[WARN] Continue without RTX pointcloud debug draw.")


def main():
    if args_cli.walk_only:
        if os.path.abspath(args_cli.low_level_checkpoint) != os.path.abspath(LEGACY_WALK_ONLY_CHECKPOINT):
            print(
                "[INFO] walk_only mode forces low-level checkpoint to legacy walking model. "
                f"Overriding: {args_cli.low_level_checkpoint} -> {LEGACY_WALK_ONLY_CHECKPOINT}"
            )
        args_cli.low_level_checkpoint = LEGACY_WALK_ONLY_CHECKPOINT

    if args_cli.enable_lidar_debug and args_cli.enable_rtx_scan_debug:
        print("[INFO] Pointcloud debug enabled; disabling green ray overlay.")
        args_cli.enable_rtx_scan_debug = False

    env_cfg = UnitreeGo2CustomEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    _align_env_cfg_for_isaacgym_low_level(env_cfg)
    # Avoid overlay conflict: in RTX mode, show only RTX scan debug if enabled.
    env_cfg.scene.height_scanner.debug_vis = bool(args_cli.enable_height_scan_debug and args_cli.sensor_mode == "height_scan")

    for i in range(env_cfg.scene.num_envs):
        custom_rl_env.base_command[str(i)] = [0.0, 0.0, 0.0]

    print(f"[INFO] Creating base env: task={args_cli.task}, num_envs={env_cfg.scene.num_envs}, terrain={args_cli.terrain}")
    print(f"[INFO] High-level sensor mode: {args_cli.sensor_mode}")
    print(f"[INFO] Target observation dim: {args_cli.lidar_bins + 3} (lidar_bins={args_cli.lidar_bins})")
    print("[INFO] Legacy low-level alignment: obs_noise=OFF, action_scale=0.25, init_z=0.42, PD=(20.0, 0.5)")
    print(
        "[INFO] Fixed layout: "
        f"{args_cli.fixed_layout} (spawn=({args_cli.spawn_x:.2f},{args_cli.spawn_y:.2f}), "
        f"gap_center=+x axis, goal_radius_fixed={args_cli.goal_radius_fixed:.2f})"
    )
    print(f"[INFO] Freeze robot for lidar inspection: {args_cli.freeze_robot}")
    base_env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = RslRlVecEnvWrapper(base_env)
    _patch_env_observation_api(base_env)
    low_actor_in_dim = _infer_legacy_actor_input_dim(args_cli.low_level_checkpoint)
    if low_actor_in_dim is not None:
        sample_obs = base_env.get_observations()["policy"]
        base_obs_dim = int(sample_obs.shape[-1])
        if base_obs_dim > low_actor_in_dim:
            print(
                f"[INFO] Low-level checkpoint expects {low_actor_in_dim} dims, "
                f"base env provides {base_obs_dim} dims. Truncating policy obs to first {low_actor_in_dim} dims."
            )
            _patch_policy_obs_dim(base_env, low_actor_in_dim)
        else:
            print(f"[INFO] Low-level obs dim match: {base_obs_dim}.")
    else:
        print("[WARN] Could not infer low-level checkpoint input dim; using full policy observation.")

    # Low-level (frozen gait) policy.
    low_cfg = _adapt_legacy_agent_cfg(unitree_go2_agent_cfg)
    low_runner = OnPolicyRunner(base_env, low_cfg, log_dir=None, device=low_cfg["device"])
    print(f"[INFO] Loading low-level checkpoint: {args_cli.low_level_checkpoint}")
    _load_legacy_checkpoint_if_needed(low_runner, args_cli.low_level_checkpoint)
    low_policy = low_runner.get_inference_policy(device=base_env.device)
    legacy_joint_map = _build_legacy_joint_mapping(base_env)

    if args_cli.walk_only:
        print(
            "[INFO] Walk-only mode enabled: "
            f"command(vx, vy, wz)=({args_cli.walk_cmd_x:.2f}, {args_cli.walk_cmd_y:.2f}, {args_cli.walk_cmd_z:.2f})"
        )
        obs_td = base_env.reset()
        _apply_walk_spawn_pose(base_env, spawn_x=args_cli.spawn_x, spawn_y=args_cli.spawn_y, spawn_z=0.5)
        obs_td = base_env.get_observations()
        step_idx = 0
        while simulation_app.is_running():
            for i in range(base_env.num_envs):
                custom_rl_env.base_command[str(i)] = [
                    float(args_cli.walk_cmd_x),
                    float(args_cli.walk_cmd_y),
                    float(args_cli.walk_cmd_z),
                ]
            with torch.inference_mode():
                low_obs = _adapt_obs_to_isaacgym_low_level(obs_td)
                low_obs = _remap_obs_to_legacy_joint_order(low_obs, legacy_joint_map)
                low_actions_legacy = low_policy(low_obs)
                low_actions = _remap_actions_to_sim_joint_order(low_actions_legacy, legacy_joint_map)
            obs_td, _, dones, _ = base_env.step(low_actions)
            has_nan = not torch.isfinite(obs_td["policy"]).all()
            step_idx += 1
            if has_nan:
                print(f"[WARN] NaN detected at step={step_idx}; resetting environment.")
                obs_td = base_env.reset()
                _apply_walk_spawn_pose(base_env, spawn_x=args_cli.spawn_x, spawn_y=args_cli.spawn_y, spawn_z=0.5)
                obs_td = base_env.get_observations()
                continue
            if torch.any(dones.to(dtype=torch.bool)):
                print(f"[WARN] Episode reset at step={step_idx}; re-applying spawn pose.")
                obs_td = base_env.reset()
                _apply_walk_spawn_pose(base_env, spawn_x=args_cli.spawn_x, spawn_y=args_cli.spawn_y, spawn_z=0.5)
                obs_td = base_env.get_observations()
            if step_idx % 200 == 0:
                base_lin = obs_td["policy"][:, :3].mean(dim=0).detach().cpu()
                cmd_obs = obs_td["policy"][:, 9:12].mean(dim=0).detach().cpu()
                act_stats = low_actions.mean(dim=0).detach().cpu()
                cmd_target = (
                    float(args_cli.walk_cmd_x),
                    float(args_cli.walk_cmd_y),
                    float(args_cli.walk_cmd_z),
                )
                print(
                    f"[INFO] Walk step={step_idx} "
                    f"target_cmd_xyz=({cmd_target[0]:.3f}, {cmd_target[1]:.3f}, {cmd_target[2]:.3f}) "
                    f"actual_vel_xyz=({base_lin[0]:.3f}, {base_lin[1]:.3f}, {base_lin[2]:.3f}) "
                    f"cmd_obs(mean)=({cmd_obs[0]:.3f}, {cmd_obs[1]:.3f}, {cmd_obs[2]:.3f}) "
                    f"act(mean,std)=({act_stats.mean().item():.3f}, {low_actions.std().item():.3f})"
                )
        base_env.close()
        return

    raise RuntimeError(
        "This script now supports walking-only migration with the legacy low-level checkpoint only. "
        "Please run with --walk_only."
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
