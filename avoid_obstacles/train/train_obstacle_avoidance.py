#!/usr/bin/env python3
"""Train high-level obstacle avoidance policy on top of frozen low-level gait policy.

High-level policy:
- input: LiDAR scan (default RTX L1 PM, 187 bins) + goal vector in body frame (dx, dy, dist)
- output: base command (vx, vy, wz)

Low-level policy:
- loaded from pretrained checkpoint (model_7850.pt)
- stays frozen and converts command-conditioned observation into joint actions
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import omni
import torch
from tensordict import TensorDict

from omni.isaac.orbit.app import AppLauncher


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train obstacle avoidance high-level policy.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--terrain", type=str, default="rough", choices=["flat", "rough"])
parser.add_argument("--robot_amount", type=int, default=1, help="Kept for compatibility; single-robot task uses 1.")
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--goal_radius_min", type=float, default=2.0)
parser.add_argument("--goal_radius_max", type=float, default=6.0)
parser.add_argument("--goal_threshold", type=float, default=0.6)
parser.add_argument("--sensor_mode", type=str, default="rtx_lidar", choices=["rtx_lidar", "height_scan"])
parser.add_argument("--rtx_lidar_profile", type=str, default="Unitree_L1_PM_approx")
parser.add_argument("--lidar_bins", type=int, default=24)
parser.add_argument("--lidar_range_max", type=float, default=20.0)
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
parser.add_argument(
    "--low_level_checkpoint",
    type=str,
    default="/home/dacheng.shen/Desktop/omniverse/avoid_obstacles/play/logs/rsl_rl/unitree_go2_rough/2024-04-06_02-37-07/model_7850.pt",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

torch.manual_seed(args_cli.seed)

# Launch sim app first.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# --------------------------------------------------------------------------------------
# Local imports (after app launch and argument setup)
# --------------------------------------------------------------------------------------
WORK_ROOT = "/home/dacheng.shen/Desktop/omniverse/avoid_obstacles"
PLAY_DIR = os.path.join(WORK_ROOT, "play")
CONFIG_DIR = os.path.join(WORK_ROOT, "configs")

if PLAY_DIR not in sys.path:
    sys.path.insert(0, PLAY_DIR)
if CONFIG_DIR not in sys.path:
    sys.path.insert(0, CONFIG_DIR)

# custom_rl_env expects `from omniverse_sim import args_cli`; provide a lightweight shim.
sys.modules["omniverse_sim"] = types.SimpleNamespace(args_cli=args_cli)

import gymnasium as gym
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

import custom_rl_env
from custom_rl_env import UnitreeGo2CustomEnvCfg
from agent_cfg import unitree_go2_agent_cfg
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.usd
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper


def _adapt_legacy_agent_cfg(cfg: dict) -> dict:
    """Adapt old rsl_rl config schema to current schema."""
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
    """Load old actor_critic checkpoint format if direct load fails."""
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
    """Normalize env.get_observations() return type for current rsl_rl API."""
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


@dataclass
class GoalConfig:
    radius_min: float = 2.0
    radius_max: float = 6.0
    threshold: float = 0.6


class HighLevelAvoidanceVecEnv(VecEnv):
    """Trainable high-level vecenv using frozen low-level gait policy."""

    def __init__(self, base_env: RslRlVecEnvWrapper, low_level_policy, goal_cfg: GoalConfig):
        self.base_env = base_env
        self.low_level_policy = low_level_policy
        self.goal_cfg = goal_cfg

        self.num_envs = base_env.num_envs
        self.device = base_env.device
        self.max_episode_length = base_env.max_episode_length
        self.num_actions = 3
        self.sensor_mode = args_cli.sensor_mode
        self.lidar_bins = int(args_cli.lidar_bins)
        self.lidar_range_max = float(args_cli.lidar_range_max)
        self.num_obs = self.lidar_bins + 3
        self.num_privileged_obs = 0
        self._rtx_lidar_sensors = []
        self.ring_obstacle_count = max(4, args_cli.ring_obstacle_count)
        self.ring_radius = max(0.8, args_cli.ring_radius)
        self.obstacle_gap_width = max(0.1, args_cli.obstacle_gap_width)
        self.obstacle_size_range = (args_cli.obstacle_size_min, args_cli.obstacle_size_max)
        self._gap_centers = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._gap_half_angle = min(math.pi * 0.8, (self.obstacle_gap_width / self.ring_radius) * 0.5)
        self._env_origins = self.base_env.env.scene.env_origins

        self.goal_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.prev_dist = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.prev_cmd = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._cached_lidar_scan = torch.full(
            (self.num_envs, self.lidar_bins), self.lidar_range_max, device=self.device, dtype=torch.float32
        )

        self._spawn_random_obstacles_once()
        self._obs_td = self.base_env.get_observations()
        self._setup_rtx_lidar_if_needed()
        self._resample_goals(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self._high_obs, self.prev_dist = self._build_high_obs(self._obs_td)
        self._high_obs_td = TensorDict({"policy": self._high_obs}, batch_size=[self.num_envs], device=self.device)

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
        # base_env.env is the wrapped Orbit RLTaskEnv
        return self.base_env.env.scene["robot"].data.root_state_w

    def _resample_goals(self, mask: torch.Tensor) -> None:
        env_ids = mask.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        goal_r_min = max(self.goal_cfg.radius_min, self.ring_radius + 0.8)
        goal_r_max = max(goal_r_min + 0.1, self.goal_cfg.radius_max)
        gap_jitter = self._gap_half_angle * 0.6
        theta = self._gap_centers[env_ids] + (2.0 * torch.rand((env_ids.numel(),), device=self.device) - 1.0) * gap_jitter
        radius = goal_r_min + (goal_r_max - goal_r_min) * torch.rand((env_ids.numel(),), device=self.device)
        offsets = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), dim=-1)
        self.goal_xy[env_ids] = self._env_origins[env_ids, :2] + offsets

    def _spawn_random_obstacles_once(self) -> None:
        stage = omni.usd.get_context().get_stage()
        for env_id in range(self.num_envs):
            root_prim = f"/World/envs/env_{env_id}/RandomObstacles"
            if not stage.GetPrimAtPath(root_prim).IsValid():
                prim_utils.create_prim(root_prim, "Xform")

            center_x = float(self._env_origins[env_id, 0].item())
            center_y = float(self._env_origins[env_id, 1].item())
            gap_center = random.uniform(0.0, 2.0 * math.pi)
            self._gap_centers[env_id] = gap_center
            gap_angle = self._gap_half_angle * 2.0
            arc_len = max(1e-6, 2.0 * math.pi - gap_angle)

            for i in range(self.ring_obstacle_count):
                size = random.uniform(*self.obstacle_size_range)
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
        print(
            f"[INFO] High-level input uses RTX lidar profile={args_cli.rtx_lidar_profile}, bins={self.lidar_bins}"
        )

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
        self._cached_lidar_scan = scans
        return scans

    def _build_high_obs(self, obs_td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        policy_obs = obs_td["policy"]
        if self.sensor_mode == "rtx_lidar":
            lidar_scan = self._read_rtx_lidar_scan()
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

    def get_observations(self):
        return self._high_obs_td

    def reset(self):
        self._obs_td = self.base_env.get_observations()
        self._resample_goals(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self._high_obs, self.prev_dist = self._build_high_obs(self._obs_td)
        self._high_obs_td = TensorDict({"policy": self._high_obs}, batch_size=[self.num_envs], device=self.device)
        self.prev_cmd.zero_()
        return self._high_obs_td

    def step(self, actions: torch.Tensor):
        cmd = torch.zeros_like(actions)
        cmd[:, 0] = torch.clamp(actions[:, 0], -1.2, 1.2)   # vx
        cmd[:, 1] = torch.clamp(actions[:, 1], -0.8, 0.8)   # vy
        cmd[:, 2] = torch.clamp(actions[:, 2], -1.8, 1.8)   # wz

        # Apply high-level command to existing command channel.
        for i in range(self.num_envs):
            custom_rl_env.base_command[str(i)] = [float(cmd[i, 0]), float(cmd[i, 1]), float(cmd[i, 2])]

        # Frozen low-level gait controller generates joint actions.
        with torch.inference_mode():
            low_actions = self.low_level_policy(self._obs_td)

        self._obs_td, _, low_dones, extras = self.base_env.step(low_actions)
        self._high_obs, dist = self._build_high_obs(self._obs_td)
        self._high_obs_td = TensorDict({"policy": self._high_obs}, batch_size=[self.num_envs], device=self.device)

        progress = self.prev_dist - dist
        scan_min = self._high_obs[:, : self.lidar_bins].min(dim=1).values
        obstacle_penalty = torch.relu(0.6 - scan_min)
        smooth_penalty = torch.sum((cmd - self.prev_cmd) ** 2, dim=-1)

        success = dist < self.goal_cfg.threshold
        done = (low_dones.to(dtype=torch.bool) | success)

        reward = (
            2.0 * progress
            - 0.7 * obstacle_penalty
            - 0.05 * smooth_penalty
            - 0.01
            + success.to(dtype=torch.float32) * 3.0
            - low_dones.to(dtype=torch.float32) * 3.0
            - (scan_min < 0.25).to(dtype=torch.float32) * 1.5
        )

        self._resample_goals(done)
        self.prev_dist = dist
        self.prev_cmd = cmd
        return self._high_obs_td, reward, done.to(dtype=torch.long), extras

    def close(self):
        return self.base_env.close()


def main():
    env_cfg = UnitreeGo2CustomEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Configure command buffers used by high-level wrapper.
    for i in range(env_cfg.scene.num_envs):
        custom_rl_env.base_command[str(i)] = [0.0, 0.0, 0.0]

    print(f"[INFO] Creating base env: task={args_cli.task}, num_envs={env_cfg.scene.num_envs}, terrain={args_cli.terrain}")
    print(f"[INFO] High-level sensor mode: {args_cli.sensor_mode}")
    print(f"[INFO] Target observation dim: {args_cli.lidar_bins + 3} (lidar_bins={args_cli.lidar_bins})")
    base_env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = RslRlVecEnvWrapper(base_env)
    _patch_env_observation_api(base_env)

    # --- Load frozen low-level gait policy ---
    low_cfg = _adapt_legacy_agent_cfg(unitree_go2_agent_cfg)
    low_runner = OnPolicyRunner(base_env, low_cfg, log_dir=None, device=low_cfg["device"])
    print(f"[INFO] Loading low-level checkpoint: {args_cli.low_level_checkpoint}")
    _load_legacy_checkpoint_if_needed(low_runner, args_cli.low_level_checkpoint)
    low_policy = low_runner.get_inference_policy(device=base_env.device)

    # --- Build high-level avoidance env ---
    hl_env = HighLevelAvoidanceVecEnv(
        base_env=base_env,
        low_level_policy=low_policy,
        goal_cfg=GoalConfig(
            radius_min=args_cli.goal_radius_min,
            radius_max=args_cli.goal_radius_max,
            threshold=args_cli.goal_threshold,
        ),
    )

    # --- High-level PPO config ---
    high_cfg_legacy = {
        "seed": args_cli.seed,
        "device": "cpu" if args_cli.cpu else "cuda",
        "num_steps_per_env": 24,
        "max_iterations": args_cli.max_iterations,
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.5,
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 5e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "save_interval": 50,
        "experiment_name": "go2_avoid_obstacles_hl",
        "run_name": "",
        "logger": "tensorboard",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
    }
    high_cfg = _adapt_legacy_agent_cfg(high_cfg_legacy)

    log_root = os.path.join(WORK_ROOT, "logs", "rsl_rl", high_cfg["experiment_name"])
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] High-level training logs: {log_dir}")

    runner = OnPolicyRunner(hl_env, high_cfg, log_dir=log_dir, device=high_cfg["device"])
    print("[INFO] Start high-level obstacle avoidance training...")
    t0 = time.time()
    runner.learn(num_learning_iterations=high_cfg["max_iterations"], init_at_random_ep_len=True)
    print(f"[INFO] Training finished in {time.time() - t0:.1f}s")

    hl_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
