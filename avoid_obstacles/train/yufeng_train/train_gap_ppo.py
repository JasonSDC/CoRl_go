#!/usr/bin/env python3
"""Train high-level PPO for ring-gap navigation with frozen low-level gait policy.

Task:
- Robot starts inside a ring of obstacles with one random gap.
- Goal is placed 1m outside the gap direction.
- High-level action is (vx, vy, wz), sent through custom_rl_env.base_command.

Reward:
- collision: -1.0
- step: -0.01
- reach goal: +10.0
- progress bonus: only when distance to goal decreases (anti-oscillation reward hacking).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
import threading
import time
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import omni
import torch
from tensordict import TensorDict

from omni.isaac.orbit.app import AppLauncher


parser = argparse.ArgumentParser(description="Train PPO high-level policy for random-gap ring navigation.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=1048)
parser.add_argument("--terrain", type=str, default="rough", choices=["flat", "rough"])
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--experiment_name", type=str, default="go2_gap_random_hole_ppo")
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "neptune"])
parser.add_argument("--save_interval", type=int, default=50, help="Save checkpoint every N PPO iterations.")

parser.add_argument("--sensor_mode", type=str, default="rtx_lidar", choices=["rtx_lidar", "height_scan"])
parser.add_argument("--rtx_lidar_profile", type=str, default="Unitree_L1_PM_approx")
parser.add_argument("--lidar_bins", type=int, default=72)
parser.add_argument("--lidar_range_max", type=float, default=20.0)
parser.add_argument("--lidar_range_min", type=float, default=0.05)
parser.add_argument("--self_filter_radius", type=float, default=0.05)
parser.add_argument(
    "--horizontal_elevation_deg",
    type=float,
    default=3.0,
    help="Keep near-horizontal lidar echoes only (abs(elevation) <= this value).",
)
parser.add_argument(
    "--high_level_interval_steps",
    type=int,
    default=2,
    help="How many control steps per high-level inference (2 -> 0.04s if control is 0.02s).",
)

parser.add_argument("--ring_obstacle_count", type=int, default=20)
parser.add_argument("--ring_radius", type=float, default=2.2)
parser.add_argument("--obstacle_size_min", type=float, default=0.25)
parser.add_argument("--obstacle_size_max", type=float, default=0.7)
parser.add_argument("--obstacle_height", type=float, default=1.0)
parser.add_argument("--obstacle_gap_width", type=float, default=0.93)
parser.add_argument("--goal_outside_distance", type=float, default=1.0)
parser.add_argument("--goal_threshold", type=float, default=0.6)

parser.add_argument("--reward_collision", type=float, default=-1.0)
parser.add_argument("--reward_step", type=float, default=-0.01)
parser.add_argument("--reward_goal", type=float, default=10.0)
parser.add_argument("--reward_progress_scale", type=float, default=1.2)
parser.add_argument(
    "--progress_clip_per_step",
    type=float,
    default=0.25,
    help="Clip positive progress reward per step to avoid reward spikes.",
)
parser.add_argument(
    "--collision_distance",
    type=float,
    default=0.22,
    help="Lidar min-range threshold to count as obstacle collision.",
)

parser.add_argument(
    "--low_level_checkpoint",
    type=str,
    default="/home/dacheng.shen/Desktop/omniverse/avoid_obstacles/play/logs/rsl_rl/unitree_go2_rough/2024-04-06_02-37-07/model_7850.pt",
)

parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Training default: run headless (no rendering window).
if hasattr(args_cli, "headless"):
    args_cli.headless = True

torch.manual_seed(args_cli.seed)
random.seed(args_cli.seed)
np.random.seed(args_cli.seed)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

WORK_ROOT = "/home/dacheng.shen/Desktop/omniverse/avoid_obstacles"
PLAY_DIR = os.path.join(WORK_ROOT, "play")
CONFIG_DIR = os.path.join(WORK_ROOT, "configs")
if PLAY_DIR not in sys.path:
    sys.path.insert(0, PLAY_DIR)
if CONFIG_DIR not in sys.path:
    sys.path.insert(0, CONFIG_DIR)

# custom_rl_env expects this module path.
sys.modules["omniverse_sim"] = types.SimpleNamespace(args_cli=args_cli)

import gymnasium as gym
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

import custom_rl_env
from agent_cfg import unitree_go2_agent_cfg
from custom_rl_env import UnitreeGo2CustomEnvCfg
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.usd
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from pxr import Gf


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
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": init_noise_std},
        }
        adapted["critic"] = {"class_name": "MLPModel", "hidden_dims": critic_hidden_dims, "activation": activation}

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


def _format_progress_bar(current: int, total: int, width: int = 32) -> str:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    ratio = float(current) / float(total)
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {100.0 * ratio:6.2f}% ({current}/{total})"


def _start_checkpoint_progress_monitor(log_dir: str, total_iters: int):
    stop_evt = threading.Event()
    latest = {"iter": 0}

    def _worker():
        ckpt_re = re.compile(r"^model_(\d+)\.pt$")
        log_path = Path(log_dir)
        last_print_iter = -1
        last_heartbeat_t = 0.0
        while not stop_evt.is_set():
            best = 0
            if log_path.exists():
                for p in log_path.glob("model_*.pt"):
                    m = ckpt_re.match(p.name)
                    if not m:
                        continue
                    best = max(best, int(m.group(1)))
            latest["iter"] = min(best, int(total_iters))
            now_t = time.time()
            if latest["iter"] != last_print_iter or (now_t - last_heartbeat_t) >= 15.0:
                print(f"[PROGRESS] {_format_progress_bar(latest['iter'], total_iters)}")
                last_print_iter = latest["iter"]
                last_heartbeat_t = now_t
            stop_evt.wait(1.0)

    th = threading.Thread(target=_worker, name="ckpt-progress-monitor", daemon=True)
    th.start()
    return stop_evt, th, latest


@dataclass
class GoalConfig:
    threshold: float = 0.6
    outside_distance: float = 1.0


class RingGapHighLevelVecEnv(VecEnv):
    def __init__(self, base_env: RslRlVecEnvWrapper, low_level_policy, goal_cfg: GoalConfig):
        self.base_env = base_env
        self.low_level_policy = low_level_policy
        self.goal_cfg = goal_cfg

        self.num_envs = base_env.num_envs
        self.device = base_env.device
        self.max_episode_length = base_env.max_episode_length
        self.num_actions = 3
        # High-level obs: lidar bins + goal relative position in body frame (x, y).
        self.num_obs = int(args_cli.lidar_bins) + 2
        self.num_privileged_obs = 0

        self.sensor_mode = args_cli.sensor_mode
        self.lidar_bins = int(args_cli.lidar_bins)
        self.lidar_range_max = float(args_cli.lidar_range_max)
        self.high_level_interval_steps = max(1, int(args_cli.high_level_interval_steps))
        self.ring_obstacle_count = max(4, int(args_cli.ring_obstacle_count))
        self.ring_radius = max(0.8, float(args_cli.ring_radius))
        self.obstacle_gap_width = max(0.1, float(args_cli.obstacle_gap_width))
        self.obstacle_height = float(args_cli.obstacle_height)
        self.obstacle_size_range = (float(args_cli.obstacle_size_min), float(args_cli.obstacle_size_max))
        self._gap_half_angle = min(math.pi * 0.8, (self.obstacle_gap_width / self.ring_radius) * 0.5)
        self._gap_centers = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._env_origins = self.base_env.env.scene.env_origins

        self.goal_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.prev_dist = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._cached_lidar_scan = torch.full(
            (self.num_envs, self.lidar_bins), self.lidar_range_max, device=self.device, dtype=torch.float32
        )
        self._rtx_lidar_sensors = []
        self._obs_sizes = [
            [random.uniform(*self.obstacle_size_range) for _ in range(self.ring_obstacle_count)]
            for _ in range(self.num_envs)
        ]

        self._create_layout_prims_once()
        self._refresh_layout(torch.ones((self.num_envs,), dtype=torch.bool, device=self.device))
        self._obs_td = self.base_env.get_observations()
        self._setup_rtx_lidar_if_needed()
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

    def _set_prim_translation(self, prim_path: str, x: float, y: float, z: float) -> None:
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        attr = prim.GetAttribute("xformOp:translate")
        if not attr.IsValid():
            attr = prim.CreateAttribute("xformOp:translate", "double3")
        attr.Set(Gf.Vec3d(float(x), float(y), float(z)))

    def _create_layout_prims_once(self) -> None:
        stage = omni.usd.get_context().get_stage()
        for env_id in range(self.num_envs):
            root_prim = f"/World/envs/env_{env_id}/TrainRandomGapLayout"
            if not stage.GetPrimAtPath(root_prim).IsValid():
                prim_utils.create_prim(root_prim, "Xform")

            for i in range(self.ring_obstacle_count):
                size = self._obs_sizes[env_id][i]
                prim_path = f"{root_prim}/obs_{i}"
                if stage.GetPrimAtPath(prim_path).IsValid():
                    continue
                cube_cfg = sim_utils.CuboidCfg(
                    size=(size, size, self.obstacle_height),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2), roughness=0.7),
                )
                cube_cfg.func(prim_path, cube_cfg, translation=(0.0, 0.0, -10.0))

            goal_path = f"{root_prim}/goal"
            if not stage.GetPrimAtPath(goal_path).IsValid():
                goal_cfg = sim_utils.CuboidCfg(
                    size=(0.25, 0.25, 0.25),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2), roughness=0.5),
                )
                goal_cfg.func(goal_path, goal_cfg, translation=(0.0, 0.0, 0.125))

    def _refresh_layout(self, mask: torch.Tensor) -> None:
        env_ids = mask.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        gap_angle = 2.0 * self._gap_half_angle
        arc_len = max(1e-6, 2.0 * math.pi - gap_angle)
        goal_radius = self.ring_radius + float(self.goal_cfg.outside_distance)

        for env_id_t in env_ids:
            env_id = int(env_id_t.item())
            cx = float(self._env_origins[env_id, 0].item())
            cy = float(self._env_origins[env_id, 1].item())
            gap_center = random.uniform(-math.pi, math.pi)
            self._gap_centers[env_id] = gap_center

            root_prim = f"/World/envs/env_{env_id}/TrainRandomGapLayout"
            for i in range(self.ring_obstacle_count):
                frac = (i + 0.5) / float(self.ring_obstacle_count)
                theta = gap_center + self._gap_half_angle + frac * arc_len
                ox = cx + self.ring_radius * math.cos(theta)
                oy = cy + self.ring_radius * math.sin(theta)
                oz = 0.5 * self.obstacle_height
                self._set_prim_translation(f"{root_prim}/obs_{i}", ox, oy, oz)

            gx = cx + goal_radius * math.cos(gap_center)
            gy = cy + goal_radius * math.sin(gap_center)
            self.goal_xy[env_id, 0] = gx
            self.goal_xy[env_id, 1] = gy
            self._set_prim_translation(f"{root_prim}/goal", gx, gy, 0.125)

    def _setup_rtx_lidar_if_needed(self) -> None:
        if self.sensor_mode != "rtx_lidar":
            return
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_mgr.set_extension_enabled_immediate("omni.isaac.sensor", True)
        ext_mgr.set_extension_enabled_immediate("omni.sensors.nv.lidar", True)
        from omni.isaac.sensor import LidarRtx

        for env_id in range(self.num_envs):
            sensor = LidarRtx(
                f"/World/envs/env_{env_id}/Robot/base/lidar_sensor_train_gap",
                rotation_frequency=20,
                pulse_time=1,
                translation=(0.0, 0.0, 0.1),
                orientation=(1.0, 0.0, 0.0, 0.0),
                config_file_name=args_cli.rtx_lidar_profile,
            )
            sensor.add_point_cloud_data_to_frame()
            sensor.initialize()
            self._rtx_lidar_sensors.append(sensor)
        print(f"[INFO] High-level input uses RTX lidar profile={args_cli.rtx_lidar_profile}, bins={self.lidar_bins}")

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
            z = pts[:, 2] if pts.shape[1] >= 3 else np.zeros_like(x)
            ranges = np.sqrt(x * x + y * y)
            angles = np.arctan2(y, x)
            elev_deg = np.degrees(np.arctan2(z, np.maximum(ranges, 1e-6)))
            valid = np.isfinite(ranges) & np.isfinite(angles) & np.isfinite(elev_deg)
            valid &= np.abs(elev_deg) <= float(args_cli.horizontal_elevation_deg)
            valid &= ranges > max(float(args_cli.lidar_range_min), float(args_cli.self_filter_radius))
            valid &= ranges < self.lidar_range_max
            binned = np.full((self.lidar_bins,), self.lidar_range_max, dtype=np.float32)
            if np.any(valid):
                idx = np.digitize(angles[valid], edges) - 1
                idx = np.clip(idx, 0, self.lidar_bins - 1)
                np.minimum.at(binned, idx, ranges[valid])
            scans[env_id] = torch.from_numpy(binned).to(device=self.device)
        self._cached_lidar_scan = scans
        return scans

    def _build_high_obs(self, obs_td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        lidar_scan = self._extract_lidar_scan(obs_td)
        return self._build_high_obs_from_scan(obs_td, lidar_scan)

    def _extract_lidar_scan(self, obs_td: TensorDict) -> torch.Tensor:
        policy_obs = obs_td["policy"]
        if self.sensor_mode == "rtx_lidar":
            return self._read_rtx_lidar_scan()
        height_scan = policy_obs[:, -187:]
        if self.lidar_bins == 187:
            return height_scan
        return torch.nn.functional.interpolate(
            height_scan.unsqueeze(1), size=self.lidar_bins, mode="linear", align_corners=False
        ).squeeze(1)

    def _build_high_obs_from_scan(self, obs_td: TensorDict, lidar_scan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        root = self.base_env.env.scene["robot"].data.root_state_w
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

        goal_xy_body = torch.stack((rel_body_x, rel_body_y), dim=-1)
        high_obs = torch.cat((lidar_scan, goal_xy_body), dim=-1)
        return high_obs, dist

    def get_observations(self):
        return self._high_obs_td

    def reset(self):
        self._refresh_layout(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self._obs_td = self.base_env.get_observations()
        self._high_obs, self.prev_dist = self._build_high_obs(self._obs_td)
        self._high_obs_td = TensorDict({"policy": self._high_obs}, batch_size=[self.num_envs], device=self.device)
        return self._high_obs_td

    def step(self, actions: torch.Tensor):
        cmd = torch.zeros_like(actions)
        cmd[:, 0] = torch.clamp(actions[:, 0], -1.2, 1.2)  # vx
        cmd[:, 1] = torch.clamp(actions[:, 1], -0.8, 0.8)  # vy
        cmd[:, 2] = torch.clamp(actions[:, 2], -1.8, 1.8)  # wz

        for i in range(self.num_envs):
            custom_rl_env.base_command[str(i)] = [float(cmd[i, 0]), float(cmd[i, 1]), float(cmd[i, 2])]

        done_acc = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        extras = {}
        lidar_scan_agg: torch.Tensor | None = None
        for _ in range(self.high_level_interval_steps):
            with torch.inference_mode():
                low_actions = self.low_level_policy(self._obs_td)
            self._obs_td, _, low_dones, extras = self.base_env.step(low_actions)
            done_acc |= low_dones.to(dtype=torch.bool)
            scan_now = self._extract_lidar_scan(self._obs_td)
            if lidar_scan_agg is None:
                lidar_scan_agg = scan_now
            else:
                lidar_scan_agg = torch.minimum(lidar_scan_agg, scan_now)

        if lidar_scan_agg is None:
            lidar_scan_agg = self._extract_lidar_scan(self._obs_td)
        self._high_obs, dist = self._build_high_obs_from_scan(self._obs_td, lidar_scan_agg)
        self._high_obs_td = TensorDict({"policy": self._high_obs}, batch_size=[self.num_envs], device=self.device)

        scan_min = lidar_scan_agg.min(dim=1).values
        success = dist < float(args_cli.goal_threshold)
        collision = scan_min < float(args_cli.collision_distance)
        timeout = done_acc
        done = success | collision | timeout

        # Anti-reward-hacking: reward only positive progress (getting closer), never oscillation-away bonus.
        progress = torch.clamp(self.prev_dist - dist, min=0.0, max=float(args_cli.progress_clip_per_step))
        reward = (
            torch.full_like(dist, float(args_cli.reward_step) * float(self.high_level_interval_steps))
            + float(args_cli.reward_progress_scale) * progress
            + success.to(dtype=torch.float32) * float(args_cli.reward_goal)
            + collision.to(dtype=torch.float32) * float(args_cli.reward_collision)
        )

        if torch.any(done):
            self._refresh_layout(done)

        self.prev_dist = dist
        return self._high_obs_td, reward, done.to(dtype=torch.long), extras

    def close(self):
        return self.base_env.close()


def main():
    env_cfg = UnitreeGo2CustomEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    for i in range(env_cfg.scene.num_envs):
        custom_rl_env.base_command[str(i)] = [0.0, 0.0, 0.0]

    print(f"[INFO] Creating base env: task={args_cli.task}, num_envs={env_cfg.scene.num_envs}, terrain={args_cli.terrain}")
    print("[INFO] Reward: collision=-1, step=-0.01, goal=+10, positive-progress bonus enabled.")
    print(
        f"[INFO] Ring gap task: ring_radius={args_cli.ring_radius:.2f}m, gap_width={args_cli.obstacle_gap_width:.2f}m, "
        f"goal_outside={args_cli.goal_outside_distance:.2f}m"
    )
    print(
        f"[INFO] Lidar config: bins={args_cli.lidar_bins}, range=({args_cli.lidar_range_min:.2f},{args_cli.lidar_range_max:.2f})m, "
        f"self_filter_radius={args_cli.self_filter_radius:.2f}m, horizontal_elevation={args_cli.horizontal_elevation_deg:.1f}deg, "
        "sensor_z=0.1m"
    )
    print(
        f"[INFO] High-level obs dim={int(args_cli.lidar_bins) + 2} "
        f"(lidar={args_cli.lidar_bins} + goal_xy=2)"
    )
    print(
        f"[INFO] High-level inference interval: {int(args_cli.high_level_interval_steps)} control steps "
        "(LiDAR min-aggregated over this interval)."
    )

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = RslRlVecEnvWrapper(base_env)
    _patch_env_observation_api(base_env)

    low_cfg = _adapt_legacy_agent_cfg(unitree_go2_agent_cfg)
    low_runner = OnPolicyRunner(base_env, low_cfg, log_dir=None, device=low_cfg["device"])
    print(f"[INFO] Loading low-level checkpoint: {args_cli.low_level_checkpoint}")
    _load_legacy_checkpoint_if_needed(low_runner, args_cli.low_level_checkpoint)
    low_policy = low_runner.get_inference_policy(device=base_env.device)

    hl_env = RingGapHighLevelVecEnv(
        base_env=base_env,
        low_level_policy=low_policy,
        goal_cfg=GoalConfig(threshold=float(args_cli.goal_threshold), outside_distance=float(args_cli.goal_outside_distance)),
    )

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
        "save_interval": int(args_cli.save_interval),
        "experiment_name": str(args_cli.experiment_name),
        "run_name": str(args_cli.run_name),
        "logger": str(args_cli.logger),
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
    }
    high_cfg = _adapt_legacy_agent_cfg(high_cfg_legacy)

    log_root = os.path.join(WORK_ROOT, "logs", "rsl_rl", high_cfg["experiment_name"])
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] Training logs: {log_dir}")
    print(f"[INFO] Checkpoint save interval: every {high_cfg['save_interval']} iterations")

    runner = OnPolicyRunner(hl_env, high_cfg, log_dir=log_dir, device=high_cfg["device"])
    print("[INFO] Start PPO training for random-gap ring task...")
    print(f"[INFO] Progress bar target: {high_cfg['max_iterations']} iterations")
    monitor_stop, monitor_thread, _ = _start_checkpoint_progress_monitor(
        log_dir=log_dir, total_iters=int(high_cfg["max_iterations"])
    )
    t0 = time.time()
    try:
        runner.learn(num_learning_iterations=high_cfg["max_iterations"], init_at_random_ep_len=True)
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=2.0)
    print(f"[PROGRESS] {_format_progress_bar(int(high_cfg['max_iterations']), int(high_cfg['max_iterations']))}")
    final_ckpt_path = os.path.join(log_dir, "model_final.pt")
    try:
        runner.save(final_ckpt_path)
        print(f"[INFO] Final checkpoint saved: {final_ckpt_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to save final checkpoint explicitly: {exc}")
    print(f"[INFO] Training finished in {time.time() - t0:.1f}s")

    for i in range(env_cfg.scene.num_envs):
        custom_rl_env.base_command[str(i)] = [0.0, 0.0, 0.0]
    hl_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
