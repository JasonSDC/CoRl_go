#!/usr/bin/env python3
"""Play mode with no policy and forced zero XYZ base velocity."""

from __future__ import annotations

import argparse
import os
import sys
import types

import numpy as np
import torch
from omni.isaac.orbit.app import AppLauncher


parser = argparse.ArgumentParser(description="Run Go2 with no policy and zero base velocity.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "rough"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--print_interval", type=int, default=100)
parser.add_argument("--rtx_lidar_profile", type=str, default="Unitree_L1_PM_approx")
parser.add_argument("--lidar_bins", type=int, default=24)
parser.add_argument("--lidar_range_max", type=float, default=20.0)
parser.add_argument("--print_lidar_interval", type=int, default=50)
parser.add_argument("--front_obstacle_distance", type=float, default=1.0)
parser.add_argument("--front_obstacle_size", type=float, default=0.35)
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

torch.manual_seed(args_cli.seed)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

WORK_ROOT = "/home/dacheng.shen/Desktop/omniverse/avoid_obstacles"
PLAY_DIR = os.path.join(WORK_ROOT, "play")
if PLAY_DIR not in sys.path:
    sys.path.insert(0, PLAY_DIR)

# custom_rl_env expects `from omniverse_sim import args_cli`.
sys.modules["omniverse_sim"] = types.SimpleNamespace(args_cli=args_cli)

import gymnasium as gym
import omni
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.usd

import custom_rl_env
from custom_rl_env import UnitreeGo2CustomEnvCfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper


def _enforce_zero_xyz_velocity(env: RslRlVecEnvWrapper) -> None:
    robot = env.env.scene["robot"]
    root_state = robot.data.root_state_w.clone()
    root_state[:, 7:10] = 0.0
    root_state[:, 10:13] = 0.0
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if hasattr(robot, "write_root_velocity_to_sim"):
        robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
    elif hasattr(robot, "write_root_state_to_sim"):
        robot.write_root_state_to_sim(root_state, env_ids=env_ids)


def _spawn_front_obstacle(env: RslRlVecEnvWrapper) -> None:
    stage = omni.usd.get_context().get_stage()
    env_origins = env.env.scene.env_origins
    size = float(args_cli.front_obstacle_size)
    dist = float(args_cli.front_obstacle_distance)
    for env_id in range(env.num_envs):
        root_prim = f"/World/envs/env_{env_id}/FrontObstacle"
        if not stage.GetPrimAtPath(root_prim).IsValid():
            prim_utils.create_prim(root_prim, "Xform")
        prim_path = f"{root_prim}/box"
        if stage.GetPrimAtPath(prim_path).IsValid():
            continue
        ox = float(env_origins[env_id, 0].item() + dist)
        oy = float(env_origins[env_id, 1].item())
        oz = size * 0.5
        cube_cfg = sim_utils.CuboidCfg(
            size=(size, size, size),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.25, 0.25), roughness=0.7),
        )
        cube_cfg.func(prim_path, cube_cfg, translation=(ox, oy, oz))


def _setup_rtx_lidar(env: RslRlVecEnvWrapper):
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("omni.isaac.sensor", True)
    ext_mgr.set_extension_enabled_immediate("omni.sensors.nv.lidar", True)
    from omni.isaac.sensor import LidarRtx

    sensors = []
    for env_id in range(env.num_envs):
        sensor = LidarRtx(
            f"/World/envs/env_{env_id}/Robot/base/lidar_sensor_zero",
            rotation_frequency=20,
            pulse_time=1,
            translation=(0.0, 0.0, 0.4),
            orientation=(1.0, 0.0, 0.0, 0.0),
            config_file_name=args_cli.rtx_lidar_profile,
        )
        sensor.add_point_cloud_data_to_frame()
        sensor.initialize()
        sensors.append(sensor)
    print(f"[INFO] RTX lidar ready: profile={args_cli.rtx_lidar_profile}, bins={args_cli.lidar_bins}")
    return sensors


def _read_lidar_bins(sensor, bins: int, range_max: float) -> np.ndarray:
    frame = sensor.get_current_frame()
    points = frame.get("point_cloud_data", None)
    binned = np.full((bins,), range_max, dtype=np.float32)
    if points is None:
        return binned
    pts = np.asarray(points)
    if pts.size == 0 or pts.shape[-1] < 3:
        return binned
    x = pts[:, 0]
    y = pts[:, 1]
    ranges = np.sqrt(x * x + y * y)
    angles = np.arctan2(y, x)
    valid = np.isfinite(ranges) & np.isfinite(angles) & (ranges > 0.05) & (ranges < range_max)
    if not np.any(valid):
        return binned
    edges = np.linspace(-np.pi, np.pi, bins + 1, dtype=np.float32)
    idx = np.digitize(angles[valid], edges) - 1
    idx = np.clip(idx, 0, bins - 1)
    np.minimum.at(binned, idx, ranges[valid])
    return binned


def main() -> None:
    env_cfg = UnitreeGo2CustomEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    for i in range(env_cfg.scene.num_envs):
        custom_rl_env.base_command[str(i)] = [0.0, 0.0, 0.0]

    print(f"[INFO] Creating env: task={args_cli.task}, num_envs={env_cfg.scene.num_envs}, terrain={args_cli.terrain}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    _spawn_front_obstacle(env)
    _ = env.reset()
    _enforce_zero_xyz_velocity(env)
    lidar_sensors = _setup_rtx_lidar(env)
    print("[INFO] No policy loaded. Base linear velocity is forced to zero every step.")
    print(
        f"[INFO] Spawned front obstacle: distance={args_cli.front_obstacle_distance:.2f}m, "
        f"size={args_cli.front_obstacle_size:.2f}m"
    )

    zero_actions = torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=torch.float32)
    step_idx = 0
    while simulation_app.is_running():
        _ = env.step(zero_actions)
        _enforce_zero_xyz_velocity(env)
        step_idx += 1
        if args_cli.print_interval > 0 and (step_idx % args_cli.print_interval == 0):
            vel = env.env.scene["robot"].data.root_lin_vel_w[0].detach().float().cpu()
            print(f"[INFO] step={step_idx} root_lin_vel_w=({vel[0]:+.4f}, {vel[1]:+.4f}, {vel[2]:+.4f})")
        if args_cli.print_lidar_interval > 0 and (step_idx % args_cli.print_lidar_interval == 0):
            scan = _read_lidar_bins(lidar_sensors[0], args_cli.lidar_bins, args_cli.lidar_range_max)
            text = ", ".join([f"bin{i}:{scan[i]:.2f}m" for i in range(args_cli.lidar_bins)])
            print(f"[INFO] lidar env_0 step={step_idx}: {text}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
