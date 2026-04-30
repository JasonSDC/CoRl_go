#!/usr/bin/env python3
"""Render one RTX lidar frame: hit rays + red hit markers."""

from __future__ import annotations

import argparse
import locale
import os
import sys
import time
import types

import numpy as np
import torch
from omni.isaac.orbit.app import AppLauncher


parser = argparse.ArgumentParser(description="Render one-frame RTX lidar hits and rays.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "rough"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--rtx_lidar_profile", type=str, default="Unitree_L1_PM_approx")
parser.add_argument("--lidar_range_max", type=float, default=20.0)
parser.add_argument("--lidar_range_min", type=float, default=0.05)
parser.add_argument("--warmup_steps", type=int, default=120)
parser.add_argument("--front_obstacle_distance", type=float, default=1.0)
parser.add_argument("--front_obstacle_size", type=float, default=0.35)
parser.add_argument("--ray_thickness", type=float, default=1.5)
parser.add_argument("--point_marker_size", type=float, default=0.05)
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _force_utf8_locale() -> None:
    # omni.isaac.sensor reads JSON with default text encoding during extension startup.
    # On shells with C/ASCII locale this crashes on built-in profile comments (e.g., "°", "±").
    os.environ.setdefault("LANG", "C.UTF-8")
    os.environ.setdefault("LC_ALL", "C.UTF-8")
    try:
        locale.setlocale(locale.LC_ALL, "")
    except Exception:
        pass


_force_utf8_locale()
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
            f"/World/envs/env_{env_id}/Robot/base/lidar_sensor_single_frame",
            rotation_frequency=20,
            pulse_time=1,
            translation=(0.0, 0.0, 0.4),
            orientation=(1.0, 0.0, 0.0, 0.0),
            config_file_name=args_cli.rtx_lidar_profile,
        )
        sensor.add_point_cloud_data_to_frame()
        sensor.initialize()
        sensors.append(sensor)
    return sensors


def _quat_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _read_one_scan_points_local(sensor) -> np.ndarray:
    frame = sensor.get_current_frame()
    points = frame.get("point_cloud_data", None)
    if points is None:
        return np.empty((0, 3), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    return pts[:, :3]


def _draw_single_frame_hits_and_rays(env: RslRlVecEnvWrapper, points_local: np.ndarray) -> int:
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("omni.isaac.debug_draw", True)
    import omni.isaac.debug_draw._debug_draw as omni_debug_draw

    draw = omni_debug_draw.acquire_debug_draw_interface()

    root_state = env.env.scene["robot"].data.root_state_w[0].detach().float().cpu()
    base_pos = root_state[:3]
    yaw = _quat_to_yaw(root_state[3:7])
    c = float(torch.cos(yaw).item())
    s = float(torch.sin(yaw).item())
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    sensor_origin = np.array([base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.4], dtype=np.float32)
    pts_world = (points_local @ rot_z.T) + sensor_origin

    ranges = np.linalg.norm(points_local[:, :2], axis=1)
    valid = np.isfinite(pts_world).all(axis=1)
    valid &= np.isfinite(ranges)
    valid &= ranges > float(args_cli.lidar_range_min)
    valid &= ranges < float(args_cli.lidar_range_max)
    hit_points = pts_world[valid]

    draw.clear_lines()
    if hit_points.shape[0] == 0:
        return 0

    starts = np.repeat(sensor_origin[None, :], hit_points.shape[0], axis=0)
    ray_colors = [[1.0, 0.25, 0.25, 0.95]] * hit_points.shape[0]
    ray_sizes = [float(args_cli.ray_thickness)] * hit_points.shape[0]
    draw.draw_lines(starts.tolist(), hit_points.tolist(), ray_colors, ray_sizes)

    marker_half = 0.5 * float(args_cli.point_marker_size)
    marker_starts = hit_points.copy()
    marker_ends = hit_points.copy()
    marker_starts[:, 2] -= marker_half
    marker_ends[:, 2] += marker_half
    point_colors = [[1.0, 0.0, 0.0, 1.0]] * hit_points.shape[0]
    point_sizes = [max(2.0, float(args_cli.ray_thickness) + 0.5)] * hit_points.shape[0]
    draw.draw_lines(marker_starts.tolist(), marker_ends.tolist(), point_colors, point_sizes)
    return int(hit_points.shape[0])


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
    sensor = lidar_sensors[0]
    print(f"[INFO] RTX lidar ready: profile={args_cli.rtx_lidar_profile}")

    zero_actions = torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=torch.float32)
    points_local = np.empty((0, 3), dtype=np.float32)
    for step_idx in range(1, int(args_cli.warmup_steps) + 1):
        _ = env.step(zero_actions)
        _enforce_zero_xyz_velocity(env)
        pts = _read_one_scan_points_local(sensor)
        if pts.shape[0] > 0:
            points_local = pts
            print(f"[INFO] Got lidar frame at step={step_idx}, raw_points={pts.shape[0]}")
            break
    if points_local.shape[0] == 0:
        print("[WARN] No lidar points captured in warmup window; rendering empty frame.")

    hit_count = _draw_single_frame_hits_and_rays(env, points_local)
    print(f"[INFO] Rendered one lidar frame. hit_count={hit_count}.")
    print("[INFO] Frozen single-frame visualization. Close window to exit.")

    while simulation_app.is_running():
        if hasattr(simulation_app, "update"):
            simulation_app.update()
        else:
            time.sleep(1.0 / 60.0)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
