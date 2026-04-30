#!/usr/bin/env python3
"""Render five raw lidar frames (same logic as single-frame script)."""

from __future__ import annotations

import argparse
import json
import locale
import math
import os
import sys
import time
import types

import numpy as np
import torch
from omni.isaac.orbit.app import AppLauncher


parser = argparse.ArgumentParser(description="Render five raw RTX lidar frames and freeze.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "rough"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--rtx_lidar_profile", type=str, default="Unitree_L1_PM_approx")
parser.add_argument("--lidar_range_max", type=float, default=20.0)
parser.add_argument("--lidar_range_min", type=float, default=0.05)
parser.add_argument("--lidar_bins", type=int, default=72)
parser.add_argument("--obstacle_distance_threshold", type=float, default=5.0)
parser.add_argument("--warmup_steps", type=int, default=120)
parser.add_argument("--capture_seconds", type=float, default=0.04)
parser.add_argument(
    "--capture_sample_mode",
    type=str,
    default="report",
    choices=["report", "control"],
    help="Sampling mode for one-shot capture window.",
)
parser.add_argument(
    "--env_layout",
    type=str,
    default="obstacle_avoidance",
    choices=["obstacle_avoidance", "front_obstacle"],
)
parser.add_argument("--front_obstacle_distance", type=float, default=1.0)
parser.add_argument("--front_obstacle_size", type=float, default=0.35)
parser.add_argument("--front_obstacle_height", type=float, default=1.0)
parser.add_argument("--ring_obstacle_count", type=int, default=20)
parser.add_argument("--ring_radius", type=float, default=2.2)
parser.add_argument("--obstacle_gap_width", type=float, default=0.93)
parser.add_argument("--ring_obstacle_size", type=float, default=0.5)
parser.add_argument("--ring_obstacle_height", type=float, default=1.0)
parser.add_argument("--horizontal_elevation_deg", type=float, default=3.0)
parser.add_argument("--self_filter_radius", type=float, default=0.0)
parser.add_argument("--ray_thickness", type=float, default=1.2)
parser.add_argument("--point_marker_size", type=float, default=0.05)
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _force_utf8_locale() -> None:
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


def _set_robot_fixed_forward_pose(env: RslRlVecEnvWrapper) -> None:
    robot = env.env.scene["robot"]
    env_origins = env.env.scene.env_origins
    root_state = robot.data.root_state_w.clone()
    root_state[:, 0] = env_origins[:, 0]
    root_state[:, 1] = env_origins[:, 1]
    root_state[:, 2] = env_origins[:, 2] + 0.5
    root_state[:, 3] = 1.0  # w
    root_state[:, 4] = 0.0
    root_state[:, 5] = 0.0
    root_state[:, 6] = 0.0
    root_state[:, 7:13] = 0.0
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if hasattr(robot, "write_root_state_to_sim"):
        robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    else:
        if hasattr(robot, "write_root_pose_to_sim"):
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        if hasattr(robot, "write_root_velocity_to_sim"):
            robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)


def _spawn_obstacle_avoidance_layout(env: RslRlVecEnvWrapper) -> None:
    stage = omni.usd.get_context().get_stage()
    env_origins = env.env.scene.env_origins
    ring_count = max(4, int(args_cli.ring_obstacle_count))
    ring_radius = max(0.8, float(args_cli.ring_radius))
    gap_width = max(0.1, float(args_cli.obstacle_gap_width))
    size = float(args_cli.ring_obstacle_size)
    height = float(args_cli.ring_obstacle_height)
    gap_half_angle = min(math.pi * 0.8, (gap_width / ring_radius) * 0.5)
    gap_center = 0.5 * math.pi  # +y opening (robot left side)
    gap_angle = 2.0 * gap_half_angle
    arc_len = max(1e-6, 2.0 * math.pi - gap_angle)

    for env_id in range(env.num_envs):
        root_prim = f"/World/envs/env_{env_id}/RandomObstacles"
        if not stage.GetPrimAtPath(root_prim).IsValid():
            prim_utils.create_prim(root_prim, "Xform")

        cx = float(env_origins[env_id, 0].item())
        cy = float(env_origins[env_id, 1].item())
        for i in range(ring_count):
            frac = (i + 0.5) / float(ring_count)
            theta_local = gap_half_angle + frac * arc_len
            theta = theta_local + gap_center
            x = cx + ring_radius * math.cos(theta)
            y = cy + ring_radius * math.sin(theta)
            z = 0.5 * height
            prim_path = f"{root_prim}/obs_{i}"
            if stage.GetPrimAtPath(prim_path).IsValid():
                continue
            cube_cfg = sim_utils.CuboidCfg(
                size=(size, size, height),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2), roughness=0.7),
            )
            cube_cfg.func(prim_path, cube_cfg, translation=(x, y, z))


def _spawn_front_obstacle(env: RslRlVecEnvWrapper) -> None:
    stage = omni.usd.get_context().get_stage()
    env_origins = env.env.scene.env_origins
    size = float(args_cli.front_obstacle_size)
    height = float(args_cli.front_obstacle_height)
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
        oz = height * 0.5
        cube_cfg = sim_utils.CuboidCfg(
            size=(size, size, height),
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
            f"/World/envs/env_{env_id}/Robot/base/lidar_sensor_multi_frame",
            rotation_frequency=20,
            pulse_time=1,
            translation=(0.0, 0.0, 0.1),
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


def _resolve_lidar_profile_path(profile_name: str) -> str | None:
    if profile_name.endswith(".json") and os.path.isfile(profile_name):
        return profile_name
    candidate_paths = [
        os.path.join(WORK_ROOT, "env", "lidar", f"{profile_name}.json"),
        os.path.join("/home/dacheng.shen/Desktop/omniverse/IsaacLab/source/data/sensors/lidar", f"{profile_name}.json"),
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            return path
    return None


def _load_lidar_rates(profile_name: str) -> tuple[float, float]:
    profile_path = _resolve_lidar_profile_path(profile_name)
    if profile_path is None:
        return 10.0, 900.0
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        profile = data.get("profile", {})
        scan_rate = float(profile.get("scanRateBaseHz", 10.0))
        report_rate = float(profile.get("reportRateBaseHz", 900.0))
        return scan_rate, report_rate
    except Exception:
        return 10.0, 900.0


def _compute_lidar_bins(
    points_local: np.ndarray,
    bins: int,
    range_min: float,
    range_max: float,
    self_filter_radius: float,
) -> np.ndarray:
    out = np.full((bins,), range_max, dtype=np.float32)
    if points_local.shape[0] == 0:
        return out
    x = points_local[:, 0]
    y = points_local[:, 1]
    ranges = np.sqrt(x * x + y * y)
    angles = np.arctan2(y, x)
    valid = np.isfinite(ranges) & np.isfinite(angles) & (ranges > max(range_min, self_filter_radius)) & (ranges < range_max)
    if not np.any(valid):
        return out
    edges = np.linspace(-np.pi, np.pi, bins + 1, dtype=np.float32)
    idx = np.digitize(angles[valid], edges) - 1
    idx = np.clip(idx, 0, bins - 1)
    np.minimum.at(out, idx, ranges[valid])
    return out


def _format_bins_with_degrees(bin_ranges: np.ndarray) -> str:
    bins = int(bin_ranges.shape[0])
    if bins <= 0:
        return ""
    # Use bin center angle in degrees for human-readable per-bin prints.
    deg_centers = np.linspace(-180.0, 180.0, bins, endpoint=False, dtype=np.float32)
    parts = [f"{float(deg_centers[i]):6.1f}deg:{float(bin_ranges[i]):5.2f}m" for i in range(bins)]
    return " | ".join(parts)


def _filter_horizontal_points(points_local: np.ndarray, max_elevation_deg: float) -> np.ndarray:
    if points_local.shape[0] == 0:
        return points_local
    xy_norm = np.linalg.norm(points_local[:, :2], axis=1)
    elev_deg = np.degrees(np.arctan2(points_local[:, 2], np.maximum(xy_norm, 1e-6)))
    return points_local[np.abs(elev_deg) <= float(max_elevation_deg)]


def _draw_frames(env: RslRlVecEnvWrapper, frames_local: list[np.ndarray]) -> tuple[int, int, int]:
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("omni.isaac.debug_draw", True)
    import omni.isaac.debug_draw._debug_draw as omni_debug_draw

    draw = omni_debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()

    root_state = env.env.scene["robot"].data.root_state_w[0].detach().float().cpu()
    base_pos = root_state[:3]
    yaw = _quat_to_yaw(root_state[3:7])
    c = float(torch.cos(yaw).item())
    s = float(torch.sin(yaw).item())
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    sensor_origin = np.array([base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.1], dtype=np.float32)

    # Draw self-filter radius as a yellow circle so ignored near-body area is visible.
    circle_r = max(0.0, float(args_cli.self_filter_radius))
    if circle_r > 1e-6:
        seg_n = 72
        theta = np.linspace(0.0, 2.0 * np.pi, seg_n + 1, dtype=np.float32)
        circle_pts = np.stack(
            [
                sensor_origin[0] + circle_r * np.cos(theta),
                sensor_origin[1] + circle_r * np.sin(theta),
                np.full((seg_n + 1,), sensor_origin[2], dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        c_starts = circle_pts[:-1].tolist()
        c_ends = circle_pts[1:].tolist()
        c_colors = [[1.0, 1.0, 0.0, 0.95]] * seg_n
        c_sizes = [max(1.0, float(args_cli.ray_thickness))] * seg_n
        draw.draw_lines(c_starts, c_ends, c_colors, c_sizes)

    all_local_hits: list[np.ndarray] = []
    frame_hit_counts: list[int] = []

    for pts_local in frames_local:
        if pts_local.shape[0] == 0:
            frame_hit_counts.append(0)
            continue
        # Keep near-horizontal rays only.
        pts_local = _filter_horizontal_points(pts_local, float(args_cli.horizontal_elevation_deg))
        if pts_local.shape[0] == 0:
            frame_hit_counts.append(0)
            continue
        pts_world = (pts_local @ rot_z.T) + sensor_origin
        ranges = np.linalg.norm(pts_local[:, :2], axis=1)
        valid = np.isfinite(pts_world).all(axis=1)
        valid &= np.isfinite(ranges)
        valid &= ranges > float(args_cli.lidar_range_min)
        valid &= ranges < float(args_cli.lidar_range_max)
        valid_local = pts_local[valid]
        hit_points = pts_world[valid]
        frame_hit_counts.append(int(hit_points.shape[0]))
        if hit_points.shape[0] == 0:
            continue
        all_local_hits.append(valid_local)

    # Draw per-bin full coverage rays: hit=red, miss=green.
    if len(all_local_hits) == 0:
        local_points = np.empty((0, 3), dtype=np.float32)
    else:
        local_points = np.concatenate(all_local_hits, axis=0).astype(np.float32, copy=False)
    binned = _compute_lidar_bins(
        local_points,
        bins=int(args_cli.lidar_bins),
        range_min=float(args_cli.lidar_range_min),
        range_max=float(args_cli.lidar_range_max),
        self_filter_radius=float(args_cli.self_filter_radius),
    )
    bins = int(binned.shape[0])
    bin_centers = np.linspace(-np.pi, np.pi, bins, endpoint=False, dtype=np.float32)
    angles = bin_centers + float(yaw.item())
    bin_ends = np.stack(
        [
            sensor_origin[0] + binned * np.cos(angles),
            sensor_origin[1] + binned * np.sin(angles),
            np.full((bins,), sensor_origin[2], dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    bin_starts = np.repeat(sensor_origin[None, :], bins, axis=0).astype(np.float32, copy=False)
    # Obstacle decision threshold: echoes farther than this are treated as non-obstacles.
    obstacle_threshold = min(float(args_cli.obstacle_distance_threshold), float(args_cli.lidar_range_max))
    hit_mask = binned <= obstacle_threshold
    bin_ray_colors = [[1.0, 0.2, 0.2, 0.8] if bool(v) else [0.2, 1.0, 0.2, 0.65] for v in hit_mask.tolist()]
    bin_ray_sizes = [max(0.8, float(args_cli.ray_thickness) * 0.8)] * bins
    draw.draw_lines(bin_starts.tolist(), bin_ends.tolist(), bin_ray_colors, bin_ray_sizes)

    marker_half_bin = 0.5 * float(args_cli.point_marker_size)
    bin_marker_starts = bin_ends.copy()
    bin_marker_ends = bin_ends.copy()
    bin_marker_starts[:, 2] -= marker_half_bin
    bin_marker_ends[:, 2] += marker_half_bin
    bin_point_colors = [[1.0, 0.0, 0.0, 1.0] if bool(v) else [0.0, 1.0, 0.0, 1.0] for v in hit_mask.tolist()]
    bin_point_sizes = [max(1.5, float(args_cli.ray_thickness))] * bins
    draw.draw_lines(bin_marker_starts.tolist(), bin_marker_ends.tolist(), bin_point_colors, bin_point_sizes)

    return int(np.count_nonzero(hit_mask)), int(np.count_nonzero(~hit_mask)), len(frames_local)


def main() -> None:
    env_cfg = UnitreeGo2CustomEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    for i in range(env_cfg.scene.num_envs):
        custom_rl_env.base_command[str(i)] = [0.0, 0.0, 0.0]

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    if args_cli.env_layout == "obstacle_avoidance":
        _spawn_obstacle_avoidance_layout(env)
    else:
        _spawn_front_obstacle(env)
    _ = env.reset()
    _set_robot_fixed_forward_pose(env)
    _enforce_zero_xyz_velocity(env)
    lidar_sensors = _setup_rtx_lidar(env)
    sensor = lidar_sensors[0]
    scan_rate_hz, report_rate_hz = _load_lidar_rates(str(args_cli.rtx_lidar_profile))
    print(f"[INFO] RTX lidar ready: profile={args_cli.rtx_lidar_profile}")
    print("[INFO] Realtime mode: continuously render latest lidar scan.")
    print(f"[INFO] Environment layout: {args_cli.env_layout}")
    print(
        f"[INFO] lidar rates: scan={scan_rate_hz:.1f}Hz report={report_rate_hz:.1f}Hz "
        f"sample_mode={args_cli.capture_sample_mode}"
    )

    zero_actions = torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=torch.float32)
    for _ in range(int(args_cli.warmup_steps)):
        _ = env.step(zero_actions)
        _set_robot_fixed_forward_pose(env)
        _enforce_zero_xyz_velocity(env)

    step_dt = float(getattr(env, "step_dt", 0.02))
    report_samples_per_step = max(1, int(round(report_rate_hz * step_dt)))
    print(
        f"[INFO] step_dt={step_dt:.3f}s, report_samples_per_step={report_samples_per_step}"
    )

    frame_idx = 0
    while simulation_app.is_running():
        _ = env.step(zero_actions)
        _set_robot_fixed_forward_pose(env)
        _enforce_zero_xyz_velocity(env)

        frames_local: list[np.ndarray] = []
        if args_cli.capture_sample_mode == "control":
            pts = _read_one_scan_points_local(sensor)
            frames_local.append(pts)
        else:
            # Report-rate sampling: collect multiple lidar reads within one control step.
            for sample_idx in range(report_samples_per_step):
                if sample_idx > 0 and hasattr(simulation_app, "update"):
                    simulation_app.update()
                pts = _read_one_scan_points_local(sensor)
                frames_local.append(pts)

        hit_bins, miss_bins, rendered_frames = _draw_frames(env, frames_local)
        if frame_idx % 20 == 0:
            print(
                f"[INFO] realtime frame={frame_idx} rendered_frames={rendered_frames} "
                f"obstacle_threshold={float(args_cli.obstacle_distance_threshold):.2f}m "
                f"hit_bins={hit_bins} miss_bins={miss_bins}"
            )
        frame_idx += 1
        if hasattr(simulation_app, "update"):
            simulation_app.update()
        else:
            time.sleep(1.0 / 60.0)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
