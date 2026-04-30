#!/usr/bin/env python3
"""Single-env random-gap lidar preview (feasibility toy example).

This script uses the same environment family as:
`avoid_obstacles/play/yufeng_play/preview_random_gap_env.py`
but keeps only one environment for quick visual inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import types

import numpy as np
import torch
from omni.isaac.orbit.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render 1 random-gap env with RTX lidar (5deg/bin) debug.")
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0")
    parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "rough"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ring_obstacle_count", type=int, default=20)
    parser.add_argument("--ring_radius", type=float, default=2.2)
    parser.add_argument("--obstacle_gap_width", type=float, default=0.93)
    parser.add_argument("--ring_obstacle_size", type=float, default=0.5)
    parser.add_argument("--ring_obstacle_height", type=float, default=1.0)
    parser.add_argument("--goal_outside_distance", type=float, default=1.0)
    parser.add_argument("--goal_marker_size", type=float, default=0.25)
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--rtx_lidar_profile", type=str, default="Unitree_L1")
    parser.add_argument("--lidar_bins", type=int, default=72, help="72 bins = 5 degrees per bin.")
    parser.add_argument("--lidar_range_min", type=float, default=0.05)
    parser.add_argument("--lidar_range_max", type=float, default=20.0)
    parser.add_argument("--obstacle_distance_threshold", type=float, default=5.0)
    parser.add_argument("--lidar_rotation_hz", type=float, default=10.0, help="Lidar rotation frequency.")
    parser.add_argument("--lidar_report_hz", type=float, default=1000.0, help="Lidar report/sample frequency.")
    parser.add_argument("--horizontal_elevation_deg", type=float, default=3.0)
    parser.add_argument("--ray_thickness", type=float, default=1.2)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _build_parser().parse_args()
torch.manual_seed(args_cli.seed)
random.seed(args_cli.seed)

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


def _set_robot_fixed_pose(env: RslRlVecEnvWrapper) -> None:
    """Keep robot stationary so we only inspect environment + lidar."""
    robot = env.env.scene["robot"]
    env_origins = env.env.scene.env_origins
    root_state = robot.data.root_state_w.clone()
    root_state[:, 0] = env_origins[:, 0]
    root_state[:, 1] = env_origins[:, 1]
    root_state[:, 2] = env_origins[:, 2] + 0.5
    # Quaternion wxyz -> face world +X (yaw=0)
    root_state[:, 3] = 1.0
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


def _spawn_goal_ring_with_text(
    stage,
    root_prim: str,
    center_x: float,
    center_y: float,
    ring_radius: float,
    ring_width: float,
    ring_height: float,
    text_yaw_rad: float,
) -> None:
    """Spawn a flat green ring and 'GOAL' block letters at center."""
    # Approximate a ring with many thin cuboids on XY plane.
    seg_count = 36
    mid_r = ring_radius
    seg_len = max(0.05, 2.0 * math.pi * mid_r / seg_count)
    for seg_id in range(seg_count):
        theta = (2.0 * math.pi * seg_id) / seg_count
        ox = center_x + mid_r * math.cos(theta)
        oy = center_y + mid_r * math.sin(theta)
        oz = 0.5 * ring_height
        prim_path = f"{root_prim}/goal_ring/seg_{seg_id}"
        if stage.GetPrimAtPath(prim_path).IsValid():
            continue
        seg_cfg = sim_utils.CuboidCfg(
            size=(seg_len, ring_width, ring_height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.9, 0.2), roughness=0.45),
        )
        seg_cfg.func(
            prim_path,
            seg_cfg,
            translation=(ox, oy, oz),
            orientation=(math.cos(theta * 0.5), 0.0, 0.0, math.sin(theta * 0.5)),
        )

    # Draw simple block letters "GOAL" with thin white cuboids.
    letter_h = 0.24
    letter_w = 0.13
    stroke = 0.03
    gap = 0.06
    text_z = ring_height + 0.01

    def _stroke(path: str, lx: float, ly: float, length: float, yaw_local: float):
        if stage.GetPrimAtPath(path).IsValid():
            return
        c = math.cos(text_yaw_rad)
        s = math.sin(text_yaw_rad)
        wx = center_x + c * lx - s * ly
        wy = center_y + s * lx + c * ly
        yaw = text_yaw_rad + yaw_local
        txt_cfg = sim_utils.CuboidCfg(
            size=(length, stroke, stroke),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.4),
        )
        txt_cfg.func(
            path,
            txt_cfg,
            translation=(wx, wy, text_z),
            orientation=(math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)),
        )

    # Letter anchors.
    total_w = 4 * letter_w + 3 * gap
    x0 = -0.5 * total_w + 0.5 * letter_w
    y0 = 0.0

    # G
    gx = x0
    _stroke(f"{root_prim}/goal_text/G_top", gx, y0 + 0.5 * letter_h, letter_w, 0.0)
    _stroke(f"{root_prim}/goal_text/G_bot", gx, y0 - 0.5 * letter_h, letter_w, 0.0)
    _stroke(f"{root_prim}/goal_text/G_left", gx - 0.5 * letter_w, y0, letter_h, 0.5 * math.pi)
    _stroke(f"{root_prim}/goal_text/G_mid", gx + 0.1 * letter_w, y0, 0.8 * letter_w, 0.0)
    _stroke(f"{root_prim}/goal_text/G_right_low", gx + 0.5 * letter_w, y0 - 0.18 * letter_h, 0.65 * letter_h, 0.5 * math.pi)

    # O
    ox = x0 + (letter_w + gap)
    _stroke(f"{root_prim}/goal_text/O_top", ox, y0 + 0.5 * letter_h, letter_w, 0.0)
    _stroke(f"{root_prim}/goal_text/O_bot", ox, y0 - 0.5 * letter_h, letter_w, 0.0)
    _stroke(f"{root_prim}/goal_text/O_left", ox - 0.5 * letter_w, y0, letter_h, 0.5 * math.pi)
    _stroke(f"{root_prim}/goal_text/O_right", ox + 0.5 * letter_w, y0, letter_h, 0.5 * math.pi)

    # A
    ax = x0 + 2.0 * (letter_w + gap)
    _stroke(f"{root_prim}/goal_text/A_top", ax, y0 + 0.5 * letter_h, letter_w, 0.0)
    _stroke(f"{root_prim}/goal_text/A_mid", ax, y0, letter_w, 0.0)
    _stroke(f"{root_prim}/goal_text/A_left", ax - 0.5 * letter_w, y0, letter_h, 0.5 * math.pi)
    _stroke(f"{root_prim}/goal_text/A_right", ax + 0.5 * letter_w, y0, letter_h, 0.5 * math.pi)

    # L
    lx = x0 + 3.0 * (letter_w + gap)
    _stroke(f"{root_prim}/goal_text/L_left", lx - 0.5 * letter_w, y0, letter_h, 0.5 * math.pi)
    _stroke(f"{root_prim}/goal_text/L_bot", lx, y0 - 0.5 * letter_h, letter_w, 0.0)


def _spawn_one_random_gap_layout(env: RslRlVecEnvWrapper) -> None:
    stage = omni.usd.get_context().get_stage()
    env_origins = env.env.scene.env_origins

    ring_count = max(4, int(args_cli.ring_obstacle_count))
    ring_radius = max(0.8, float(args_cli.ring_radius))
    gap_width = max(0.1, float(args_cli.obstacle_gap_width))
    obs_size = float(args_cli.ring_obstacle_size)
    obs_height = float(args_cli.ring_obstacle_height)
    goal_size = float(args_cli.goal_marker_size)
    goal_distance = ring_radius + max(0.1, float(args_cli.goal_outside_distance))
    gap_half_angle = min(math.pi * 0.8, (gap_width / ring_radius) * 0.5)
    gap_angle = 2.0 * gap_half_angle
    arc_len = max(1e-6, 2.0 * math.pi - gap_angle)

    env_id = 0
    gap_center = random.uniform(-math.pi, math.pi)
    root_prim = f"/World/envs/env_{env_id}/RandomGapLayout"
    if not stage.GetPrimAtPath(root_prim).IsValid():
        prim_utils.create_prim(root_prim, "Xform")

    cx = float(env_origins[env_id, 0].item())
    cy = float(env_origins[env_id, 1].item())
    for i in range(ring_count):
        frac = (i + 0.5) / float(ring_count)
        theta = gap_center + gap_half_angle + frac * arc_len
        ox = cx + ring_radius * math.cos(theta)
        oy = cy + ring_radius * math.sin(theta)
        oz = 0.5 * obs_height
        prim_path = f"{root_prim}/obs_{i}"
        cube_cfg = sim_utils.CuboidCfg(
            size=(obs_size, obs_size, obs_height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2), roughness=0.7),
        )
        cube_cfg.func(prim_path, cube_cfg, translation=(ox, oy, oz))

    gx = cx + goal_distance * math.cos(gap_center)
    gy = cy + goal_distance * math.sin(gap_center)
    _spawn_goal_ring_with_text(
        stage=stage,
        root_prim=root_prim,
        center_x=gx,
        center_y=gy,
        ring_radius=2.0 * max(0.18, goal_size * 0.9),
        ring_width=max(0.03, goal_size * 0.18),
        ring_height=max(0.02, goal_size * 0.06),
        text_yaw_rad=math.atan2(cy - gy, cx - gx) - 0.5 * math.pi,
    )

    print(
        f"[INFO] env_0 gap_center={math.degrees(gap_center):+.1f} deg, "
        f"goal=({gx:.2f}, {gy:.2f})"
    )


def _setup_rtx_lidar(env: RslRlVecEnvWrapper):
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("omni.isaac.sensor", True)
    ext_mgr.set_extension_enabled_immediate("omni.sensors.nv.lidar", True)
    from omni.isaac.sensor import LidarRtx

    sensor = LidarRtx(
        "/World/envs/env_0/Robot/base/lidar_sensor_toy",
        rotation_frequency=float(args_cli.lidar_rotation_hz),
        pulse_time=1,
        translation=(0.0, 0.0, 0.1),
        orientation=(1.0, 0.0, 0.0, 0.0),
        config_file_name=str(args_cli.rtx_lidar_profile),
    )
    sensor.add_point_cloud_data_to_frame()
    sensor.initialize()
    return sensor


def _read_one_scan_points_local(sensor) -> np.ndarray:
    frame = sensor.get_current_frame()
    points = frame.get("point_cloud_data", None)
    if points is None:
        return np.empty((0, 3), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    return pts[:, :3]


def _filter_horizontal_points(points_local: np.ndarray, max_elevation_deg: float) -> np.ndarray:
    if points_local.shape[0] == 0:
        return points_local
    xy_norm = np.linalg.norm(points_local[:, :2], axis=1)
    elev_deg = np.degrees(np.arctan2(points_local[:, 2], np.maximum(xy_norm, 1e-6)))
    return points_local[np.abs(elev_deg) <= float(max_elevation_deg)]


def _compute_lidar_bins(points_local: np.ndarray) -> np.ndarray:
    bins = int(args_cli.lidar_bins)
    out = np.full((bins,), float(args_cli.lidar_range_max), dtype=np.float32)
    if points_local.shape[0] == 0:
        return out
    x = points_local[:, 0]
    y = points_local[:, 1]
    ranges = np.sqrt(x * x + y * y)
    angles = np.arctan2(y, x)
    valid = np.isfinite(ranges) & np.isfinite(angles)
    valid &= ranges > float(args_cli.lidar_range_min)
    valid &= ranges < float(args_cli.lidar_range_max)
    if not np.any(valid):
        return out
    edges = np.linspace(-np.pi, np.pi, bins + 1, dtype=np.float32)
    idx = np.digitize(angles[valid], edges) - 1
    idx = np.clip(idx, 0, bins - 1)
    np.minimum.at(out, idx, ranges[valid])
    return out


def _draw_binned_scan(env: RslRlVecEnvWrapper, binned: np.ndarray) -> tuple[int, int]:
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("omni.isaac.debug_draw", True)
    import omni.isaac.debug_draw._debug_draw as omni_debug_draw

    draw = omni_debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()

    root_state = env.env.scene["robot"].data.root_state_w[0].detach().float().cpu()
    base_pos = root_state[:3]
    sensor_origin = np.array([base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.1], dtype=np.float32)

    bins = int(binned.shape[0])
    # 5-deg bins when bins=72
    angle_centers = np.linspace(-np.pi, np.pi, bins, endpoint=False, dtype=np.float32)
    ends = np.stack(
        [
            sensor_origin[0] + binned * np.cos(angle_centers),
            sensor_origin[1] + binned * np.sin(angle_centers),
            np.full((bins,), sensor_origin[2], dtype=np.float32),
        ],
        axis=1,
    )
    starts = np.repeat(sensor_origin[None, :], bins, axis=0)
    obstacle_threshold = min(float(args_cli.obstacle_distance_threshold), float(args_cli.lidar_range_max))
    hit_mask = binned <= obstacle_threshold
    colors = [[1.0, 0.2, 0.2, 0.85] if bool(v) else [0.2, 1.0, 0.2, 0.65] for v in hit_mask.tolist()]
    sizes = [float(args_cli.ray_thickness)] * bins
    draw.draw_lines(starts.tolist(), ends.tolist(), colors, sizes)
    return int(np.count_nonzero(hit_mask)), int(np.count_nonzero(~hit_mask))


def _try_load_profile_rates(profile_name: str) -> tuple[float, float]:
    candidate_paths = [
        os.path.join(WORK_ROOT, "env", "lidar", f"{profile_name}.json"),
        os.path.join("/home/dacheng.shen/Desktop/omniverse/IsaacLab/source/data/sensors/lidar", f"{profile_name}.json"),
    ]
    for path in candidate_paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profile = data.get("profile", {})
            scan_rate = float(profile.get("scanRateBaseHz", args_cli.lidar_rotation_hz))
            report_rate = float(profile.get("reportRateBaseHz", args_cli.lidar_report_hz))
            return scan_rate, report_rate
        except Exception:
            pass
    return float(args_cli.lidar_rotation_hz), float(args_cli.lidar_report_hz)


def main() -> None:
    env_cfg = UnitreeGo2CustomEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    # Keep command zero so robot does not move (sensor-only inspection).
    custom_rl_env.base_command["0"] = [0.0, 0.0, 0.0]

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    _ = env.reset()
    _spawn_one_random_gap_layout(env)
    _set_robot_fixed_pose(env)
    lidar_sensor = _setup_rtx_lidar(env)
    scan_rate_hz = float(args_cli.lidar_rotation_hz)
    report_rate_hz = float(args_cli.lidar_report_hz)
    step_dt = float(getattr(env, "step_dt", 0.02))
    report_samples_per_step = max(1, int(round(report_rate_hz * step_dt)))

    print("[INFO] Single-env preview ready. Showing RTX lidar binned rays.")
    print(
        f"[INFO] bins={int(args_cli.lidar_bins)} (~{360.0 / float(args_cli.lidar_bins):.1f}deg/bin), "
        f"rotation={scan_rate_hz:.1f}Hz, report={report_rate_hz:.1f}Hz, "
        f"report_samples_per_step={report_samples_per_step}, obstacle_threshold={float(args_cli.obstacle_distance_threshold):.2f}m"
    )
    print("[INFO] Close the simulator window when done.")

    zero_actions = torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=torch.float32)
    step_idx = 0
    while simulation_app.is_running():
        _ = env.step(zero_actions)
        _set_robot_fixed_pose(env)

        frames_local: list[np.ndarray] = []
        for sample_idx in range(report_samples_per_step):
            if sample_idx > 0 and hasattr(simulation_app, "update"):
                simulation_app.update()
            pts = _read_one_scan_points_local(lidar_sensor)
            pts = _filter_horizontal_points(pts, float(args_cli.horizontal_elevation_deg))
            frames_local.append(pts)

        if len(frames_local) > 0:
            merged = np.concatenate([f for f in frames_local if f.shape[0] > 0], axis=0) if any(
                f.shape[0] > 0 for f in frames_local
            ) else np.empty((0, 3), dtype=np.float32)
        else:
            merged = np.empty((0, 3), dtype=np.float32)
        binned = _compute_lidar_bins(merged)
        hit_bins, miss_bins = _draw_binned_scan(env, binned)

        step_idx += 1
        if int(args_cli.print_interval) > 0 and step_idx % int(args_cli.print_interval) == 0:
            print(
                f"[INFO] step={step_idx} "
                f"lidar(min/mean/max)=({float(binned.min()):.3f}, {float(binned.mean()):.3f}, {float(binned.max()):.3f}) "
                f"hit_bins={hit_bins} miss_bins={miss_bins}"
            )
        if hasattr(simulation_app, "update"):
            simulation_app.update()
        else:
            time.sleep(1.0 / 60.0)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
