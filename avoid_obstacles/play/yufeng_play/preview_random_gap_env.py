#!/usr/bin/env python3
"""Preview random ring-gap obstacle environments (5 random layouts)."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import types

import torch
from omni.isaac.orbit.app import AppLauncher


parser = argparse.ArgumentParser(description="Preview 5 random ring-gap obstacle layouts.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=5)
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "rough"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ring_obstacle_count", type=int, default=20)
parser.add_argument("--ring_radius", type=float, default=2.2)
parser.add_argument("--obstacle_gap_width", type=float, default=0.93)
parser.add_argument("--ring_obstacle_size", type=float, default=0.5)
parser.add_argument("--ring_obstacle_height", type=float, default=1.0)
parser.add_argument("--goal_outside_distance", type=float, default=1.0, help="Goal distance outside the ring.")
parser.add_argument("--goal_marker_size", type=float, default=0.25)
parser.add_argument("--print_interval", type=int, default=100)
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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


def _set_robot_fixed_forward_pose(env: RslRlVecEnvWrapper) -> None:
    robot = env.env.scene["robot"]
    env_origins = env.env.scene.env_origins
    root_state = robot.data.root_state_w.clone()
    root_state[:, 0] = env_origins[:, 0]
    root_state[:, 1] = env_origins[:, 1]
    root_state[:, 2] = env_origins[:, 2] + 0.5
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


def _spawn_random_gap_layouts(env: RslRlVecEnvWrapper) -> None:
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

    print("[INFO] Random gap layouts:")
    for env_id in range(env.num_envs):
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
        gz = 0.5 * goal_size
        goal_path = f"{root_prim}/goal"
        goal_cfg = sim_utils.CuboidCfg(
            size=(goal_size, goal_size, goal_size),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2), roughness=0.5),
        )
        goal_cfg.func(goal_path, goal_cfg, translation=(gx, gy, gz))

        print(
            f"  env_{env_id}: gap_center={math.degrees(gap_center):+.1f} deg, "
            f"goal=({gx:.2f}, {gy:.2f})"
        )


def main() -> None:
    env_cfg = UnitreeGo2CustomEnvCfg()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    for i in range(env_cfg.scene.num_envs):
        custom_rl_env.base_command[str(i)] = [0.0, 0.0, 0.0]

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    _ = env.reset()
    _spawn_random_gap_layouts(env)
    _set_robot_fixed_forward_pose(env)
    _enforce_zero_xyz_velocity(env)

    print(
        f"[INFO] Preview running with {env.num_envs} random layouts. "
        "Close the app window when done."
    )
    zero_actions = torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=torch.float32)
    step_idx = 0
    while simulation_app.is_running():
        _ = env.step(zero_actions)
        _set_robot_fixed_forward_pose(env)
        _enforce_zero_xyz_velocity(env)
        step_idx += 1
        if int(args_cli.print_interval) > 0 and step_idx % int(args_cli.print_interval) == 0:
            print(f"[INFO] preview step={step_idx}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
