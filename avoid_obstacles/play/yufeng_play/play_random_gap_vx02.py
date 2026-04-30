#!/usr/bin/env python3
"""Render walking Go2 in a fully-closed obstacle ring with vx=0.2."""

from __future__ import annotations

import argparse
import math
import os
import sys
import types
import json

import numpy as np
import torch
from tensordict import TensorDict

from omni.isaac.orbit.app import AppLauncher


LEGACY_WALK_CHECKPOINT = (
    "/home/dacheng.shen/Desktop/omniverse/avoid_obstacles/weights/model_1500.pt"
)
WORK_ROOT = "/home/dacheng.shen/Desktop/omniverse/avoid_obstacles"
PLAY_DIR = os.path.join(WORK_ROOT, "play")
CONFIG_DIR = os.path.join(WORK_ROOT, "configs")


parser = argparse.ArgumentParser(description="Closed-ring environment with fixed walking command vx=0.2.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "rough"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--walk_cmd_x", type=float, default=0.2)
parser.add_argument("--walk_cmd_y", type=float, default=0.0)
parser.add_argument("--walk_cmd_z", type=float, default=0.0)
parser.add_argument("--spawn_x", type=float, default=0.0)
parser.add_argument("--spawn_y", type=float, default=0.0)
parser.add_argument("--spawn_z", type=float, default=0.5)
parser.add_argument("--ring_obstacle_count", type=int, default=20)
parser.add_argument("--ring_radius", type=float, default=2.2)
parser.add_argument("--ring_obstacle_size", type=float, default=0.5)
parser.add_argument("--ring_obstacle_height", type=float, default=1.0)
parser.add_argument("--rtx_lidar_profile", type=str, default="Unitree_L1_PM_approx")
parser.add_argument("--lidar_bins", type=int, default=72, help="72 bins = 5 degrees per bin")
parser.add_argument("--lidar_range_min", type=float, default=0.05)
parser.add_argument("--lidar_range_max", type=float, default=20.0)
parser.add_argument("--obstacle_distance_threshold", type=float, default=5.0)
parser.add_argument("--ray_thickness", type=float, default=1.2)
parser.add_argument("--lidar_print_per_step", type=int, default=20)
parser.add_argument("--legacy_joint_remap", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--print_interval", type=int, default=200)
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

torch.manual_seed(args_cli.seed)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if PLAY_DIR not in sys.path:
    sys.path.insert(0, PLAY_DIR)
if CONFIG_DIR not in sys.path:
    sys.path.insert(0, CONFIG_DIR)

# custom_rl_env expects this module to exist.
sys.modules["omniverse_sim"] = types.SimpleNamespace(args_cli=args_cli)

import gymnasium as gym
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.usd
from rsl_rl.runners import OnPolicyRunner

import custom_rl_env
from agent_cfg import unitree_go2_agent_cfg
from custom_rl_env import UnitreeGo2CustomEnvCfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper


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


def _patch_policy_obs_dim(env, policy_obs_dim: int) -> None:
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
    if weight is None or len(weight.shape) != 2:
        return None
    return int(weight.shape[1])


def _adapt_obs_to_isaacgym_low_level(obs_td: TensorDict) -> TensorDict:
    if "policy" not in obs_td:
        return obs_td
    policy = obs_td["policy"]
    if not torch.is_tensor(policy) or policy.shape[-1] < 48:
        return obs_td
    scaled = policy.clone()
    scaled[:, 0:3] *= 2.0
    scaled[:, 3:6] *= 0.25
    scaled[:, 9:11] *= 2.0
    scaled[:, 11:12] *= 0.25
    scaled[:, 24:36] *= 0.05
    scaled = torch.clamp(scaled, -100.0, 100.0)
    return TensorDict({"policy": scaled}, batch_size=[scaled.shape[0]], device=scaled.device)


def _build_legacy_joint_mapping(base_env) -> torch.Tensor | None:
    if not args_cli.legacy_joint_remap:
        return None
    sim_joint_names = list(base_env.env.scene["robot"].joint_names)
    name_to_sim_idx = {name: idx for idx, name in enumerate(sim_joint_names)}
    missing = [name for name in LEGACY_GO2_JOINT_ORDER if name not in name_to_sim_idx]
    if missing:
        print(f"[WARN] Legacy remap disabled. Missing joints in Isaac Sim model: {missing}")
        return None

    mapping = torch.tensor(
        [name_to_sim_idx[name] for name in LEGACY_GO2_JOINT_ORDER],
        dtype=torch.long,
        device=base_env.device,
    )
    print(f"[INFO] Isaac Sim joint order: {sim_joint_names}")
    print(f"[INFO] Applying legacy joint remap (legacy->sim): {mapping.tolist()}")
    return mapping


def _remap_obs_to_legacy_joint_order(obs_td: TensorDict, mapping: torch.Tensor | None) -> TensorDict:
    if mapping is None or "policy" not in obs_td:
        return obs_td
    policy = obs_td["policy"]
    if not torch.is_tensor(policy) or policy.shape[-1] < 48:
        return obs_td
    remapped = policy.clone()
    remapped[:, 12:24] = policy[:, 12:24].index_select(1, mapping)
    remapped[:, 24:36] = policy[:, 24:36].index_select(1, mapping)
    remapped[:, 36:48] = policy[:, 36:48].index_select(1, mapping)
    return TensorDict({"policy": remapped}, batch_size=[remapped.shape[0]], device=remapped.device)


def _remap_actions_to_sim_joint_order(actions_legacy: torch.Tensor, mapping: torch.Tensor | None) -> torch.Tensor:
    if mapping is None:
        return actions_legacy
    out = torch.zeros_like(actions_legacy)
    out[:, mapping] = actions_legacy
    return out


def _align_env_cfg_for_isaacgym_low_level(env_cfg: UnitreeGo2CustomEnvCfg) -> None:
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.actions.joint_pos.scale = 0.25
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.42)
    actuator = env_cfg.scene.robot.actuators.get("base_legs", None)
    if actuator is not None:
        actuator.stiffness = 20.0
        actuator.damping = 0.5


def _apply_walk_spawn_pose(base_env) -> None:
    robot = base_env.env.scene["robot"]
    env_origins = base_env.env.scene.env_origins
    root = robot.data.root_state_w.clone()
    root[:, 0] = env_origins[:, 0] + float(args_cli.spawn_x)
    root[:, 1] = env_origins[:, 1] + float(args_cli.spawn_y)
    root[:, 2] = env_origins[:, 2] + float(args_cli.spawn_z)
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
    default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = torch.zeros_like(default_joint_pos)
    if hasattr(robot, "write_joint_state_to_sim"):
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


def _spawn_closed_ring_layouts(base_env) -> None:
    stage = omni.usd.get_context().get_stage()
    env_origins = base_env.env.scene.env_origins
    ring_count = max(4, int(args_cli.ring_obstacle_count))
    ring_radius = max(0.8, float(args_cli.ring_radius))
    obs_size = float(args_cli.ring_obstacle_size)
    obs_height = float(args_cli.ring_obstacle_height)

    print("[INFO] Closed ring layouts (no gap):")
    for env_id in range(base_env.num_envs):
        root_prim = f"/World/envs/env_{env_id}/ClosedRingLayoutWalk"
        if not stage.GetPrimAtPath(root_prim).IsValid():
            prim_utils.create_prim(root_prim, "Xform")

        cx = float(env_origins[env_id, 0].item())
        cy = float(env_origins[env_id, 1].item())
        for i in range(ring_count):
            theta = (2.0 * math.pi) * (i / float(ring_count))
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
        print(f"  env_{env_id}: obstacles={ring_count}, radius={ring_radius:.2f}m (fully closed)")


def _resolve_lidar_profile_path(profile_name: str) -> str | None:
    if profile_name.endswith(".json") and os.path.isfile(profile_name):
        return profile_name
    candidates = [
        os.path.join(WORK_ROOT, "env", "lidar", f"{profile_name}.json"),
        os.path.join("/home/dacheng.shen/Desktop/omniverse/IsaacLab/source/data/sensors/lidar", f"{profile_name}.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _load_lidar_rates(profile_name: str) -> tuple[float, float]:
    profile_path = _resolve_lidar_profile_path(profile_name)
    if profile_path is None:
        return 10.0, 1000.0
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        profile = data.get("profile", {})
        scan_rate = float(profile.get("scanRateBaseHz", 10.0))
        report_rate = float(profile.get("reportRateBaseHz", 1000.0))
        return scan_rate, report_rate
    except Exception:
        return 10.0, 1000.0


def _setup_rtx_lidar(base_env):
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("omni.isaac.sensor", True)
    ext_mgr.set_extension_enabled_immediate("omni.sensors.nv.lidar", True)
    ext_mgr.set_extension_enabled_immediate("omni.isaac.debug_draw", True)
    from omni.isaac.sensor import LidarRtx
    import omni.isaac.debug_draw._debug_draw as omni_debug_draw

    sensors = []
    for env_id in range(base_env.num_envs):
        sensor = LidarRtx(
            f"/World/envs/env_{env_id}/Robot/base/lidar_sensor_random_gap",
            rotation_frequency=20,
            pulse_time=1,
            translation=(0.0, 0.0, 0.1),
            orientation=(1.0, 0.0, 0.0, 0.0),
            config_file_name=args_cli.rtx_lidar_profile,
        )
        sensor.add_point_cloud_data_to_frame()
        sensor.initialize()
        sensors.append(sensor)
    draw = omni_debug_draw.acquire_debug_draw_interface()
    return sensors, draw


def _draw_lidar_bins(base_env, sensor, draw) -> tuple[int, int]:
    binned = _sample_lidar_bins(sensor)
    return _draw_lidar_bins_from_binned(base_env, binned, draw)


def _sample_lidar_bins(sensor) -> np.ndarray:
    frame = sensor.get_current_frame()
    points = frame.get("point_cloud_data", None)
    if points is None:
        points = np.empty((0, 3), dtype=np.float32)
    else:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            points = np.empty((0, 3), dtype=np.float32)
        else:
            points = points[:, :3]

    bins = int(args_cli.lidar_bins)
    binned = np.full((bins,), float(args_cli.lidar_range_max), dtype=np.float32)
    if points.shape[0] > 0:
        x = points[:, 0]
        y = points[:, 1]
        ranges = np.sqrt(x * x + y * y)
        angles = np.arctan2(y, x)
        valid = np.isfinite(ranges) & np.isfinite(angles) & (ranges > float(args_cli.lidar_range_min)) & (
            ranges < float(args_cli.lidar_range_max)
        )
        if np.any(valid):
            edges = np.linspace(-np.pi, np.pi, bins + 1, dtype=np.float32)
            idx = np.digitize(angles[valid], edges) - 1
            idx = np.clip(idx, 0, bins - 1)
            np.minimum.at(binned, idx, ranges[valid])
    return binned


def _draw_lidar_bins_from_binned(base_env, binned: np.ndarray, draw) -> tuple[int, int]:
    bins = int(binned.shape[0])
    root = base_env.env.scene["robot"].data.root_state_w[0].detach().float().cpu()
    base_pos = root[:3]
    quat = root[3:7]  # wxyz
    w, xq, yq, zq = quat[0], quat[1], quat[2], quat[3]
    yaw = torch.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq * yq + zq * zq)).item()
    origin = np.array([base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.1], dtype=np.float32)

    angles_world = np.linspace(-np.pi, np.pi, bins, endpoint=False, dtype=np.float32) + float(yaw)
    ends = np.stack(
        [
            origin[0] + binned * np.cos(angles_world),
            origin[1] + binned * np.sin(angles_world),
            np.full((bins,), origin[2], dtype=np.float32),
        ],
        axis=1,
    )
    starts = np.repeat(origin[None, :], bins, axis=0)
    hit_mask = binned <= float(args_cli.obstacle_distance_threshold)
    colors = [[1.0, 0.2, 0.2, 0.8] if bool(v) else [0.2, 1.0, 0.2, 0.65] for v in hit_mask.tolist()]
    sizes = [max(1.0, float(args_cli.ray_thickness))] * bins
    draw.clear_lines()
    draw.draw_lines(starts.tolist(), ends.tolist(), colors, sizes)
    return int(np.count_nonzero(hit_mask)), int(np.count_nonzero(~hit_mask))


def main() -> None:
    print(f"[INFO] Fixed low-level checkpoint: {LEGACY_WALK_CHECKPOINT}")
    env_cfg = UnitreeGo2CustomEnvCfg()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    _align_env_cfg_for_isaacgym_low_level(env_cfg)
    for i in range(env_cfg.scene.num_envs):
        custom_rl_env.base_command[str(i)] = [0.0, 0.0, 0.0]

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = RslRlVecEnvWrapper(base_env)
    _patch_env_observation_api(base_env)
    _ = base_env.reset()
    _spawn_closed_ring_layouts(base_env)

    low_actor_in_dim = _infer_legacy_actor_input_dim(LEGACY_WALK_CHECKPOINT)
    if low_actor_in_dim is not None:
        sample_obs = base_env.get_observations()["policy"]
        base_obs_dim = int(sample_obs.shape[-1])
        if base_obs_dim > low_actor_in_dim:
            print(f"[INFO] Truncating low-level obs: {base_obs_dim} -> {low_actor_in_dim}")
            _patch_policy_obs_dim(base_env, low_actor_in_dim)

    low_cfg = _adapt_legacy_agent_cfg(unitree_go2_agent_cfg)
    low_runner = OnPolicyRunner(base_env, low_cfg, log_dir=None, device=low_cfg["device"])
    _load_legacy_checkpoint_if_needed(low_runner, LEGACY_WALK_CHECKPOINT)
    low_policy = low_runner.get_inference_policy(device=base_env.device)
    legacy_joint_map = _build_legacy_joint_mapping(base_env)
    lidar_sensors, lidar_draw = _setup_rtx_lidar(base_env)
    scan_rate_hz, report_rate_hz = _load_lidar_rates(str(args_cli.rtx_lidar_profile))

    obs_td = base_env.reset()
    _apply_walk_spawn_pose(base_env)
    obs_td = base_env.get_observations()
    print(
        "[INFO] Start render: "
        f"target_cmd_xyz=({args_cli.walk_cmd_x:.3f}, {args_cli.walk_cmd_y:.3f}, {args_cli.walk_cmd_z:.3f}), "
        f"lidar_profile={args_cli.rtx_lidar_profile}, "
        f"profile_scan_rate={scan_rate_hz:.1f}Hz, profile_report_rate={report_rate_hz:.1f}Hz, "
        "sensor_rotation_frequency=20Hz"
    )
    print(
        "[INFO] Debug mode: one control step, then capture 1s lidar window "
        f"with {int(args_cli.lidar_print_per_step)} prints and pause on that frame."
    )

    step_idx = 0
    paused = False
    while simulation_app.is_running():
        if paused:
            if hasattr(simulation_app, "update"):
                simulation_app.update()
            continue

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
        step_idx += 1
        if torch.any(dones.to(dtype=torch.bool)):
            obs_td = base_env.reset()
            _apply_walk_spawn_pose(base_env)
            obs_td = base_env.get_observations()

        # Print lidar data N times for a 1-second capture window.
        samples_n = max(1, int(args_cli.lidar_print_per_step))
        sample_dt = 1.0 / float(samples_n)
        last_binned = None
        for i in range(samples_n):
            if i > 0 and hasattr(simulation_app, "update"):
                simulation_app.update()
            binned = _sample_lidar_bins(lidar_sensors[0])
            last_binned = binned
            binned_str = ", ".join(f"{v:.2f}" for v in binned.tolist())
            print(
                f"[LIDAR] step={step_idx} t={(i + 1) * sample_dt:.3f}s "
                f"sample={i + 1}/{samples_n} bins=[{binned_str}]"
            )

        if last_binned is not None:
            hit_bins, miss_bins = _draw_lidar_bins_from_binned(base_env, last_binned, lidar_draw)
        else:
            hit_bins, miss_bins = _draw_lidar_bins(base_env, lidar_sensors[0], lidar_draw)

        base_lin = obs_td["policy"][:, :3].mean(dim=0).detach().cpu()
        print(
            f"[INFO] step={step_idx} "
            f"target_cmd_xyz=({args_cli.walk_cmd_x:.3f}, {args_cli.walk_cmd_y:.3f}, {args_cli.walk_cmd_z:.3f}) "
            f"actual_vel_xyz=({base_lin[0]:.3f}, {base_lin[1]:.3f}, {base_lin[2]:.3f}) "
            f"lidar_hit_bins={hit_bins} lidar_miss_bins={miss_bins}"
        )
        print("[INFO] Paused after this control step. Keeping current render frame.")
        paused = True

    base_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
