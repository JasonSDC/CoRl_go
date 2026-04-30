#!/usr/bin/env python3
"""High-level command play: send (vx, vy, wz) and run low-level gait policy."""

from __future__ import annotations

import argparse
import os
import sys
import types

import torch
from tensordict import TensorDict
from omni.isaac.orbit.app import AppLauncher


parser = argparse.ArgumentParser(description="Play with high-level xyz velocity command.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "rough"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--vx", type=float, default=0.6, help="High-level forward speed command.")
parser.add_argument("--vy", type=float, default=0.0, help="High-level lateral speed command.")
parser.add_argument("--wz", type=float, default=0.0, help="High-level yaw rate command.")
parser.add_argument("--ramp_seconds", type=float, default=0.8, help="Linear ramp-up to target command.")
parser.add_argument("--print_interval", type=int, default=100, help="Print state every N steps.")
parser.add_argument(
    "--low_level_checkpoint",
    type=str,
    default="/home/dacheng.shen/Desktop/omniverse/avoid_obstacles/play/logs/rsl_rl/unitree_go2_rough/2024-04-06_02-37-07/model_7850.pt",
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
if PLAY_DIR not in sys.path:
    sys.path.insert(0, PLAY_DIR)

# custom_rl_env expects `from omniverse_sim import args_cli`.
sys.modules["omniverse_sim"] = types.SimpleNamespace(args_cli=args_cli)

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import custom_rl_env
from custom_rl_env import UnitreeGo2CustomEnvCfg
from agent_cfg import unitree_go2_agent_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper


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


def main() -> None:
    env_cfg = UnitreeGo2CustomEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    for i in range(env_cfg.scene.num_envs):
        custom_rl_env.base_command[str(i)] = [0.0, 0.0, 0.0]

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    _patch_env_observation_api(env)

    low_cfg = _adapt_legacy_agent_cfg(unitree_go2_agent_cfg)
    low_runner = OnPolicyRunner(env, low_cfg, log_dir=None, device=low_cfg["device"])
    print(f"[INFO] Loading low-level checkpoint: {args_cli.low_level_checkpoint}")
    _load_legacy_checkpoint_if_needed(low_runner, args_cli.low_level_checkpoint)
    low_policy = low_runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    step_idx = 0
    step_dt = float(getattr(env, "step_dt", 0.02))
    ramp_steps = max(1, int(round(float(args_cli.ramp_seconds) / max(1e-6, step_dt))))
    print(
        f"[INFO] High-level command mode: target_cmd=(vx={args_cli.vx:+.2f}, vy={args_cli.vy:+.2f}, wz={args_cli.wz:+.2f}), "
        f"ramp_steps={ramp_steps}, step_dt={step_dt:.3f}s"
    )

    while simulation_app.is_running():
        with torch.inference_mode():
            scale = min(1.0, float(step_idx + 1) / float(ramp_steps))
            cmd_vx = float(args_cli.vx) * scale
            cmd_vy = float(args_cli.vy) * scale
            cmd_wz = float(args_cli.wz) * scale
            for i in range(env.num_envs):
                custom_rl_env.base_command[str(i)] = [cmd_vx, cmd_vy, cmd_wz]

            actions = low_policy(obs)
            obs, _, _, _ = env.step(actions)

        step_idx += 1
        if args_cli.print_interval > 0 and step_idx % int(args_cli.print_interval) == 0:
            root_lin = env.env.scene["robot"].data.root_lin_vel_w[0].detach().float().cpu()
            root_ang = env.env.scene["robot"].data.root_ang_vel_w[0].detach().float().cpu()
            print(
                f"[INFO] step={step_idx} cmd=({cmd_vx:+.2f},{cmd_vy:+.2f},{cmd_wz:+.2f}) "
                f"root_lin=({root_lin[0]:+.2f},{root_lin[1]:+.2f},{root_lin[2]:+.2f}) "
                f"root_ang=({root_ang[0]:+.2f},{root_ang[1]:+.2f},{root_ang[2]:+.2f})"
            )

    for i in range(env.num_envs):
        custom_rl_env.base_command[str(i)] = [0.0, 0.0, 0.0]
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
