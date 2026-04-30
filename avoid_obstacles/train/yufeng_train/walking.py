#!/usr/bin/env python3
"""Train Go2 walking with unitree_rl_gym task/reward definitions.

Outputs (tensorboard + checkpoints) are saved to:
/home/dacheng.shen/Desktop/omniverse/avoid_obstacles/train/yufeng_train/<timestamp>/
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime


UNITREE_GYM_ROOT = "/home/dacheng.shen/Desktop/omniverse/unitree_rl_gym"
DEFAULT_OUTPUT_ROOT = "/home/dacheng.shen/Desktop/omniverse/avoid_obstacles/train/yufeng_train"
DEFAULT_CHECKPOINT_PATH = (
    "/home/dacheng.shen/Desktop/omniverse/avoid_obstacles/weights/model_1500.pt"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Go2 walking policy via unitree_rl_gym.")
    parser.add_argument("--task", type=str, default="go2", choices=["go2"])
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--load_run", type=str, default="-1")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--resume_path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Absolute/relative path to a checkpoint .pt file. If set, overrides --resume/--load_run/--checkpoint.",
    )
    parser.add_argument(
        "--target_lin_vel_x",
        type=float,
        default=1.0,
        help="Fixed forward command speed (m/s). Set to 0 to disable walking.",
    )
    return parser


def _resolve_resume_path(args: argparse.Namespace) -> str | None:
    if args.resume_path:
        resume_path = os.path.abspath(os.path.expanduser(args.resume_path))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        return resume_path

    if not args.resume:
        return None

    output_root = os.path.abspath(os.path.expanduser(args.output_root))
    if not os.path.isdir(output_root):
        raise ValueError(f"Resume requested but output root does not exist: {output_root}")

    if str(args.load_run) == "-1":
        runs = sorted(
            [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]
        )
        if not runs:
            raise ValueError(f"No run directories found under: {output_root}")
        load_run_dir = os.path.join(output_root, runs[-1])
    else:
        load_run = os.path.expanduser(str(args.load_run))
        load_run_dir = load_run if os.path.isabs(load_run) else os.path.join(output_root, load_run)
        if not os.path.isdir(load_run_dir):
            raise ValueError(f"Run directory not found: {load_run_dir}")

    def _model_sort_key(filename: str) -> int:
        match = re.fullmatch(r"model_(\d+)\.pt", filename)
        return int(match.group(1)) if match else -1

    if int(args.checkpoint) == -1:
        models = [f for f in os.listdir(load_run_dir) if _model_sort_key(f) >= 0]
        models.sort(key=_model_sort_key)
        if not models:
            raise ValueError(f"No model_*.pt files found under: {load_run_dir}")
        model_name = models[-1]
    else:
        model_name = f"model_{int(args.checkpoint)}.pt"

    resume_path = os.path.join(load_run_dir, model_name)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    return resume_path


def main() -> None:
    # Avoid permission issues in shared cache when isaacgym builds gymtorch extension.
    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(DEFAULT_OUTPUT_ROOT, ".torch_extensions")
        os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

    if UNITREE_GYM_ROOT not in sys.path:
        sys.path.insert(0, UNITREE_GYM_ROOT)

    # Use unitree_rl_gym training method / rewards.
    try:
        import legged_gym.envs  # noqa: F401  # registers tasks into task_registry
        from legged_gym.utils.helpers import class_to_dict, set_seed
        from legged_gym.utils.task_registry import task_registry
        from rsl_rl.runners import OnPolicyRunner
    except ModuleNotFoundError as exc:
        if "isaacgym" in str(exc):
            raise RuntimeError(
                "unitree_rl_gym requires Isaac Gym python package ('isaacgym'). "
                "Current environment does not provide it. "
                "Please run this script in a Unitree/IsaacGym environment, or install isaacgym first."
            ) from exc
        raise

    args = _build_parser().parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.output_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[INFO] unitree_rl_gym root: {UNITREE_GYM_ROOT}")
    print(f"[INFO] task={args.task} num_envs={args.num_envs} max_iterations={args.max_iterations}")
    print(f"[INFO] output folder: {run_dir}")

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    # Fix command profile for forward walking at a constant speed.
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e6
    env_cfg.commands.ranges.lin_vel_x = [float(args.target_lin_vel_x), float(args.target_lin_vel_x)]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # CLI overrides.
    train_cfg.seed = int(args.seed)
    train_cfg.runner.max_iterations = int(args.max_iterations)
    train_cfg.runner.save_interval = int(args.save_interval)
    train_cfg.runner.resume = bool(args.resume)
    train_cfg.runner.load_run = str(args.load_run)
    train_cfg.runner.checkpoint = int(args.checkpoint)
    train_cfg.runner.experiment_name = "yufeng_go2_walking"
    train_cfg.runner.run_name = timestamp

    set_seed(train_cfg.seed)
    runner = OnPolicyRunner(env, class_to_dict(train_cfg), run_dir, device=args.rl_device)
    resume_path = _resolve_resume_path(args)
    if resume_path:
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(resume_path)
    print("[INFO] Start training...")
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    env.close()
    print(f"[INFO] Training done. Saved under: {run_dir}")


if __name__ == "__main__":
    main()
