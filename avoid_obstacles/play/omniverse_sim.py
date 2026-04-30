"""Script to play a checkpoint if an RL agent from RSL-RL."""
from __future__ import annotations


"""Launch Isaac Sim Simulator first."""
import argparse
# tensordict before AppLauncher so Kit's old pip_prebundle typing_extensions does not win on sys.path.
from tensordict import TensorDict
from omni.isaac.orbit.app import AppLauncher


import cli_args  
import time
import os
import threading
import copy


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--custom_env", type=str, default="office", help="Setup the environment")
parser.add_argument("--robot", type=str, default="go2", help="Setup the robot")
parser.add_argument("--terrain", type=str, default="rough", help="Setup the robot")
parser.add_argument("--robot_amount", type=int, default=1, help="Setup the robot amount")


# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
_launcher_kw = {}
if os.environ.get("GO2_MULTI_GPU", "1").strip().lower() in ("0", "false", "no", "off"):
    _launcher_kw["multi_gpu"] = False
app_launcher = AppLauncher(args_cli, **_launcher_kw)
simulation_app = app_launcher.app


import omni

ros2_mode = os.environ.get("GO2_ENABLE_ROS2", "auto").strip().lower()
ROS2_ENABLED = ros2_mode in ("1", "true", "yes", "on", "auto")
# Isaac Sim 2023.1.1 ROS bridge does not support Jazzy; disable by default in auto mode.
if ros2_mode == "auto" and os.environ.get("ROS_DISTRO", "").strip().lower() == "jazzy":
    ROS2_ENABLED = False


ext_manager = omni.kit.app.get_app().get_extension_manager()
if ROS2_ENABLED:
    try:
        ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)
    except Exception as exc:
        print(f"[WARN] Failed to enable ROS2 bridge: {exc}. Running without ROS2 bridge.")
        ROS2_ENABLED = False
else:
    print("[INFO] ROS2 bridge disabled (set GO2_ENABLE_ROS2=1 to force-enable).")

# FOR VR SUPPORT
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.core", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.steamvr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.simulatedxr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.openxr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.telemetry", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.profile.vr", True)


"""Rest everything follows."""
import gymnasium as gym
import torch
import carb


from omni.isaac.orbit_tasks.utils import get_checkpoint_path
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper
)
import omni.isaac.orbit.sim as sim_utils
import omni.appwindow
from rsl_rl.runners import OnPolicyRunner


if ROS2_ENABLED:
    try:
        import rclpy
        from ros2 import RobotBaseNode, add_camera, add_rtx_lidar, pub_robo_data_ros2
        from geometry_msgs.msg import Twist
    except Exception as exc:
        print(f"[WARN] ROS2 Python imports failed: {exc}. Running without ROS2 interfaces.")
        ROS2_ENABLED = False


from agent_cfg import unitree_go2_agent_cfg, unitree_g1_agent_cfg
from custom_rl_env import UnitreeGo2CustomEnvCfg, G1RoughEnvCfg
import custom_rl_env

from omnigraph import create_front_cam_omnigraph


def _add_local_debug_lidar(num_envs: int, robot_type: str):
    """Create RTX LiDAR sensors and attach debug pointcloud writer for local viewport display."""
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    # In non-ROS mode this extension is not always auto-loaded.
    ext_mgr.set_extension_enabled_immediate("omni.isaac.sensor", True)
    ext_mgr.set_extension_enabled_immediate("omni.sensors.nv.lidar", True)

    import omni.replicator.core as rep
    from omni.isaac.sensor import LidarRtx

    lidar_profile = os.environ.get("GO2_RTX_LIDAR_PROFILE", "Unitree_L1").strip() or "Unitree_L1"
    print(f"[INFO] Using RTX LiDAR profile: {lidar_profile}")

    for i in range(num_envs):
        if robot_type == "g1":
            lidar_sensor = LidarRtx(
                f"/World/envs/env_{i}/Robot/head_link/lidar_sensor",
                rotation_frequency=200,
                pulse_time=1,
                translation=(0.0, 0.0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                config_file_name=lidar_profile,
            )
        else:
            lidar_sensor = LidarRtx(
                f"/World/envs/env_{i}/Robot/base/lidar_sensor",
                rotation_frequency=200,
                pulse_time=1,
                translation=(0.0, 0.0, 0.4),
                orientation=(1.0, 0.0, 0.0, 0.0),
                config_file_name=lidar_profile,
            )

        writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
        writer.attach([lidar_sensor.get_render_product_path()])


def _init_height_scan_debug_draw():
    """Initialize debug-draw interface for height-scan pointcloud rendering."""
    import omni.isaac.debug_draw._debug_draw as omni_debug_draw

    draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    x_vals = torch.linspace(-0.8, 0.8, 17)
    y_vals = torch.linspace(-0.5, 0.5, 11)
    grid = torch.stack(torch.meshgrid(x_vals, y_vals, indexing="ij"), dim=-1).reshape(-1, 2)
    return draw_interface, grid


def _draw_height_scan_pointcloud(draw_interface, grid_xy: torch.Tensor, env, obs: TensorDict):
    """Render observation height-scan as pointcloud-like short line segments."""
    if "policy" not in obs:
        return
    policy_obs = obs["policy"]
    if policy_obs.shape[-1] < 187:
        return

    scan = policy_obs[0, -187:].detach().float().cpu()
    base_state = env.env.scene["robot"].data.root_state_w[0].detach().float().cpu()
    base_pos = base_state[:3]
    quat_wxyz = base_state[3:7]
    w, x, y, z = quat_wxyz
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    c = torch.cos(yaw)
    s = torch.sin(yaw)
    rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    world_xy = torch.matmul(grid_xy, rot.T) + base_pos[:2]
    world_z = base_pos[2] + scan
    points = torch.column_stack((world_xy, world_z))

    starts = points.clone()
    starts[:, 2] -= 0.015
    ends = points

    draw_interface.clear_lines()
    colors = [[0.1, 0.9, 1.0, 1.0]] * points.shape[0]
    sizes = [2.0] * points.shape[0]
    draw_interface.draw_lines(starts.tolist(), ends.tolist(), colors, sizes)


def _adapt_legacy_agent_cfg(cfg: dict) -> dict:
    """Adapt old rsl_rl config schema to current schema."""
    adapted = copy.deepcopy(cfg)
    algorithm_cfg = adapted.setdefault("algorithm", {})
    algorithm_cfg.setdefault("class_name", "PPO")
    adapted.setdefault("obs_groups", {})

    # Legacy config used "policy" key with ActorCritic settings.
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

    # New rsl_rl expects explicit observation groups.
    if not adapted.get("obs_groups"):
        adapted["obs_groups"] = {"actor": ["policy"], "critic": ["policy"]}

    return adapted


def _load_legacy_checkpoint_if_needed(ppo_runner: OnPolicyRunner, checkpoint_path: str) -> None:
    """Load old actor_critic checkpoint format if direct load fails."""
    try:
        ppo_runner.load(checkpoint_path)
        return
    except Exception as exc:
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
    """Normalize env.get_observations() return type for new rsl_rl API."""
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


def sub_keyboard_event(event, *args, **kwargs) -> bool:

    if len(custom_rl_env.base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'W':
                custom_rl_env.base_command["0"] = [1, 0, 0]
            if event.input.name == 'S':
                custom_rl_env.base_command["0"] = [-1, 0, 0]
            if event.input.name == 'A':
                custom_rl_env.base_command["0"] = [0, 1, 0]
            if event.input.name == 'D':
                custom_rl_env.base_command["0"] = [0, -1, 0]
            if event.input.name == 'Q':
                custom_rl_env.base_command["0"] = [0, 0, 1]
            if event.input.name == 'E':
                custom_rl_env.base_command["0"] = [0, 0, -1]

            if len(custom_rl_env.base_command) > 1:
                if event.input.name == 'I':
                    custom_rl_env.base_command["1"] = [1, 0, 0]
                if event.input.name == 'K':
                    custom_rl_env.base_command["1"] = [-1, 0, 0]
                if event.input.name == 'J':
                    custom_rl_env.base_command["1"] = [0, 1, 0]
                if event.input.name == 'L':
                    custom_rl_env.base_command["1"] = [0, -1, 0]
                if event.input.name == 'U':
                    custom_rl_env.base_command["1"] = [0, 0, 1]
                if event.input.name == 'O':
                    custom_rl_env.base_command["1"] = [0, 0, -1]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            for i in range(len(custom_rl_env.base_command)):
                custom_rl_env.base_command[str(i)] = [0, 0, 0]
    return True


def setup_custom_env():
    try:
        if (args_cli.custom_env == "warehouse" and args_cli.terrain == 'flat'):
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/warehouse.usd")
            cfg_scene.func("/World/warehouse", cfg_scene, translation=(0.0, 0.0, 0.0))

        if (args_cli.custom_env == "office" and args_cli.terrain == 'flat'):
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/office.usd")
            cfg_scene.func("/World/office", cfg_scene, translation=(0.0, 0.0, 0.0))
    except:
        print("Error loading custom environment. You should download custom envs folder from: https://drive.google.com/drive/folders/1vVGuO1KIX1K6mD6mBHDZGm9nk2vaRyj3?usp=sharing")


def cmd_vel_cb(msg, num_robot):
    x = msg.linear.x
    y = msg.linear.y
    z = msg.angular.z
    custom_rl_env.base_command[str(num_robot)] = [x, y, z]



def add_cmd_sub(num_envs):
    if not ROS2_ENABLED:
        return
    node_test = rclpy.create_node('position_velocity_publisher')
    for i in range(num_envs):
        node_test.create_subscription(Twist, f'robot{i}/cmd_vel', lambda msg, i=i: cmd_vel_cb(msg, str(i)), 10)
    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node_test,), daemon=True)
    thread.start()



def specify_cmd_for_robots(numv_envs):
    for i in range(numv_envs):
        custom_rl_env.base_command[str(i)] = [0, 0, 0]
def run_sim():
    
    # acquire input interface
    _input = carb.input.acquire_input_interface()
    _appwindow = omni.appwindow.get_default_app_window()
    _keyboard = _appwindow.get_keyboard()
    _sub_keyboard = _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)

    """Play with RSL-RL agent."""
    # parse configuration
    
    env_cfg = UnitreeGo2CustomEnvCfg()
    
    if args_cli.robot == "g1":
        env_cfg = G1RoughEnvCfg()

    # add N robots to env 
    env_cfg.scene.num_envs = args_cli.robot_amount

    # create ros2 camera stream omnigraph
    if ROS2_ENABLED:
        for i in range(env_cfg.scene.num_envs):
            create_front_cam_omnigraph(i)
        
    specify_cmd_for_robots(env_cfg.scene.num_envs)

    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_agent_cfg

    if args_cli.robot == "g1":
        agent_cfg: RslRlOnPolicyRunnerCfg = unitree_g1_agent_cfg

    agent_cfg = _adapt_legacy_agent_cfg(agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    _patch_env_observation_api(env)
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(log_root_path, agent_cfg["load_run"], agent_cfg["load_checkpoint"])
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    _load_legacy_checkpoint_if_needed(ppo_runner, resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs = env.get_observations()

    # initialize ROS2 node and interfaces if enabled
    base_node = None
    annotator_lst = []
    height_scan_draw = None
    height_scan_grid = None
    if ROS2_ENABLED:
        rclpy.init()
        base_node = RobotBaseNode(env_cfg.scene.num_envs)
        add_cmd_sub(env_cfg.scene.num_envs)
        annotator_lst = add_rtx_lidar(env_cfg.scene.num_envs, args_cli.robot, False)
        add_camera(env_cfg.scene.num_envs, args_cli.robot)
    elif os.environ.get("GO2_LIDAR_DEBUG", "1").strip().lower() in ("1", "true", "yes", "on"):
        try:
            _ext = omni.kit.app.get_app().get_extension_manager()
            if os.environ.get("GO2_USE_RTX_LIDAR_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on"):
                _add_local_debug_lidar(env_cfg.scene.num_envs, args_cli.robot)
                print("[INFO] RTX LiDAR + replicator debug draw enabled.")
            else:
                # orbit.python.headless.kit does not pull this in by default.
                _ext.set_extension_enabled_immediate("omni.isaac.debug_draw", True)
                height_scan_draw, height_scan_grid = _init_height_scan_debug_draw()
                print("[INFO] Height-scan pointcloud debug draw enabled.")
        except Exception as exc:
            print(f"[WARN] Failed to enable LiDAR debug draw: {exc}")
    setup_custom_env()
    fixed_cmd = [1.0, 0.0, 0.0]  # target high-level command: vx, vy, wz
    print(
        f"[INFO] Fixed high-level command enabled on terrain={args_cli.terrain}: "
        f"cmd=({fixed_cmd[0]:+.2f}, {fixed_cmd[1]:+.2f}, {fixed_cmd[2]:+.2f})"
    )
    
    start_time = time.time()
    step_idx = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            for i in range(env_cfg.scene.num_envs):
                custom_rl_env.base_command[str(i)] = fixed_cmd
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            step_idx += 1
            if step_idx % 20 == 0:
                root_lin_vel = env.env.scene["robot"].data.root_lin_vel_w[0].detach().float().cpu()
                print(
                    f"[INFO] step={step_idx} cmd=({fixed_cmd[0]:+.2f}, {fixed_cmd[1]:+.2f}, {fixed_cmd[2]:+.2f}) "
                    f"actual_lin_vel=({root_lin_vel[0]:+.2f}, {root_lin_vel[1]:+.2f}, {root_lin_vel[2]:+.2f})"
                )
            if height_scan_draw is not None and height_scan_grid is not None:
                _draw_height_scan_pointcloud(height_scan_draw, height_scan_grid, env, obs)
            if ROS2_ENABLED:
                pub_robo_data_ros2(args_cli.robot, env_cfg.scene.num_envs, base_node, env, annotator_lst, start_time)
    env.close()
    if ROS2_ENABLED:
        rclpy.shutdown()