# avoid_obstacles workspace

This folder is an isolated workspace for LiDAR obstacle-avoidance work without changing the original runnable demo under `go2_omniverse`.

## Structure

- `run_sim_avoid.sh`: main launcher from this workspace.
- `scripts/sync_lidar_profiles.sh`: sync LiDAR JSONs in this workspace into IsaacLab runtime path.
- `play/`: copied play-time code and local checkpoint mirror.
- `train/`: copied IsaacLab RSL-RL train/play workflow scripts.
- `env/`: copied environment config and LiDAR profiles.
- `weights/`: copied baseline checkpoint (`model_7850.pt`).
- `configs/`: copied policy/CLI config files.
- `test/`: reserved for evaluation/test scripts.
- `logs/`: reserved for new training/play logs.

## Recommended run commands

Run from anywhere:

```bash
bash /home/dacheng.shen/Desktop/omniverse/avoid_obstacles/run_sim_avoid.sh
```

GUI + PM approximate LiDAR:

```bash
GO2_HEADLESS=0 GO2_USE_RTX_LIDAR_DEBUG=1 GO2_LIDAR_DEBUG=1 GO2_RTX_LIDAR_PROFILE=Unitree_L1_PM_approx bash /home/dacheng.shen/Desktop/omniverse/avoid_obstacles/run_sim_avoid.sh --terrain rough
```

GUI + RM approximate LiDAR:

```bash
GO2_HEADLESS=0 GO2_USE_RTX_LIDAR_DEBUG=1 GO2_LIDAR_DEBUG=1 GO2_RTX_LIDAR_PROFILE=Unitree_L1_RM_approx bash /home/dacheng.shen/Desktop/omniverse/avoid_obstacles/run_sim_avoid.sh --terrain rough
```

## Notes

- The launcher syncs `Unitree_L1_PM_approx.json` and `Unitree_L1_RM_approx.json` into `IsaacLab/source/data/sensors/lidar/` before start.
- Original `go2_omniverse` files are preserved; you can continue to use them independently.
