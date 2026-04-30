#!/usr/bin/env python3
"""Run obstacle-avoidance play with 10-frame lidar temporal min enabled."""

from __future__ import annotations

import os
import sys


def main() -> None:
    target = "/home/dacheng.shen/Desktop/omniverse/avoid_obstacles/play/play_obstacle_avoidance.py"
    argv = [
        sys.executable,
        target,
        "--temporal_min_frames",
        "10",
        "--freeze_robot",
        *sys.argv[1:],
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
