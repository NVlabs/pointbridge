# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
The file show points in gripper frame with respect to the Franka Hand
gripper frame on a real FR3 robot.
"""

import numpy as np

extrapoints = [
    # gripper points
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.04],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 1
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.04],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 2
    # First horizontal line
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.08],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 3
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.05],
            [0.0, 0.0, 1.0, -0.08],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 4
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.05],
            [0.0, 0.0, 1.0, -0.08],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 5
    # Second horizontal line
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.04],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 6
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.05],
            [0.0, 0.0, 1.0, -0.04],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 7
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.05],
            [0.0, 0.0, 1.0, -0.04],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 8
]
