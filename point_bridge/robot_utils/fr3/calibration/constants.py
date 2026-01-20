# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np

CAMERA_MATRICES = {
    "cam_8": {
        "left": np.array([[531.96, 0, 625.87], [0.0, 532.235, 345.2485], [0, 0, 1]]),
        "right": np.array([[533.475, 0, 637.67], [0.0, 533.81, 364.5395], [0, 0, 1]]),
    },
}

DISTORTION_COEFFICIENTS = {
    "cam_8": {
        "left": np.zeros((5)),
        "right": np.zeros((5)),
    },
}
