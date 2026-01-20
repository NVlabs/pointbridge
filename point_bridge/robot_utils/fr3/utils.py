# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# mapping from camera names to pixel keys
camera2pixelkey = {
    "cam_8_left": "pixels8_left",
    "cam_8_right": "pixels8_right",
}
pixelkey2camera = {v: k for k, v in camera2pixelkey.items()}
