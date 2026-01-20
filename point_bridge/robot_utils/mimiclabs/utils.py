# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np
from scipy.spatial.transform import Rotation as R

# mapping from camera names to pixel keys
camera2pixelkey = {
    "agentviewleft": "pixels_left",
    "agentview": "pixels_right",
    "robot0_eye_in_hand": "pixels_egocentric",
    "cam_7": "pixels_cam_7",
}
pixelkey2camera = {v: k for k, v in camera2pixelkey.items()}

# Transform difference between mimiclabs and fr3
# FR3 robot base in mimiclabs robot base
T_robot_base = np.eye(4)
# FR3 gripper in mimiclabs gripper
T_gripper = np.eye(4)
T_gripper[:3, :3] = R.from_rotvec(
    np.array([0, 0, np.pi / 2])
).as_matrix()  # 90 degrees around z-axis


def depthimg2Meters(env, depth):
    extent = env.sim.model.stat.extent
    near = env.sim.model.vis.map.znear * extent
    far = env.sim.model.vis.map.zfar * extent
    image = near / (1 - depth * (1 - near / far))
    return image


############## Utils for many cams #########################


def normalize(v):
    """
    Normalize a vector to unit length.

    Args:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: Normalized vector.
    """
    return v / np.linalg.norm(v)


def look_at_matrix(camera_pos, target_pos, up=np.array([0, 0, 1])):
    """
    Create a 4x4 homogeneous transformation matrix representing a camera pose
    at `camera_pos` looking towards `target_pos` with `up` as the approximate up direction.

    Args:
        camera_pos (np.ndarray): The camera position (shape [3]).
        target_pos (np.ndarray): The look-at target position (shape [3]).
        up (np.ndarray, optional): The world up vector (shape [3]). Default is [0,0,1].

    Returns:
        np.ndarray: 4x4 transformation matrix for camera pose.
    """
    forward = normalize(target_pos - camera_pos)
    right = normalize(np.cross(up, forward))
    true_up = np.cross(forward, right)
    R = np.eye(4)
    R[:3, 0] = right
    R[:3, 1] = true_up
    R[:3, 2] = forward
    T = np.eye(4)
    T[:3, 3] = camera_pos
    return T @ R


def sample_cuboid_shell(
    inner_min,
    inner_max,
    outer_min,
    outer_max,
    nx=8,
    ny=8,
    nz=4,
    target=None,
    fps_func=None,
    reduce_to_num_poses=None,
):
    """
    Sample camera positions within a cuboid shell (region between an inner and outer cuboid),
    and orient each camera to look at a specified target.

    Args:
        inner_min (np.ndarray): Minimum bounds [x,y,z] of inner cuboid.
        inner_max (np.ndarray): Maximum bounds [x,y,z] of inner cuboid.
        outer_min (np.ndarray): Minimum bounds [x,y,z] of outer cuboid.
        outer_max (np.ndarray): Maximum bounds [x,y,z] of outer cuboid.
        nx (int): Number of samples along x.
        ny (int): Number of samples along y.
        nz (int): Number of samples along z.
        target (np.ndarray): Target position [x,y,z] for look-at.
        fps_func (function): Function to reduce the number of poses using Farthest Point Sampling.
        reduce_to_num_poses (int): Number of poses to reduce to.

    Returns:
        list[np.ndarray]: List of 4x4 camera pose matrices.
    """
    xs = np.linspace(outer_min[0], outer_max[0], nx)
    ys = np.linspace(outer_min[1], outer_max[1], ny)
    zs = np.linspace(outer_min[2], outer_max[2], nz)
    pos, poses = [], []
    for x in xs:
        for y in ys:
            for z in zs:
                # Exclude points inside the inner cuboid
                if not (
                    inner_min[0] < x < inner_max[0]
                    or inner_min[1] < y < inner_max[1]
                    or inner_min[2] < z < inner_max[2]
                ):
                    cam_pos = np.array([x, y, z])
                    pos.append(cam_pos)

    if fps_func is not None:
        assert (
            reduce_to_num_poses is not None
        ), "reduce_to_num_poses must be provided if fps_func is provided"
        pos = fps_func(pos, reduce_to_num_poses)

    for cam_pos in pos:
        pose = look_at_matrix(cam_pos, target)
        poses.append(pose)

    return poses


############################################################
