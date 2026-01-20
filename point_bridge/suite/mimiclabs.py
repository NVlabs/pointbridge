# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import sys

from typing import Any, NamedTuple

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set MuJoCo GL backend to avoid EGL context cleanup issues
# Use glfw if available (more stable than EGL), otherwise fall back to osmesa for headless rendering
if "MUJOCO_GL" not in os.environ:
    # Try glfw first (more stable than EGL), fall back to osmesa for headless
    # This avoids EGL context cleanup errors during garbage collection
    import sys
    if sys.platform.startswith('linux'):
        # On Linux, prefer glfw if display is available, otherwise osmesa
        if 'DISPLAY' in os.environ:
            os.environ["MUJOCO_GL"] = "glfw"
        else:
            os.environ["MUJOCO_GL"] = "osmesa"
    else:
        # On other platforms, use glfw
        os.environ["MUJOCO_GL"] = "glfw"

import gym
from gym import spaces

import dm_env
from dm_env import StepType, specs, TimeStep

import cv2
import einops
import numpy as np
from pathlib import Path
from collections import deque
from scipy.spatial.transform import Rotation as R

import robosuite as suite
from mimiclabs.mimiclabs.envs.problems import *

from point_bridge.robot_utils.common.mujoco_transforms import MujocoTransforms
from point_bridge.robot_utils.common.utils import (
    rigid_transform_3D,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    transform_points,
    farthest_point_sampling,
    depthimg2Meters,
    pixel2d_to_3d,
)
from point_bridge.robot_utils.common.franka_gripper_points import extrapoints
from point_bridge.robot_utils.mimiclabs.utils import (
    pixelkey2camera,
    T_robot_base,
    T_gripper,
)
from point_bridge.robot_utils.common.camera_utils import (
    add_camera_with_offset,
    add_cameras_from_extrinsics,
)
from point_bridge.robot_utils.mimiclabs.sample_object_points import (
    extract_object_mesh,
    sample_points_from_mesh,
)
from sentence_transformers import SentenceTransformer

# for VLM points
from point_bridge.detection_utils.utils import init_models_for_vlm_detection
from point_bridge.robot_utils.common.vlm_detection import (
    get_vlm_points_using_segments_depth,
    get_vlm_points_using_point_tracking,
)

crop_h, crop_w = (0.35, 1.0), (0.0, 1.0)


def adjust_intrinsic_matrix_for_scale(intrinsic_matrix, scale_factor):
    """
    Adjust intrinsic matrix for image scaling.

    Args:
        intrinsic_matrix: 3x3 camera intrinsic matrix
        scale_factor: scaling factor (fx, fy) for x and y dimensions

    Returns:
        Adjusted 3x3 intrinsic matrix
    """
    # TODO: move this function to robot_utils.common.utils
    if isinstance(scale_factor, (list, tuple)):
        fx, fy = scale_factor
    elif isinstance(scale_factor, (int, float)):
        fx, fy = scale_factor, scale_factor
    else:
        fx, fy = scale_factor

    # Create scaling matrix
    scale_matrix = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])

    # Apply scaling to intrinsic matrix
    adjusted_intrinsic = scale_matrix @ intrinsic_matrix

    return adjusted_intrinsic


def generate_full_scene_pointcloud(
    depth_img,
    intrinsic_matrix,
    extrinsic_matrix,
    max_depth=2.0,
    min_z_robot=0.01,
    num_points=512,
):
    """
    Generate full scene point cloud from depth image.

    Args:
        depth_img: Depth image in meters (H, W)
        intrinsic_matrix: Camera intrinsic matrix (3, 3)
        extrinsic_matrix: Camera extrinsic matrix (4, 4) - robot base in camera frame
        max_depth: Maximum depth threshold in meters
        min_z_robot: Minimum z value in robot base frame
        num_points: Number of points to sample using FPS

    Returns:
        points_3d: Point cloud in robot base frame (num_points, 3)
    """
    h, w = depth_img.shape

    # Create pixel grid
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    u_coords = u_coords.flatten()
    v_coords = v_coords.flatten()
    depths = depth_img.flatten()

    # Filter by depth threshold in camera frame
    valid_depth_mask = (depths > 0) & (depths < max_depth)
    u_coords = u_coords[valid_depth_mask]
    v_coords = v_coords[valid_depth_mask]
    depths = depths[valid_depth_mask]

    if len(depths) == 0:
        return np.zeros((num_points, 3))

    # Convert to 3D points in camera frame
    pixel_coords = np.stack([u_coords, v_coords], axis=1)
    points_3d_cam = pixel2d_to_3d(
        pixel_coords,
        depths,
        intrinsic_matrix,
        np.eye(4),  # Identity since we want camera frame
    )

    # Transform to robot base frame
    points_3d_robot = transform_points(points_3d_cam, np.linalg.inv(extrinsic_matrix))

    # Filter by z threshold in robot base frame
    valid_z_mask = points_3d_robot[:, 2] > min_z_robot
    points_3d_robot = points_3d_robot[valid_z_mask]

    if len(points_3d_robot) == 0:
        return np.zeros((num_points, 3))

    # Apply Farthest Point Sampling to reduce to desired number of points
    if len(points_3d_robot) > num_points:
        points_3d_robot = farthest_point_sampling(points_3d_robot, num_points)
    elif len(points_3d_robot) < num_points:
        # If we have fewer points, pad with zeros or repeat
        padding = np.zeros((num_points - len(points_3d_robot), 3))
        points_3d_robot = np.vstack([points_3d_robot, padding])

    return points_3d_robot


def init_models():
    # load sentence transformer
    print("Initializing Sentence Transformer ...")
    lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    return lang_model


class RGBArrayAsObservationWrapper(dm_env.Environment):
    """
    Use env.render(rgb_array) as observation
    rather than the observation environment provides

    From: https://github.com/hill-a/stable-baselines/issues/915
    """

    def __init__(
        self,
        env,
        task_name,
        width=256,
        height=256,
        use_robot=False,
        max_episode_len=300,
        max_state_dim=100,
        pixel_keys=["pixels"],
        _pixelkey2camera=None,
        num_robot_points=8,
        num_points_per_obj=128,
        robot_points_key="robot_tracks",
        object_points_key="object_tracks",
        obs_type=["image"],
        action_mode="pose",
        use_vlm_points=False,  # parameter to choose between GT and VLM points
        vlm_mode="segment_depth",  # segment_depth, point_tracking
        sam_predictor=None,
        gemini_client=None,
        molmo_client=None,
        lang_model=None,
        mast3r_model=None,
        dust3r_inference=None,
        depth_type="gt",
        depth_model=None,
        add_camera_from_extrinsics=False,
        camera_extrinsics_file=None,
        temporal_agg_strategy=None,
        visualize_3d=False,
        viser_server=None,
        use_full_scene_pcd=False,
        full_scene_num_points=512,
        max_depth_meters=2.0,
        min_z_robot_frame=-0.03,
        full_scene_camera_name=None,
    ):
        self._env = env
        self._task_name = task_name
        self._height, self._width = height, width
        self.use_robot = use_robot
        self._max_episode_len = max_episode_len
        self._max_state_dim = max_state_dim
        self._pixel_keys = pixel_keys
        self._pixelkey2camera = (
            {**pixelkey2camera, **_pixelkey2camera}
            if _pixelkey2camera is not None
            else pixelkey2camera
        )
        self._device = "cpu"
        self._obs_type = obs_type
        self._action_mode = action_mode
        self.depth_type = depth_type
        self.add_camera_from_extrinsics = add_camera_from_extrinsics
        self.camera_extrinsics_file = camera_extrinsics_file
        self.visualize_3d = visualize_3d
        # full scene PCD mode
        self.use_full_scene_pcd = use_full_scene_pcd
        self.full_scene_num_points = full_scene_num_points
        self.max_depth_meters = max_depth_meters
        self.min_z_robot_frame = min_z_robot_frame
        self.full_scene_camera_name = full_scene_camera_name

        # track vars
        self._num_robot_points = num_robot_points
        self._num_points_per_obj = num_points_per_obj

        # points params
        self._robot_points_key = robot_points_key
        self._object_points_key = f"{object_points_key}_{self._num_points_per_obj}"
        self._sample_points_per_obj = 1000

        # VLM points configuration
        self.use_vlm_points = use_vlm_points
        self.vlm_mode = vlm_mode
        self.vlm_tracker = None  # Will store the segment tracker for tracking masks
        self.vlm_objects = None  # Will store detected object names

        # VLM models
        (
            self.sam_predictor,
            self.gemini_client,
            self.molmo_client,
            self.lang_model,
            self.mast3r_model,
            self.dust3r_inference,
        ) = (
            sam_predictor,
            gemini_client,
            molmo_client,
            lang_model,
            mast3r_model,
            dust3r_inference,
        )

        self.task_description = self._env.language_instruction
        self.task_emb = self.lang_model.encode(self._env.language_instruction)

        # camera names
        self._camera_names = [
            self._pixelkey2camera[pixel_key] for pixel_key in pixel_keys
        ]

        # robot base pose
        robot_base_pos = self._env.sim.data.get_body_xpos("robot0_base")
        robot_base_ori = self._env.sim.data.get_body_xmat("robot0_base")
        self.robot_base = np.eye(4)
        self.robot_base[:3, 3] = robot_base_pos
        self.robot_base[:3, :3] = robot_base_ori
        self.robot_base = (
            self.robot_base @ T_robot_base
        )  # FR3 robot base in world frame

        # For foundation_stereo, we need to set the height and width to 448x672
        # since the TRT optimized model is only available for 448x672.
        # Otherwise resizing depth leads to artifacts in resized depth.
        if self.depth_type == "foundation_stereo":
            self._height, self._width = 448, 672

        # calibration data
        if "points" in self._obs_type:
            self.intr, self.extr, self.camera_projections = {}, {}, {}
            transforms = MujocoTransforms(
                env, self._camera_names, self._height, self._width
            )
            transformation_matrices = transforms.transforms
            camera_poses = transformation_matrices[
                "camera_projection_matrix"
            ]  # camera in world
            self.intr = transforms.camera_intrinsics

            for camera_name in self._camera_names:
                self.extr[camera_name] = (
                    np.linalg.inv(camera_poses[camera_name]) @ self.robot_base
                )  # FR3 robot base in camera frame
                intr = self.intr[camera_name]
                intr = np.column_stack((intr, np.zeros((3, 1))))
                self.camera_projections[camera_name] = intr @ self.extr[camera_name]

            if "foundation_stereo" in self.depth_type:
                self.fs_client = depth_model
                H, W = 448, 672
                self.scale_factor_fs = (W / self._width, H / self._height)
                self.adjusted_camera_intrinsics = {}
                for pixel_key in self._pixel_keys:
                    camera_name = self._pixelkey2camera[pixel_key]
                    self.adjusted_camera_intrinsics[
                        camera_name
                    ] = adjust_intrinsic_matrix_for_scale(
                        self.intr[camera_name].copy(), self.scale_factor_fs
                    )

        # eef bodyname to compute robot points
        self._bodynames = ["gripper0_eef"]  # end effector pose

        # obs = self._env.reset()
        pixel_shape = (self._height, self._width, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=pixel_shape, dtype=np.uint8
        )

        # Action spec
        self.action_shape = (10,)  # only absolute actions - (3d pos, 6d ori, gripper)
        self.action_space = specs.BoundedArray(
            self.action_shape, np.float32, -np.inf, np.inf, "action"
        )

        # Observation spec
        proprio_shape = (10,)  # (3d pos, 6d ori, gripper)
        self._obs_spec = {}
        for pixel_key in self._pixel_keys:
            self._obs_spec[pixel_key] = specs.BoundedArray(
                shape=pixel_shape,
                dtype=np.uint8,
                minimum=0,
                maximum=255,
                name=pixel_key,
            )
        self._obs_spec["proprioceptive"] = specs.BoundedArray(
            shape=proprio_shape,
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="proprioceptive",
        )
        self._obs_spec["features"] = specs.BoundedArray(
            shape=proprio_shape,
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="features",
        )

        if "points" in self._obs_type:
            # object names
            self._object_names = {}
            for obj in self._env.obj_of_interest:
                self._object_names[obj] = []
                for body_name in self._env.sim.model.body_names:
                    if obj not in body_name:
                        continue
                    try:
                        mesh = extract_object_mesh(
                            self._env, body_name, visualize=False, in_base_frame=True
                        )
                        if mesh is not None:
                            self._object_names[obj].append(body_name)
                    except:
                        pass

        # visualize 3d with viser
        if self.visualize_3d:
            self.server = viser_server

        # Track if environment has been closed to prevent multiple close calls
        self._closed = False

    def reset(self, **kwargs):
        self._step = 0

        obs = self._env.reset()
        if "state" in kwargs:
            state = kwargs["state"]
            self._env.sim.set_state_from_flattened(state)
            self._env.sim.forward()

        # Add new camera after reset using utility function
        state = self._env.sim.get_state().flatten()
        camera_name = "agentviewleft"
        offset = np.array([0, -0.12, 0])  # Offset from agentview camera
        success = add_camera_with_offset(self._env, camera_name, "agentview", offset)
        if success:
            # Reset to the same state after adding camera
            self._env.sim.reset()
            self._env.sim.set_state_from_flattened(state)
            self._env.sim.forward()

        # if use camera from extrinsics, add camera
        # camera get set to only the original ones after reset so need to reinitialize
        if self.add_camera_from_extrinsics:
            self._env, _, _pixelkey2camera = add_cameras_from_extrinsics(
                self._env, self.camera_extrinsics_file, T_robot_base
            )
            for _, cam_name in _pixelkey2camera.items():
                if cam_name not in self._camera_names:
                    self._camera_names.append(cam_name)
            # calibration data
            if "points" in self._obs_type:
                self.intr, self.extr, self.camera_projections = {}, {}, {}
                transforms = MujocoTransforms(
                    self._env, self._camera_names, self._height, self._width
                )
                transformation_matrices = transforms.transforms
                camera_poses = transformation_matrices[
                    "camera_projection_matrix"
                ]  # camera in world
                self.intr = transforms.camera_intrinsics

                for camera_name in self._camera_names:
                    self.extr[camera_name] = (
                        np.linalg.inv(camera_poses[camera_name]) @ self.robot_base
                    )  # FR3 robot base in camera frame
                    intr = self.intr[camera_name]
                    intr = np.column_stack((intr, np.zeros((3, 1))))
                    self.camera_projections[camera_name] = intr @ self.extr[camera_name]

        self.fixed_points = False
        if "object_points" in kwargs and "points" in self._obs_type:
            self.object_points = kwargs["object_points"]
            self.fixed_points = True
        self.robot_actions = None
        self.prev_gripper_state = -1  # Default open gripper

        if "points" in self._obs_type:
            pos = self._env.sim.data.get_body_xpos(self._bodynames[0])
            ori = self._env.sim.data.get_body_xmat(self._bodynames[0])
            ori = R.from_matrix(ori).as_rotvec()  # in world frame
            dummy_action = np.concatenate([pos, ori, [-1]])
            for _ in range(10):
                obs, _, _, _ = self._env.step(dummy_action)

        observation = {}
        observation["task_emb"] = self.task_emb
        observation["goal_achieved"] = False

        for key in self._pixel_keys:
            camera_name = self._pixelkey2camera[key]
            observation[key] = self._env.sim.render(
                self._width, self._height, camera_name=camera_name
            )[::-1]

        # compute current pose in robot base frame
        pos = self._env.sim.data.get_body_xpos(self._bodynames[0])
        ori = self._env.sim.data.get_body_xmat(self._bodynames[0])  # in world frame
        # keep proprio in robot base frame
        T = np.eye(4)
        T[:3, 3], T[:3, :3] = pos, ori
        T = T @ T_gripper  # FR3 gripper in world frame
        T = np.linalg.inv(self.robot_base) @ T  # FR3 gripper in FR3 robot base frame
        # store
        pos, ori = T[:3, 3], T[:3, :3]
        self.robot_base_orientation = ori
        ori = matrix_to_rotation_6d(ori)
        self._current_pose = np.concatenate(
            [pos, ori, [self.prev_gripper_state]]
        )  # gripper in robot base frame
        observation["proprioceptive"] = self._current_pose
        observation["features"] = self._current_pose

        if "points" in self._obs_type:
            # get robot and object points in robot base frame
            if not self.fixed_points:
                self.object_points = {}

            if self.use_full_scene_pcd:
                points3d_robot, points3d_objects = self.get_full_scene_pcd()
            elif self.use_vlm_points:
                if self.vlm_mode == "segment_depth":
                    vlm_func = self.get_vlm_points_using_segments_depth
                elif self.vlm_mode == "point_tracking":
                    vlm_func = self.get_vlm_points_using_point_tracking
                points3d_robot, points3d_objects = vlm_func()
            else:
                points3d_robot, points3d_objects = self.get_gt_points()

            # base robot configuration
            self.base_robot_points = np.array(points3d_robot)

            # for pixel_key in self._pixel_keys:
            observation[f"{self._robot_points_key}_3d"] = points3d_robot
            observation[f"{self._object_points_key}_3d"] = points3d_objects

            # visualize 3d with viser
            if self.visualize_3d:
                self.server.scene.add_point_cloud(
                    name="/robot_points",
                    points=points3d_robot,
                    colors=(255, 0, 0),
                    point_size=0.005,
                )
                self.server.scene.add_point_cloud(
                    name="/object_points",
                    points=einops.rearrange(points3d_objects, "n o d -> (n o) d"),
                    colors=(0, 255, 0),
                    point_size=0.01,
                )

        self.observation = observation
        self.prev_gripper = deque(maxlen=5)
        return observation

    def step(self, action):
        for _ in range(1):
            if self._action_mode == "points":
                action_shape_dim = len(action.shape)
                new_action = {
                    "future_tracks": action[..., :-3],
                    "gripper": action[..., -3:],
                }
                action = new_action

                if action_shape_dim == 2:
                    robot_actions = self.point2action_chunked(action)

                    num_actions_execute = 5
                    self.robot_actions = robot_actions[:num_actions_execute]
                    robot_action = self.robot_actions

                elif action_shape_dim == 1:
                    robot_action = self.point2action(action)
                    robot_action = np.array([robot_action])

            elif self._action_mode == "pose":
                if len(action.shape) == 1:
                    robot_action = np.array([action])
                else:
                    robot_action = action  # [:5]

            env_actions = []
            for action_i in robot_action:
                # compute gripper state
                self.prev_gripper.append(action_i[-1])
                self.prev_gripper_state = action_i[-1]

                if self._action_mode == "pose":
                    pos, ori, gripper = action_i[:3], action_i[3:9], action_i[-1:]
                    #  convert to world frame
                    ori = rotation_6d_to_matrix(ori)  # in FR3 robot base frame
                    T = np.eye(4)
                    T[:3, :3], T[:3, 3] = ori, pos
                    T = self.robot_base @ T  # FR3 gripper in world frame
                    T = T @ np.linalg.inv(T_gripper)  # gripper in world frame
                    # store
                    pos, ori = T[:3, 3], T[:3, :3]
                    ori = R.from_matrix(ori).as_rotvec()
                    action_i = np.concatenate([pos, ori, gripper])

                env_actions.append(action_i)

            for action_i in env_actions:
                self._step += 1
                obs, reward, done, info = self._env.step(action_i)
                if self._step >= self._max_episode_len:
                    break
            if self._step >= self._max_episode_len:
                break

        observation = {}
        observation["task_emb"] = self.task_emb
        observation["goal_achieved"] = done

        for key in self._pixel_keys:
            camera_name = self._pixelkey2camera[key]
            observation[key] = self._env.sim.render(
                self._width, self._height, camera_name=camera_name
            )[::-1]

        # compute current pose in robot base frame
        pos = self._env.sim.data.get_body_xpos(self._bodynames[0])
        ori = self._env.sim.data.get_body_xmat(self._bodynames[0])  # in world frame
        # keep proprio in robot base frame
        T = np.eye(4)
        T[:3, 3], T[:3, :3] = pos, ori
        T = T @ T_gripper  # FR3 gripper in world frame
        T = np.linalg.inv(self.robot_base) @ T  # FR3 gripper in FR3 robot base frame
        pos, ori = T[:3, 3], T[:3, :3]
        # store
        ori = matrix_to_rotation_6d(ori)
        self._current_pose = np.concatenate([pos, ori, [self.prev_gripper_state]])
        observation["proprioceptive"] = self._current_pose
        observation["features"] = self._current_pose

        if "points" in self._obs_type:
            # get robot and object points in robot base frame
            if self.use_full_scene_pcd:
                points3d_robot, points3d_objects = self.get_full_scene_pcd()
            elif self.use_vlm_points:
                if self.vlm_mode == "segment_depth":
                    vlm_func = self.get_vlm_points_using_segments_depth
                elif self.vlm_mode == "point_tracking":
                    vlm_func = self.get_vlm_points_using_point_tracking
                points3d_robot, points3d_objects = vlm_func()
            else:
                points3d_robot, points3d_objects = self.get_gt_points()

            observation[f"{self._robot_points_key}_3d"] = points3d_robot
            observation[f"{self._object_points_key}_3d"] = points3d_objects

            # visualize 3d with viser
            if self.visualize_3d:
                self.server.scene.add_point_cloud(
                    name="/robot_points",
                    points=points3d_robot,
                    colors=(255, 0, 0),
                    point_size=0.005,
                )
                self.server.scene.add_point_cloud(
                    name="/object_points",
                    points=einops.rearrange(points3d_objects, "n o d -> (n o) d"),
                    colors=(0, 255, 0),
                    point_size=0.002,
                )

        if self._step >= self._max_episode_len:
            done = True
        done = done | observation["goal_achieved"]

        self.observation = observation

        return observation, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode="rgb_array", width=256, height=256):
        return self._env.sim.render(width, height, camera_name="agentview")[::-1]

    def get_gt_points(self):
        # get 3d robot points
        eef_pos, eef_ori = self._current_pose[:3], self._current_pose[3:9]
        eef_ori = rotation_6d_to_matrix(eef_ori)
        T_g2b = np.eye(4)  # FR3 gripper in FR3 robot base
        T_g2b[:3, :3] = eef_ori
        T_g2b[:3, 3] = eef_pos

        # add point transform
        points3d_robot = []
        gripper_state = self._current_pose[-1]
        for idx, Tp in enumerate(extrapoints):
            if gripper_state > 0 and idx in [0, 1]:
                Tp = Tp.copy()
                Tp[1, 3] = 0.015 if idx == 0 else -0.015
            pt = T_g2b @ Tp  # pt in FR3 robot base
            pt = pt[:3, 3]
            points3d_robot.append(pt[:3])
        points3d_robot = np.array(points3d_robot)  # (N, D)

        # get 3d points for each object
        points3d_objects = []
        for object_name in self._object_names:
            body_names = self._object_names[object_name]
            num_pts = [self._sample_points_per_obj // len(body_names)] * len(body_names)
            save_num_pts = [self._num_points_per_obj // len(body_names)] * len(
                body_names
            )
            if sum(num_pts) < self._sample_points_per_obj:
                num_pts[-1] += self._sample_points_per_obj - sum(num_pts)
            if (
                self._num_points_per_obj > 0
                and sum(save_num_pts) < self._num_points_per_obj
            ):
                save_num_pts[-1] += self._num_points_per_obj - sum(save_num_pts)

            body_points = []
            for body_name, num_pt, save_num_pt in zip(
                body_names, num_pts, save_num_pts
            ):
                # object pose in world frame
                obj_pos, obj_ori = self._env.sim.data.get_body_xpos(
                    body_name
                ), self._env.sim.data.get_body_xmat(body_name)
                Tobj = np.eye(4)
                Tobj[:3, :3], Tobj[:3, 3] = obj_ori, obj_pos
                Tobj = (
                    np.linalg.inv(self.robot_base) @ Tobj
                )  # object in FR3 robot base frame

                if self._step == 0 and not self.fixed_points:
                    mesh = extract_object_mesh(
                        self._env, body_name, visualize=False, in_base_frame=True
                    )
                    points = sample_points_from_mesh(
                        mesh, n_points=num_pt, surface_only=True
                    )  # in robot base frame

                    # reduce points using Farthest Point Sampling
                    points = farthest_point_sampling(points, save_num_pt)

                    # compute points in FR3 robot base frame
                    points = transform_points(points, np.linalg.inv(T_robot_base))

                    # points in object frame
                    pts = transform_points(points, np.linalg.inv(Tobj))
                    self.object_points[body_name] = pts
                else:
                    pts = self.object_points[body_name]  # in object frame
                    points = transform_points(pts, Tobj)  # in FR3 robot base frame
                body_points.append(points[:, :3])

            body_points = np.concatenate(body_points, axis=0)
            points3d_objects.append(body_points)
        points3d_objects = np.array(points3d_objects)  # (N, O, D)

        return points3d_robot, points3d_objects

    def get_vlm_points_using_segments_depth(self):
        """
        Get VLM-based object masks and 3D points using depth information.
        Generates both image and ground truth depth for the first camera only,
        runs detect_segments to get object masks, randomly samples 1000 points
        from each mask, projects to 3D using depth and camera parameters,
        and applies farthest point sampling to reduce to desired number of points.
        """

        # Get current image and depth from the first camera only
        ref_pixel_key = self._pixel_keys[0]
        camera_name = self._pixelkey2camera[ref_pixel_key]

        # Get camera parameters for the reference camera
        intrinsic_matrix = self.intr[camera_name]
        extrinsic_matrix = self.extr[camera_name]  # FR3 robot base in camera frame

        # Render both image and depth for the reference camera
        if self.depth_type == "gt":
            ref_image, ref_depth = self._env.sim.render(
                self._width, self._height, camera_name=camera_name, depth=True
            )
            ref_image = ref_image[::-1]  # Flip vertically
            ref_depth = ref_depth[::-1]  # Flip vertically

            # Convert depth to meters
            ref_depth = depthimg2Meters(self._env, ref_depth)

        elif "foundation_stereo" in self.depth_type:
            adjusted_intrinsic_matrix = self.adjusted_camera_intrinsics[camera_name]
            img0 = self._env.sim.render(
                int(self._width * self.scale_factor_fs[1]),
                int(self._height * self.scale_factor_fs[0]),
                camera_name=self._pixelkey2camera[ref_pixel_key],
            )[::-1]
            img1 = self._env.sim.render(
                int(self._width * self.scale_factor_fs[1]),
                int(self._height * self.scale_factor_fs[0]),
                camera_name=self._pixelkey2camera[self._pixel_keys[1]],
            )[::-1]
            ref_depth = self.fs_client.get_depth_map(
                img0, img1, adjusted_intrinsic_matrix
            )
            ref_depth = cv2.resize(
                ref_depth, (self._width, self._height), interpolation=cv2.INTER_LINEAR
            )

            ref_image = self._env.sim.render(
                self._width,
                self._height,
                camera_name=self._pixelkey2camera[ref_pixel_key],
            )[::-1]

        # Initialize points2d and depths storage if not exists
        if not hasattr(self, "points2d"):
            self.points2d = {}
        if not hasattr(self, "depths"):
            self.depths = {}

        # Call the extracted function
        (
            points3d_robot,
            points3d_objects,
            self.vlm_tracker,
            self.vlm_objects,
            self.points2d,
            self.depths,
        ) = get_vlm_points_using_segments_depth(
            ref_image=ref_image,
            ref_depth=ref_depth,
            task_description=self.task_description,
            vlm_tracker=self.vlm_tracker,
            vlm_objects=self.vlm_objects,
            is_first_step=(self._step == 0),
            num_points_per_obj=self._num_points_per_obj,
            current_pose=self._current_pose,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            sam_predictor=self.sam_predictor,
            gemini_client=self.gemini_client,
            molmo_client=self.molmo_client,
            task_name=self._task_name,
            height_limits=crop_h,
            width_limits=crop_w,
            points2d=self.points2d,
            depths=self.depths,
            use_noised_foreground_points=False,
        )

        return points3d_robot, points3d_objects

    def get_vlm_points_using_point_tracking(self):
        """
        Get VLM-based object points using point tracking.
        Uses detect_segments to obtain object masks during reset (self._step == 0),
        then uses PointTracker to track points across frames.
        Samples points from masks, computes multiview correspondence, and triangulates to get 3D points.
        """

        # Get current images from all cameras
        pixel_key_images = {}
        for pixel_key in self._pixel_keys:
            camera_name = self._pixelkey2camera[pixel_key]
            image = self._env.sim.render(
                self._width, self._height, camera_name=camera_name
            )[::-1]
            pixel_key_images[pixel_key] = image

        # Initialize point trackers if not exists
        if not hasattr(self, "point_trackers"):
            self.point_trackers = None

        # Call the centralized function
        (
            points3d_robot,
            points3d_objects,
            self.vlm_objects,
            self.point_trackers,
        ) = get_vlm_points_using_point_tracking(
            pixel_key_images=pixel_key_images,
            task_description=self.task_description,
            pixel_keys=self._pixel_keys,
            camera_projections=self.camera_projections,
            vlm_objects=self.vlm_objects,
            is_first_step=(self._step == 0),
            num_points_per_obj=self._num_points_per_obj,
            current_pose=self._current_pose,
            sam_predictor=self.sam_predictor,
            gemini_client=self.gemini_client,
            molmo_client=self.molmo_client,
            mast3r_model=self.mast3r_model,
            dust3r_inference=self.dust3r_inference,
            task_name=self._task_name,
            height_limits=crop_h,
            width_limits=crop_w,
            point_trackers=self.point_trackers,
            pixelkey2camera=self._pixelkey2camera,
        )

        return points3d_robot, points3d_objects

    def get_full_scene_pcd(self):
        """
        Generate full scene point cloud from depth image using specified camera.
        Returns robot points and scene point cloud as a single "object" for compatibility.
        """
        # Get robot points
        eef_pos, eef_ori = self._current_pose[:3], self._current_pose[3:9]
        eef_ori = rotation_6d_to_matrix(eef_ori)
        T_g2b = np.eye(4)  # FR3 gripper in FR3 robot base
        T_g2b[:3, :3] = eef_ori
        T_g2b[:3, 3] = eef_pos

        # add point transform
        points3d_robot = []
        gripper_state = self._current_pose[-1]
        for idx, Tp in enumerate(extrapoints):
            if gripper_state > 0 and idx in [0, 1]:
                Tp = Tp.copy()
                Tp[1, 3] = 0.015 if idx == 0 else -0.015
            pt = T_g2b @ Tp  # pt in FR3 robot base
            pt = pt[:3, 3]
            points3d_robot.append(pt[:3])
        points3d_robot = np.array(points3d_robot)  # (N, D)

        # Determine which camera to use for depth
        camera_name = self.full_scene_camera_name
        if camera_name is None:
            # Use the last camera by default (typically the real camera)
            camera_name = self._camera_names[-1]

        # Get depth image
        if self.depth_type == "gt":
            _, depth_img = self._env.sim.render(
                self._width, self._height, camera_name=camera_name, depth=True
            )
            depth_img = depth_img[::-1]  # Flip vertically
            depth_img = depthimg2Meters(self._env, depth_img)

        elif "foundation_stereo" in self.depth_type:
            adjusted_intrinsic_matrix = self.adjusted_camera_intrinsics[camera_name]
            img0 = self._env.sim.render(
                int(self._width * self.scale_factor_fs[1]),
                int(self._height * self.scale_factor_fs[0]),
                camera_name=camera_name,
            )[::-1]
            # Get second camera for stereo
            camera_name_stereo = self._camera_names[min(1, len(self._camera_names) - 1)]
            img1 = self._env.sim.render(
                int(self._width * self.scale_factor_fs[1]),
                int(self._height * self.scale_factor_fs[0]),
                camera_name=camera_name_stereo,
            )[::-1]
            depth_img = self.fs_client.get_depth_map(
                img0, img1, adjusted_intrinsic_matrix
            )
            depth_img = cv2.resize(
                depth_img, (self._width, self._height), interpolation=cv2.INTER_LINEAR
            )

        # Get camera parameters
        intrinsic_matrix = self.intr[camera_name]
        extrinsic_matrix = self.extr[camera_name]  # FR3 robot base in camera frame

        # Generate full scene point cloud
        points3d_scene = generate_full_scene_pointcloud(
            depth_img,
            intrinsic_matrix,
            extrinsic_matrix,
            max_depth=self.max_depth_meters,
            min_z_robot=self.min_z_robot_frame,
            num_points=self.full_scene_num_points,
        )

        # Store as a single "object" for compatibility with existing structure
        points3d_objects = np.array([points3d_scene])

        return points3d_robot, points3d_objects

    def point2action(self, action):
        """
        Action is a dict with 10 points corresponding to each camera frame.
        """
        points3d = einops.rearrange(action["future_tracks"], "(n d) -> n d", d=3)

        robot_pos, ori = self.compute_action_from_3dpoints(points3d)
        gripper_state = self.compute_gripper(action)
        robot_action = self.compute_robot_action(robot_pos, ori, gripper_state)

        return robot_action

    def point2action_chunked(self, action):
        num_actions = action["future_tracks"].shape[1]
        actions = []
        for i in range(num_actions):
            act = {}
            act["future_tracks"] = action["future_tracks"][:, i]
            act["gripper"] = action["gripper"][i : i + 1]
            actions.append(self.point2action(act))
        return np.array(actions)

    def compute_action_from_3dpoints(self, points3d):
        robot_pos = (points3d[0, :3] + points3d[1, :3]) / 2  # in FR3 robot base frame
        ori, _ = rigid_transform_3D(self.base_robot_points, points3d)
        ori = ori @ self.robot_base_orientation  # in FR3 robot base frame

        return robot_pos, ori

    def compute_gripper(self, action):
        gripper_state = np.mean(action["gripper"])
        gripper_state = np.clip(gripper_state, -1, 1)
        gripper_state = np.array([gripper_state])
        return gripper_state

    def compute_robot_action(self, target_position, target_orientation, gripper):
        """
        Return absolute actions
        """
        T_target = np.eye(4)  # FR3 gripper in FR3 robot base
        T_target[:3, :3] = target_orientation
        T_target[:3, 3] = target_position

        # switch back to sim frame
        T_target = T_target @ np.linalg.inv(
            T_gripper
        )  # gripper in FR3 robot base frame

        T_eef = (
            self.robot_base @ T_target
        )  # in world frame (since robosuite does absolute OSC_POSE control in the world frame)

        target_position = T_eef[:3, 3]
        target_orientation = T_eef[:3, :3]
        target_orientation = R.from_matrix(target_orientation).as_rotvec()

        return np.concatenate([target_position, target_orientation, gripper])

    def __getattr__(self, name):
        return getattr(self._env, name)

    def close(self):
        """Properly close the environment and cleanup resources."""
        if not self._closed:
            try:
                # Get the actual robosuite environment (might be nested)
                env_to_close = self._env
                if hasattr(self._env, 'env'):
                    env_to_close = self._env.env
                
                # First, try to close renderer explicitly if method exists
                # This is the recommended way to clean up robosuite renderers
                if hasattr(env_to_close, 'close_renderer'):
                    try:
                        env_to_close.close_renderer()
                    except Exception:
                        pass
                
                # Then, try to close the renderer context explicitly to avoid EGL errors
                # This must be done before closing the environment
                if hasattr(env_to_close, 'sim'):
                    sim = env_to_close.sim
                    # Close renderer contexts if they exist
                    # Access them safely and free them in reverse order
                    if hasattr(sim, '_render_contexts') and sim._render_contexts:
                        # Make a copy of the list to iterate over safely
                        render_contexts = list(sim._render_contexts)
                        # Clear the list first to prevent double cleanup during garbage collection
                        sim._render_contexts.clear()
                        for render_context in reversed(render_contexts):
                            try:
                                # Check if context has required attributes before freeing
                                if hasattr(render_context, 'free'):
                                    # Additional safety check for EGL contexts
                                    if hasattr(render_context, 'con') or hasattr(render_context, '_context'):
                                        render_context.free()
                            except Exception:
                                # Ignore errors during cleanup - EGL context may already be destroyed
                                pass
                    # Also try to close renderer directly if it exists
                    if hasattr(sim, 'renderer') and sim.renderer is not None:
                        try:
                            if hasattr(sim.renderer, 'close'):
                                sim.renderer.close()
                            elif hasattr(sim.renderer, 'free'):
                                sim.renderer.free()
                        except Exception:
                            # Ignore errors during cleanup
                            pass
                        # Clear renderer reference to prevent double cleanup
                        sim.renderer = None
                
                # Close the underlying robosuite environment
                if hasattr(env_to_close, 'close'):
                    env_to_close.close()
                
                # Also try to close the wrapper if it exists and is different
                if hasattr(self._env, 'close') and self._env != env_to_close:
                    try:
                        self._env.close()
                    except Exception:
                        pass
                
                # Clear any render context references that might cause issues during GC
                if hasattr(env_to_close, '_render_context_offscreen'):
                    try:
                        delattr(env_to_close, '_render_context_offscreen')
                    except Exception:
                        pass
                if hasattr(env_to_close, '_render_context_on_screen'):
                    try:
                        delattr(env_to_close, '_render_context_on_screen')
                    except Exception:
                        pass
                    
            except Exception as e:
                # Silently ignore cleanup errors during shutdown
                # EGL context may already be destroyed
                pass
            finally:
                self._closed = True

    def __del__(self):
        """Cleanup during garbage collection."""
        if not self._closed:
            self.close()


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats
        self._closed = False

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def close(self):
        """Close the wrapped environment if not already closed."""
        if not self._closed:
            if hasattr(self._env, 'close'):
                self._env.close()
            self._closed = True

    def __del__(self):
        """Cleanup during garbage collection."""
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, pixel_keys, num_frames):
        self._env = env
        self._num_frames = num_frames
        self._closed = False

        self.pixel_keys = pixel_keys
        wrapped_obs_spec = env.observation_spec()[self.pixel_keys[0]]

        # frames lists
        self._frames = {}
        for key in self.pixel_keys:
            self._frames[key] = deque([], maxlen=num_frames)

        pixels_shape = wrapped_obs_spec.shape
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = env.observation_spec()  # {}
        for key in self.pixel_keys:
            self._obs_spec[key] = specs.BoundedArray(
                shape=np.concatenate(
                    [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
                ),
                dtype=np.uint8,
                minimum=0,
                maximum=255,
                name=key,
            )

    def _transform_observation(self, time_step):
        for key in self.pixel_keys:
            assert len(self._frames[key]) == self._num_frames
        obs = {}
        for key in time_step.observation:
            if key in self.pixel_keys:
                obs[key] = np.concatenate(list(self._frames[key]), axis=0)
            else:
                obs[key] = time_step.observation[key]
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = {}
        for key in self.pixel_keys:
            pixels[key] = time_step.observation[key]
            if len(pixels[key].shape) == 4:
                pixels[key] = pixels[key][0]
            pixels[key] = pixels[key].transpose(2, 0, 1)
        return pixels

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        pixels = self._extract_pixels(time_step)
        for key in self.pixel_keys:
            for _ in range(self._num_frames):
                self._frames[key].append(pixels[key])
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        for key in self.pixel_keys:
            self._frames[key].append(pixels[key])
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def close(self):
        """Close the wrapped environment if not already closed."""
        if not self._closed:
            if hasattr(self._env, 'close'):
                self._env.close()
            self._closed = True

    def __del__(self):
        """Cleanup during garbage collection."""
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._discount = 1.0
        self._closed = False

        # Action spec
        wrapped_action_spec = env.action_space
        if not hasattr(wrapped_action_spec, "minimum"):
            wrapped_action_spec.minimum = -np.ones(wrapped_action_spec.shape)
        if not hasattr(wrapped_action_spec, "maximum"):
            wrapped_action_spec.maximum = np.ones(wrapped_action_spec.shape)
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            np.float32,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        step_type = StepType.LAST if done else StepType.MID

        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

    def point2action(self, action):
        return self._env.point2action(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        return TimeStep(
            step_type=StepType.FIRST, reward=0, discount=self._discount, observation=obs
        )

    def __getattr__(self, name):
        return getattr(self._env, name)

    def close(self):
        """Close the wrapped environment if not already closed."""
        if not self._closed:
            if hasattr(self._env, 'close'):
                self._env.close()
            self._closed = True

    def __del__(self):
        """Cleanup during garbage collection."""
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self._closed = False

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def _replace(
        self, time_step, observation=None, action=None, reward=None, discount=None
    ):
        if observation is None:
            observation = time_step.observation
        if action is None:
            action = time_step.action
        if reward is None:
            reward = time_step.reward
        if discount is None:
            discount = time_step.discount
        return ExtendedTimeStep(
            observation=observation,
            step_type=time_step.step_type,
            action=action,
            reward=reward,
            discount=discount,
        )

    def point2action(self, action):
        return self._env.point2action(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def close(self):
        """Close the wrapped environment if not already closed."""
        if not self._closed:
            if hasattr(self._env, 'close'):
                self._env.close()
            self._closed = True

    def __del__(self):
        """Cleanup during garbage collection."""
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass


def make(
    bddl_dir,
    # task_name,
    task_names,
    action_repeat,
    height,
    width,
    seed,
    max_episode_len,
    max_state_dim,
    eval,  # True means use_robot=True
    pixel_keys,
    num_robot_points,
    num_points_per_obj,
    robot_points_key,
    object_points_key,
    obs_type,
    action_mode,
    use_vlm_points,
    vlm_mode,
    depth_type,
    add_camera_from_extrinsics,
    camera_extrinsics_file,
    temporal_agg_strategy,
    visualize_3d,
    use_full_scene_pcd,
    full_scene_num_points,
    max_depth_meters,
    min_z_robot_frame,
    full_scene_camera_name,
):
    if use_vlm_points:
        print("Initializing VLM models...")
        (
            sam_predictor,
            gemini_client,
            molmo_client,
            lang_model,
            mast3r_model,
            dust3r_inference,
        ) = init_models_for_vlm_detection()
    else:
        sam_predictor, gemini_client, molmo_client, mast3r_model, dust3r_inference = (
            None,
            None,
            None,
            None,
            None,
        )
        lang_model = init_models()

    if "foundation_stereo" in depth_type:
        from model_servers.foundation_stereo.client import FoundationStereoZMQClient

        depth_model = FoundationStereoZMQClient(server_address="localhost:60000")
    else:
        depth_model = None

    if visualize_3d:
        import viser

        viser_server = viser.ViserServer(port=2353)
    else:
        viser_server = None

    envs, task_descriptions = [], []
    idx2name = {}
    for idx, task_name in enumerate(task_names):
        # init env
        controller_configs = suite.load_controller_config(default_controller="OSC_POSE")
        controller_configs[
            "control_delta"
        ] = False  # only absolute actions are supported
        env_kwargs = dict(
            env_name="MimicLabs_Lab1_Tabletop_Manipulation",
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=False,
            control_freq=20,
            controller_configs=controller_configs,
            robots=["Panda"],
            bddl_file_name=os.path.join(bddl_dir, task_name + ".bddl"),
        )
        env = suite.make(**env_kwargs)
        env.reset()

        # Add new camera after reset using utility function
        state = env.sim.get_state().flatten()
        camera_name = "agentviewleft"
        offset = np.array([0, -0.12, 0])  # Offset from agentview camera
        success = add_camera_with_offset(env, camera_name, "agentview", offset)
        if success:
            # Reset to the same state after adding camera
            env.sim.reset()
            env.sim.set_state_from_flattened(state)
            env.sim.forward()

        # if use camera from extrinsics, add camera
        if add_camera_from_extrinsics:
            env, _, _pixelkey2camera = add_cameras_from_extrinsics(
                env, camera_extrinsics_file, T_robot_base
            )
        else:
            _pixelkey2camera = None

        # apply wrappers
        env = RGBArrayAsObservationWrapper(
            env,
            task_name,
            height=height,
            width=width,
            use_robot=eval,
            max_episode_len=max_episode_len,
            max_state_dim=max_state_dim,
            pixel_keys=pixel_keys,
            _pixelkey2camera=_pixelkey2camera,
            num_robot_points=num_robot_points,
            num_points_per_obj=num_points_per_obj,
            robot_points_key=robot_points_key,
            object_points_key=object_points_key,
            obs_type=obs_type,
            action_mode=action_mode,
            use_vlm_points=use_vlm_points,
            vlm_mode=vlm_mode,
            sam_predictor=sam_predictor,
            gemini_client=gemini_client,
            molmo_client=molmo_client,
            lang_model=lang_model,
            mast3r_model=mast3r_model,
            dust3r_inference=dust3r_inference,
            depth_type=depth_type,
            depth_model=depth_model,
            add_camera_from_extrinsics=add_camera_from_extrinsics,
            camera_extrinsics_file=camera_extrinsics_file,
            temporal_agg_strategy=temporal_agg_strategy,
            visualize_3d=visualize_3d,
            viser_server=viser_server,
            use_full_scene_pcd=use_full_scene_pcd,
            full_scene_num_points=full_scene_num_points,
            max_depth_meters=max_depth_meters,
            min_z_robot_frame=min_z_robot_frame,
            full_scene_camera_name=full_scene_camera_name,
        )
        env = ActionDTypeWrapper(env, np.float32)
        env = ActionRepeatWrapper(env, action_repeat)
        env = FrameStackWrapper(env, pixel_keys, num_frames=1)
        env = ExtendedTimeStepWrapper(env)

        envs.append(env)
        task_descriptions.append(env.language_instruction)

        if not eval:
            break
        idx2name[idx] = task_name

    # write task descriptions to file
    if eval:
        with open("task_names_env.txt", "w") as f:
            for idx in idx2name:
                f.write(f"{idx}: {idx2name[idx]}\n")

    return envs, task_descriptions
