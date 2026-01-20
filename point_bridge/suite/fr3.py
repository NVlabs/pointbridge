# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any, NamedTuple

import gym
from gym import spaces
import franka_env

import dm_env
from dm_env import StepType, specs, TimeStep

import cv2
import json
import einops
import numpy as np
import pickle as pkl
from pathlib import Path
from collections import deque
from scipy.spatial.transform import Rotation as R

from point_bridge.robot_utils.common.utils import (
    rigid_transform_3D,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    transform_points,
    farthest_point_sampling,
    pixel2d_to_3d,
)
from robot_utils.common.franka_gripper_points import extrapoints
from point_bridge.robot_utils.fr3.utils import pixelkey2camera
from sentence_transformers import SentenceTransformer

# for VLM points
from point_bridge.detection_utils.utils import init_models_for_vlm_detection

from point_bridge.robot_utils.common.vlm_detection import (
    get_vlm_points_using_segments_depth,
    get_vlm_points_using_point_tracking,
)

# Height and width limits for VLM detection
crop_h, crop_w = (0.0, 1.0), (0.0, 1.0)


def adjust_intrinsic_matrix_for_scale(intrinsic_matrix, scale_factor):
    """
    Adjust intrinsic matrix for image scaling.

    Args:
        intrinsic_matrix: 3x3 camera intrinsic matrix
        scale_factor: scaling factor (fx, fy) for x and y dimensions

    Returns:
        Adjusted 3x3 intrinsic matrix
    """
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


def center_crop_image(image):
    """Center crop image to save_img_size"""
    img_h, img_w = image.shape[:2]

    diff = (img_w - img_h) // 2
    diff += (img_w - img_h) // 5  # add 20% of the difference to the diff
    image = image[:, diff : diff + img_h]
    shift_h, shift_w = 0, diff

    return image, shift_h, shift_w


def init_models():
    # load sentence transformer
    print("Initializing Sentence Transformer ...")
    lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    return lang_model


def generate_full_scene_pointcloud(
    depth_img,
    intrinsic_matrix,
    extrinsic_matrix,
    max_depth=2.0,
    min_z_robot=0.01,
    num_points=512,
    y_limits=(-0.25, 0.25),
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
        y_limits: Tuple of (min_y, max_y) limits in robot base frame

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

    # Filter by y limits in robot base frame
    if y_limits is not None:
        min_y, max_y = y_limits
        valid_y_mask = (points_3d_robot[:, 1] >= min_y) & (
            points_3d_robot[:, 1] <= max_y
        )
        points_3d_robot = points_3d_robot[valid_y_mask]

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
        lang_model,
        width=256,
        height=256,
        use_robot=False,
        max_episode_len=300,
        max_state_dim=100,
        pixel_keys=["pixels"],
        num_robot_points=8,
        num_points_per_obj=8,
        robot_points_key="robot_tracks",
        object_points_key="object_tracks",
        obs_type=["image"],
        action_mode="pose",
        use_vlm_points=False,  # New parameter to choose between GT and VLM points
        vlm_mode="segment_depth",  # segment_depth, point_tracking
        sam_predictor=None,
        gemini_client=None,
        molmo_client=None,
        mast3r_model=None,
        dust3r_inference=None,
        depth_type="gt",
        depth_model=None,
        temporal_agg_strategy=None,
        visualize_3d=False,
        server_deploy=False,
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
        self._device = "cpu"
        self._obs_type = obs_type
        self._action_mode = action_mode
        self.use_noised_foreground_points = False
        self.visualize_3d = visualize_3d
        self.server_deploy = server_deploy
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
        self.depth_type = depth_type
        self.depth_model = depth_model

        # init models
        self.lang_model = lang_model

        # Initialize task description - for FR3, this will be set during reset
        # self.task_description = None
        # self.task_description = "put the bowl on the plate"
        # self.task_description = "put the mug on the plate"
        # self.task_description = "fold the towel"
        # self.task_description = "close the oven door"
        # self.task_description = "put the lemon in the bowl"
        # self.task_description = "put the bowl in the basket"
        self.task_description = "stack the left red bowl on the right bowl"
        # self.task_description = "close the drawer"
        # self.task_description = "stack the left bowl on the right bowl"
        # self.task_description = "put the bowl in the oven"

        self.task_emb = self.lang_model.encode(self.task_description)

        # camera names
        self._camera_names = [pixelkey2camera[pixel_key] for pixel_key in pixel_keys]

        # For foundation_stereo, we need to set the height and width to 448x672
        # since the TRT optimized model is only available for 448x672.
        # Otherwise resizing depth leads to artifacts in resized depth.
        if self.depth_type == "foundation_stereo":
            self._height, self._width = 448, 672

        # calibration data
        if "points" in self._obs_type:
            # For FR3, we'll load camera intrinsics and extrinsics from saved data
            # This will be initialized in reset() when we have access to the environment
            self.camera_intrinsics = {}
            self.camera_extrinsics = {}
            self.camera_projections = {}

        # obs = self._env.reset()
        pixel_shape = (self._height, self._width, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=pixel_shape, dtype=np.uint8
        )

        # Action spec
        self.action_shape = (10,)  # (3d pos, 6d ori, gripper)
        self.action_space = specs.BoundedArray(
            self.action_shape, np.float32, -np.inf, np.inf, "action"
        )

        # Action spec for dm_env compatibility
        self._action_spec = specs.BoundedArray(
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
            self._object_names = {}

        # visualize 3d with viser
        if self.visualize_3d:
            import viser

            self.server = viser.ViserServer(port=2360)

    def reset(self, **kwargs):
        self._step = 0

        obs = self._env.reset()

        self.robot_actions = None
        self.prev_gripper_state = -1  # Default open gripper

        # Load camera calibration data if available
        if "points" in self._obs_type:  # and not hasattr(self, "camera_projections"):
            file_path = Path(os.path.abspath(__file__))
            file_path = (
                file_path.parent.parent / "robot_utils/fr3_nyu/calibration_matrices"
            )
            if file_path.exists():
                self.load_camera_calibration(file_path)
            else:
                # Load camera intrinsics and extrinsics from saved data
                # This would typically come from the expert demonstrations
                if "camera_intrinsics" in kwargs:
                    self.camera_intrinsics = kwargs["camera_intrinsics"]
                if "camera_extrinsics" in kwargs:
                    self.camera_extrinsics = kwargs["camera_extrinsics"]

            if self.depth_type == "foundation_stereo":
                for camera_name in self._camera_names:
                    # For ZED camera, the intrinsics are for 1280x720.
                    # We need to scale them to the current image size.
                    self.camera_intrinsics[
                        camera_name
                    ] = adjust_intrinsic_matrix_for_scale(
                        self.camera_intrinsics[camera_name].copy(),
                        (self._width / 1280, self._height / 720),
                    )

            # Initialize camera projections
            self.camera_projections = {}
            for camera_name in self._camera_names:
                intr = self.camera_intrinsics[camera_name]
                extr = self.camera_extrinsics[camera_name]
                intr = np.column_stack((intr, np.zeros(3)))
                self.camera_projections[camera_name] = intr @ extr

            if "foundation_stereo" in self.depth_type:
                self.fs_client = self.depth_model
                H, W = 448, 672
                self.scale_factor_fs = (W / self._width, H / self._height)
                self.adjusted_camera_intrinsics = {}
                for pixel_key in self._pixel_keys:
                    camera_name = pixelkey2camera[pixel_key]
                    if self.scale_factor_fs != (1.0, 1.0):
                        self.adjusted_camera_intrinsics[
                            camera_name
                        ] = adjust_intrinsic_matrix_for_scale(
                            self.camera_intrinsics[camera_name].copy(),
                            self.scale_factor_fs,
                        )
                    else:
                        self.adjusted_camera_intrinsics[
                            camera_name
                        ] = self.camera_intrinsics[camera_name].copy()

        observation = {}
        observation["task_emb"] = self.task_emb
        observation["goal_achieved"] = False

        for key in self._pixel_keys:
            frame = obs[key]
            if frame.shape[:2] != (self._height, self._width):
                if self._height == self._width:
                    frame, _, _ = center_crop_image(frame)

                frame = cv2.resize(
                    frame, (self._width, self._height), interpolation=cv2.INTER_LINEAR
                )
            observation[key] = frame

        # compute current pose in robot base frame
        pos, ori = obs["features"][:3], obs["features"][3:7]
        ori = matrix_to_rotation_6d(R.from_quat(ori).as_matrix())
        gripper = [-1]  # obs["gripper_position"]
        self._current_pose = np.concatenate(
            [pos, ori, gripper]
        )  # gripper in robot base frame

        # Update robot base orientation for point-based actions
        self.robot_base_orientation = R.from_quat(obs["features"][3:7]).as_matrix()

        observation["proprioceptive"] = self._current_pose
        observation["features"] = self._current_pose

        if "points" in self._obs_type:
            # get robot and object points in robot base frame
            if self.use_full_scene_pcd:
                points3d_robot, points3d_objects = self.get_full_scene_pcd(obs)
            elif self.use_vlm_points:
                if self.vlm_mode == "segment_depth":
                    vlm_func = self.get_vlm_points_using_segments_depth
                elif self.vlm_mode == "point_tracking":
                    vlm_func = self.get_vlm_points_using_point_tracking
                points3d_robot, points3d_objects = vlm_func(obs)
            else:
                points3d_robot, points3d_objects = self.get_gt_points()

            # base robot points
            self.base_robot_points = points3d_robot

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

                    num_actions_execute = 5  # 10
                    self.robot_actions = robot_actions[:num_actions_execute]
                    robot_action = self.robot_actions

                elif action_shape_dim == 1:
                    robot_action = self.point2action(action)
                    robot_action = np.array([robot_action])
            elif self._action_mode == "pose":
                if len(action.shape) == 1:
                    robot_action = np.array([action])
                else:
                    robot_action = action

            env_actions = []
            for action_i in robot_action:
                # compute gripper state
                self.prev_gripper.append(action_i[-1])
                self.prev_gripper_state = action_i[-1]

                if self._action_mode == "pose":
                    pos, ori, gripper = action_i[:3], action_i[3:9], action_i[-1:]

                    ori = rotation_6d_to_matrix(ori)  # in robot base frame

                    ori = R.from_matrix(ori).as_rotvec()
                    action_i = np.concatenate([pos, ori, gripper])

                env_actions.append(action_i)

            for action_i in env_actions:
                self._step += 1
                if not self.server_deploy:
                    obs, reward, done, _, info = self._env.step(action_i)
                else:
                    obs = self._env.step(action_i)
                    reward, done, info = 0.0, False, {}
                if self._step >= self._max_episode_len:
                    break
            if self._step >= self._max_episode_len:
                break

        observation = {}
        observation["task_emb"] = self.task_emb
        observation["goal_achieved"] = done

        for key in self._pixel_keys:
            frame = obs[key]

            if self._height == self._width:
                frame, _, _ = center_crop_image(frame)

            if frame.shape[:2] != (self._height, self._width):
                frame = cv2.resize(
                    frame, (self._width, self._height), interpolation=cv2.INTER_LINEAR
                )
            observation[key] = frame

        # compute current pose in robot base frame
        pos, ori = obs["features"][:3], obs["features"][3:7]
        ori = matrix_to_rotation_6d(R.from_quat(ori).as_matrix())
        gripper = [obs["features"][-1]]
        self._current_pose = np.concatenate(
            [pos, ori, gripper]
        )  # gripper in robot base frame
        observation["proprioceptive"] = self._current_pose
        observation["features"] = self._current_pose

        if "points" in self._obs_type:
            # get robot and object points in robot base frame
            if self.use_full_scene_pcd:
                points3d_robot, points3d_objects = self.get_full_scene_pcd(obs)
            elif self.use_vlm_points:
                if self.vlm_mode == "segment_depth":
                    vlm_func = self.get_vlm_points_using_segments_depth
                elif self.vlm_mode == "point_tracking":
                    vlm_func = self.get_vlm_points_using_point_tracking
                points3d_robot, points3d_objects = vlm_func(obs)
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
        return np.zeros((height, width, 3)).astype(np.uint8)

    def get_gt_points(self):
        # get 3d robot points
        eef_pos, eef_ori = self._current_pose[:3], self._current_pose[3:9]
        eef_ori = rotation_6d_to_matrix(eef_ori)
        T_g2b = np.eye(4)  # gripper in robot base
        T_g2b[:3, :3] = eef_ori
        T_g2b[:3, 3] = eef_pos

        # add point transform
        points3d_robot = []
        gripper_state = self._current_pose[-1]
        for idx, Tp in enumerate(extrapoints):
            if gripper_state > 0 and idx in [0, 1]:
                Tp = Tp.copy()
                Tp[1, 3] = 0.015 if idx == 0 else -0.015
            pt = T_g2b @ Tp  # pt in robot base
            pt = pt[:3, 3]
            points3d_robot.append(pt[:3])
        points3d_robot = np.array(points3d_robot)  # (N, D)

        # For FR3, we'll use saved object points or VLM detection
        # This is a placeholder - in practice, you'd load from saved data
        points3d_objects = np.zeros((0, self._num_points_per_obj, 3))

        return points3d_robot, points3d_objects

    def get_vlm_points_using_segments_depth(self, obs):
        """
        Get VLM-based object masks and 3D points using depth information.
        Generates both image and ground truth depth for the first camera only,
        runs detect_segments to get object masks, randomly samples 1000 points
        from each mask, projects to 3D using depth and camera parameters,
        and applies farthest point sampling to reduce to desired number of points.
        """

        # Get current image and depth from the first camera only
        ref_pixel_key = self._pixel_keys[0]
        camera_name = pixelkey2camera[ref_pixel_key]

        # For FR3, we get images from observation
        ref_image = obs[ref_pixel_key]
        if ref_image.shape[:2] != (self._height, self._width):
            ref_image = cv2.resize(
                ref_image, (self._width, self._height), interpolation=cv2.INTER_LINEAR
            )

        # Get camera parameters for the reference camera
        camera_name = pixelkey2camera[ref_pixel_key]
        intrinsic_matrix = self.camera_intrinsics[camera_name]
        extrinsic_matrix = self.camera_extrinsics[
            camera_name
        ]  # robot base in camera frame

        # Get depth using foundation stereo or from environment
        if self.depth_type == "gt":
            # For FR3, we might not have depth - this would need to be adapted
            ref_depth = obs[f"depth{camera_name.split('_')[-1]}"]
            ref_depth = ref_depth / 1000.0  # in m
        elif "foundation_stereo" in self.depth_type:
            if self.scale_factor_fs != (1.0, 1.0):
                img0 = cv2.resize(
                    ref_image,
                    (0, 0),
                    fx=self.scale_factor_fs[1],
                    fy=self.scale_factor_fs[0],
                )
            else:
                img0 = ref_image.copy()

            img1 = obs[self._pixel_keys[1]]
            if img1.shape[:2] != (self._height, self._width):
                img1 = cv2.resize(
                    img1, (self._width, self._height), interpolation=cv2.INTER_LINEAR
                )

            if self.scale_factor_fs != (1.0, 1.0):
                img1 = cv2.resize(
                    img1,
                    (0, 0),
                    fx=self.scale_factor_fs[1],
                    fy=self.scale_factor_fs[0],
                )

            adjusted_intrinsic_matrix = self.adjusted_camera_intrinsics[camera_name]

            print("Querying depth map with FS ...")
            ref_depth = self.fs_client.get_depth_map(
                img0, img1, adjusted_intrinsic_matrix
            )

            # resize ref_depth to original image size
            if ref_depth.shape != (self._height, self._width):
                ref_depth = cv2.resize(
                    ref_depth,
                    (self._width, self._height),
                    interpolation=cv2.INTER_LINEAR,
                )

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
            use_noised_foreground_points=self.use_noised_foreground_points,
        )

        return points3d_robot, points3d_objects

    def get_vlm_points_using_point_tracking(self, obs):
        """
        Get VLM-based object points using point tracking.
        Uses detect_segments to obtain object masks during reset (self._step == 0),
        then uses PointTracker to track points across frames.
        Samples points from masks, computes multiview correspondence, and triangulates to get 3D points.
        """

        # Get current images from all cameras
        pixel_key_images = {}
        for pixel_key in self._pixel_keys:
            image = obs[pixel_key]
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
            pixelkey2camera=pixelkey2camera,
            image_percentage_for_tracking=0.4,
            use_noised_foreground_points=self.use_noised_foreground_points,
        )

        return points3d_robot, points3d_objects

    def get_full_scene_pcd(self, obs):
        """
        Generate full scene point cloud from depth image using specified camera.
        Returns robot points and scene point cloud as a single "object" for compatibility.
        """
        # Get robot points
        eef_pos, eef_ori = self._current_pose[:3], self._current_pose[3:9]
        eef_ori = rotation_6d_to_matrix(eef_ori)
        T_g2b = np.eye(4)  # gripper in robot base
        T_g2b[:3, :3] = eef_ori
        T_g2b[:3, 3] = eef_pos

        # add point transform
        points3d_robot = []
        gripper_state = self._current_pose[-1]
        for idx, Tp in enumerate(extrapoints):
            if gripper_state > 0 and idx in [0, 1]:
                Tp = Tp.copy()
                Tp[1, 3] = 0.015 if idx == 0 else -0.015
            pt = T_g2b @ Tp  # pt in robot base
            pt = pt[:3, 3]
            points3d_robot.append(pt[:3])
        points3d_robot = np.array(points3d_robot)  # (N, D)

        # Determine which camera to use for depth
        camera_name = self.full_scene_camera_name
        if camera_name is None:
            # Use the first camera by default
            camera_name = self._camera_names[0]

        # Get depth image
        if self.depth_type == "gt":
            depth_key = (
                f"depth{camera_name.split('_')[-1]}"
                if "_" in camera_name
                else f"depth{camera_name}"
            )
            if depth_key in obs:
                depth_img = obs[depth_key] / 1000.0  # Convert from mm to meters
            else:
                # Try alternative key format
                depth_key = f"depth{camera_name}"
                if depth_key in obs:
                    depth_img = obs[depth_key] / 1000.0
                else:
                    # Fallback: use first available depth key
                    depth_keys = [k for k in obs.keys() if k.startswith("depth")]
                    if depth_keys:
                        depth_img = obs[depth_keys[0]] / 1000.0
                    else:
                        # No depth available, return zeros
                        points3d_scene = np.zeros((self.full_scene_num_points, 3))
                        points3d_objects = np.array([points3d_scene])
                        return points3d_robot, points3d_objects

        elif "foundation_stereo" in self.depth_type:
            # Get images for stereo
            ref_pixel_key = self._pixel_keys[0]
            ref_image = obs[ref_pixel_key]
            if ref_image.shape[:2] != (self._height, self._width):
                ref_image = cv2.resize(
                    ref_image,
                    (self._width, self._height),
                    interpolation=cv2.INTER_LINEAR,
                )

            if self.scale_factor_fs != (1.0, 1.0):
                img0 = cv2.resize(
                    ref_image,
                    (0, 0),
                    fx=self.scale_factor_fs[1],
                    fy=self.scale_factor_fs[0],
                )
            else:
                img0 = ref_image.copy()

            if len(self._pixel_keys) > 1:
                img1 = obs[self._pixel_keys[1]]
                if img1.shape[:2] != (self._height, self._width):
                    img1 = cv2.resize(
                        img1,
                        (self._width, self._height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                if self.scale_factor_fs != (1.0, 1.0):
                    img1 = cv2.resize(
                        img1,
                        (0, 0),
                        fx=self.scale_factor_fs[1],
                        fy=self.scale_factor_fs[0],
                    )
            else:
                # Only one camera, can't do stereo
                points3d_scene = np.zeros((self.full_scene_num_points, 3))
                points3d_objects = np.array([points3d_scene])
                return points3d_robot, points3d_objects

            adjusted_intrinsic_matrix = self.adjusted_camera_intrinsics[camera_name]
            depth_img = self.fs_client.get_depth_map(
                img0, img1, adjusted_intrinsic_matrix
            )
            if depth_img.shape != (self._height, self._width):
                depth_img = cv2.resize(
                    depth_img,
                    (self._width, self._height),
                    interpolation=cv2.INTER_LINEAR,
                )
        else:
            # Fallback: try to get depth from obs
            depth_key = (
                f"depth{camera_name.split('_')[-1]}"
                if "_" in camera_name
                else f"depth{camera_name}"
            )
            if depth_key in obs:
                depth_img = obs[depth_key] / 1000.0
            else:
                # No depth available, return zeros
                points3d_scene = np.zeros((self.full_scene_num_points, 3))
                points3d_objects = np.array([points3d_scene])
                return points3d_robot, points3d_objects

        # Get camera parameters
        intrinsic_matrix = self.camera_intrinsics[camera_name]
        extrinsic_matrix = self.camera_extrinsics[
            camera_name
        ]  # robot base in camera frame

        # Generate full scene point cloud
        points3d_scene = generate_full_scene_pointcloud(
            depth_img,
            intrinsic_matrix,
            extrinsic_matrix,
            max_depth=self.max_depth_meters,
            min_z_robot=self.min_z_robot_frame,
            num_points=self.full_scene_num_points,
            y_limits=(-0.4, 0.4),
        )

        # Store as a single "object" for compatibility with existing structure
        points3d_objects = np.array([points3d_scene])

        return points3d_robot, points3d_objects

    def point2action(self, action):
        """
        Action is a dict with 10 points corresponding to each camera frame.
        """

        robot_pts_end_idx = self._num_robot_points
        points3d = action["future_tracks"][:robot_pts_end_idx, :3]

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
        robot_pos = (points3d[0, :3] + points3d[1, :3]) / 2  # in robot base frame
        ori, _ = rigid_transform_3D(self.base_robot_points, points3d)
        ori = ori @ self.robot_base_orientation  # in robot base frame

        return robot_pos, ori

    def compute_gripper(self, action):
        gripper_state = np.mean(action["gripper"])
        gripper_state = np.clip(gripper_state, -1, 1)
        gripper_state = np.array([gripper_state])
        return gripper_state

    def compute_robot_action(self, target_position, target_orientation, gripper):
        """
        Return absolute actions for FR3 (already in robot base frame)
        """
        # For FR3, we're already in the robot base frame, so we can return the target directly

        # Convert rotation matrix to rotation vector
        target_orientation_vec = R.from_matrix(target_orientation).as_rotvec()

        return np.concatenate([target_position, target_orientation_vec, gripper])

    def load_camera_calibration(self, dir_path):
        # camera intrinsics
        camera_extrinsics_json = {}
        with open(dir_path / "camera_extrinsics.json", "r") as f:
            camera_extrinsics_json = json.load(f)
        camera_intrinsics_json = {}
        with open(dir_path / "camera_intrinsics.json", "r") as f:
            camera_intrinsics_json = json.load(f)

        # camera extrinsics
        self.camera_extrinsics = {}  # robot base in camera frame
        for camera_name, extrinsics in camera_extrinsics_json.items():
            translation = np.array(
                extrinsics["translation"]
            ).flatten()  # Convert to 1D array
            rotation = np.array(extrinsics["rotation"])  # camera in robot base frame
            extrinsic_matrix = np.eye(4, dtype=np.float32)
            extrinsic_matrix[:3, :3] = rotation
            extrinsic_matrix[:3, 3] = translation
            self.camera_extrinsics[camera_name] = np.linalg.inv(
                extrinsic_matrix
            )  # robot base in camera frame

        # camera intrinsics
        self.camera_intrinsics = {}
        for camera_name, intrinsics in camera_intrinsics_json.items():
            intrinsics_matrix = np.array(intrinsics)
            self.camera_intrinsics[camera_name] = intrinsics_matrix

    def __getattr__(self, name):
        return getattr(self._env, name)

    # def __del__(self):
    #     self._env.env.close()


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

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


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, pixel_keys, num_frames):
        self._env = env
        self._num_frames = num_frames

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


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._discount = 1.0

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

    # def __del__(self):
    #     self._env.env.close()


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

    # def __del__(self):
    #     self._env.env.close()


def make(
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
    temporal_agg_strategy,
    visualize_3d,
    server_deploy,
    use_full_scene_pcd=False,
    full_scene_num_points=4196,
    max_depth_meters=1,
    min_z_robot_frame=0.0,
    full_scene_camera_name=None,
):
    # init models
    if eval:
        if not server_deploy:
            env = gym.make(
                "Franka-v1",
                cam_ids=[8],
                zed_ids=[8],
                height=height,
                width=width,
                use_robot=eval,
                use_gt_depth=(use_vlm_points == True and depth_type == "gt"),
                side="left",
            )  # absolute environment
            env.reset()
        else:
            from model_servers.robot_fr3_nyu.client import RobotZMQClient

            env = RobotZMQClient(server_address="100.94.192.118:7999")
            env.reset()
    else:
        env = None

    # Initialize VLM models if needed
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

    # Initialize foundation stereo if needed
    if "foundation_stereo" in depth_type:
        from model_servers.foundation_stereo.client import FoundationStereoZMQClient

        depth_model = FoundationStereoZMQClient(server_address="localhost:60000")
    else:
        depth_model = None

    envs, task_descriptions, idx2name = [], [], {}
    for idx, task_name in enumerate(task_names):
        # apply wrappers
        env = RGBArrayAsObservationWrapper(
            env,
            task_name,
            # task_info,
            lang_model,
            height=height,
            width=width,
            use_robot=eval,
            max_episode_len=max_episode_len,
            max_state_dim=max_state_dim,
            pixel_keys=pixel_keys,
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
            mast3r_model=mast3r_model,
            dust3r_inference=dust3r_inference,
            depth_type=depth_type,
            depth_model=depth_model,
            temporal_agg_strategy=temporal_agg_strategy,
            visualize_3d=visualize_3d,
            server_deploy=server_deploy,
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
        task_descriptions.append(task_name)

        if not eval:
            break
        idx2name[idx] = task_name

    # write task descriptions to file
    if eval:
        with open("task_names_env.txt", "w") as f:
            for idx in idx2name:
                f.write(f"{idx}: {idx2name[idx]}\n")

    return envs, task_descriptions
