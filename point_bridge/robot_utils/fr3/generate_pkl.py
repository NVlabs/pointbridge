# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Script to generate pkl files for expert demonstrations from hdf5 files with point labels.
The points on the first frame for each demo is obtained by auto keypoint labeling.
All the points are grounded in the robot's base frame.

Points reduced using Farthest Point Sampling.

This script has been updated to use centralized VLM detection functions from
robot_utils.common.vlm_detection. It supports the following VLM modes:
- segment_depth: Uses segment-based detection with depth information
- point_tracking: Uses point tracking across frames

For real data, VLMs are always used for object detection and tracking.
"""

import sys

sys.path.append("../../")

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import cv2
import json
import pickle as pkl
from pathlib import Path
import numpy as np
from pandas import read_csv
from scipy.spatial.transform import Rotation as R

import torch

from point_bridge.robot_utils.common.utils import (
    pixel3d_to_2d,
    matrix_to_rotation_6d,
)
from point_bridge.robot_utils.fr3.utils import camera2pixelkey
from point_bridge.robot_utils.common.franka_gripper_points import extrapoints

# VLM imports
from point_bridge.detection_utils.utils import init_models_for_vlm_detection
from point_bridge.robot_utils.common.vlm_detection import (
    get_vlm_points_using_segments_depth,
    get_vlm_points_using_point_tracking,
)

# Height and width limits for VLM detection
crop_h, crop_w = (0.0, 1.0), (0.0, 1.0)

DATASET_PATH = Path("/path/to/teleop_data/processed_data")
task_names = [
    "task_1",
    "task_2",
]

SAVE_DATA_PATH = Path("/path/to/pointbridge/expert_demos")
CALIBRATION_PATH = Path(
    "/path/to/pointbridge/point_bridge/robot_utils/fr3_nyu/calibration_matrices"
)

benchmark_name = "fr3"
camera_indices = [8]
save_pixels = True
NUM_DEMOS = None  # Set to None to process all demos
save_dir_suffix = "teleop_data"
original_img_size = (640, 480)
start_crop = (None, None)
save_img_size = (640, 480)  # (128, 128)  # (width, height) - cropped size
use_noised_foreground_points = False
process_points = True
# zed camera
zed_indices = [8]  # Zed does not have depth
zed_image_size = (672, 448)  # (1280, 720)

# default device
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# get task names
if task_names is None:
    task_names = [f.name for f in DATASET_PATH.iterdir() if f.is_dir()]
num_tasks = len(task_names)

# VLM parameters
vlm_mode = "segment_depth"  # segment_depth, point_tracking
use_foundation_stereo = True  # only for segment_depth mode
if vlm_mode == "segment_depth":
    assert (
        use_foundation_stereo
    ), "foundation stereo must be used for generating robot data"

# append vlm mode to save_dir_suffix
if process_points:
    save_dir_suffix += f"_{vlm_mode}"

# point params
num_points_per_obj = 1000
SAVED_POINTS_PER_OBJ = [128]  # 0 means just save random points

# orientation of the robot at the 0th step
robot_base_orientation = R.from_rotvec([np.pi, 0, 0]).as_matrix()

# Cropping parameters: crop from start_crop to img_size if start_crop is not (None, None)
original_width, original_height = original_img_size
crop_width, crop_height = save_img_size
crop_x_offset, crop_y_offset = 0, 0
if start_crop[0] is not None:
    crop_x_offset = int(original_width * start_crop[0])
if start_crop[1] is not None:
    crop_y_offset = int(original_height * start_crop[1])


def center_crop_image(image):
    """Center crop image to save_img_size"""
    img_h, img_w = image.shape[:2]

    diff = (img_w - img_h) // 2
    diff += (img_w - img_h) // 5  # add 20% of the difference to the diff
    image = image[:, diff : diff + img_h]
    shift_h, shift_w = 0, diff

    return image, shift_h, shift_w


def adjust_camera_intrinsics_for_crop(intrinsics_matrix, crop_x_offset, crop_y_offset):
    """Adjust camera intrinsics matrix for center cropping"""
    # Create a new intrinsics matrix with adjusted principal point
    new_intrinsics = intrinsics_matrix.copy()
    new_intrinsics[0, 2] -= crop_x_offset  # Adjust cx
    new_intrinsics[1, 2] -= crop_y_offset  # Adjust cy
    return new_intrinsics


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


def extract_number(s):
    s = s.strip()
    # Remove any leading/trailing brackets or whitespace
    s = s.strip("[]")
    # Match 'np.float32(number)' or 'np.float64(number)'
    match = re.match(r"np\.float(?:32|64)\((-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)", s)
    if match:
        return float(match.group(1))
    else:
        # Match plain numbers, including negatives and decimals
        match = re.match(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)
        if match:
            return float(match.group(0))
        else:
            raise ValueError(f"Cannot extract number from '{s}'")


if not save_pixels:
    save_dir_suffix += "_no_images"

# create save directory
SAVE_DATA_PATH = (
    SAVE_DATA_PATH / f"{benchmark_name}_{save_dir_suffix}"
    if save_dir_suffix != ""
    else SAVE_DATA_PATH / benchmark_name
)
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)


# Initialize VLM models
print("Initializing VLM models...")
(
    sam_predictor,
    gemini_client,
    molmo_client,
    lang_model,
    mast3r_model,
    dust3r_inference,
) = init_models_for_vlm_detection()
print("VLM models initialized")
print(f"Using VLM mode: {vlm_mode}")

if vlm_mode == "segment_depth" and use_foundation_stereo:
    from model_servers.foundation_stereo.client import FoundationStereoZMQClient

    foundation_stereo_model = FoundationStereoZMQClient(
        server_address="localhost:60000"
    )

    zed_image_size = (672, 448)


# get camera extrinsics and intrinsics
if CALIBRATION_PATH is not None:
    camera_extrinsics_json = {}
    with open(CALIBRATION_PATH / "camera_extrinsics.json", "r") as f:
        camera_extrinsics_json = json.load(f)
    camera_intrinsics_json = {}
    with open(CALIBRATION_PATH / "camera_intrinsics.json", "r") as f:
        camera_intrinsics_json = json.load(f)

    # Convert camera extrinsics to 4x4 transformation matrices
    camera_extrinsics = {}  # camera in robot base frame
    for camera_name, extrinsics in camera_extrinsics_json.items():
        translation = np.array(
            extrinsics["translation"]
        ).flatten()  # Convert to 1D array
        rotation = np.array(extrinsics["rotation"])
        extrinsic_matrix = np.eye(4, dtype=np.float32)
        extrinsic_matrix[:3, :3] = rotation
        extrinsic_matrix[:3, 3] = translation
        camera_extrinsics[camera_name] = np.linalg.inv(
            extrinsic_matrix
        )  # robot base in camera frame

    # Convert camera intrinsics to 3x3 matrices
    camera_intrinsics = {}
    for camera_name, intrinsics in camera_intrinsics_json.items():
        intrinsics_matrix = np.array(intrinsics)
        camera_intrinsics[camera_name] = intrinsics_matrix

        if vlm_mode == "segment_depth" and use_foundation_stereo:
            camera_intrinsics[camera_name] = adjust_intrinsic_matrix_for_scale(
                camera_intrinsics[camera_name],
                (zed_image_size[0] / 1280, zed_image_size[1] / 720),
            )

else:
    raise RuntimeError("CALIBRATION_PATH must be provided for real data")

tasks_stored = 0
save_task_names = []
for task_name in task_names:
    print(f"Processing {tasks_stored+1}/{num_tasks}: {task_name}")

    # task instructions
    task_file = DATASET_PATH / task_name / "label.txt"
    with open(task_file, "r") as f:
        task_description = f.read()
    # Initialize sentence transformer for task embedding
    task_emb = lang_model.encode(task_description)

    dirs = [x for x in (DATASET_PATH / task_name).iterdir() if x.is_dir()]

    observations = []
    actions = []

    for demo_idx, data_point in enumerate(sorted(dirs)):
        if NUM_DEMOS is not None and demo_idx >= NUM_DEMOS:
            break

        print(f"Processing demo {demo_idx + 1}")

        try:
            observation = {}

            # Initialize observation data structures
            CAMERA_POSES, CAMERA_INTRINSICS, ROBOT_BASES = [], [], []
            if process_points:
                observation["hand_tracks_3d"], observation["robot_tracks_3d"] = [], []
                for saved_points_per_obj in SAVED_POINTS_PER_OBJ:
                    save_key_name = "object_tracks"
                    save_key_name += (
                        f"_{saved_points_per_obj}"
                        if saved_points_per_obj > 0
                        else "_random"
                    )
                    observation[f"{save_key_name}_3d"] = []
                # for camera_name in camera_names:
                for cam_idx in camera_indices:
                    camera_names = (
                        [f"cam_{cam_idx}"]
                        if cam_idx not in zed_indices
                        else [f"cam_{cam_idx}_left", f"cam_{cam_idx}_right"]
                    )
                    for camera_name in camera_names:
                        pixel_key = camera2pixelkey[camera_name]
                        observation[f"hand_tracks_{pixel_key}"] = []
                        observation[f"robot_tracks_{pixel_key}"] = []
                        for saved_points_per_obj in SAVED_POINTS_PER_OBJ:
                            save_key_name = "object_tracks"
                            save_key_name += (
                                f"_{saved_points_per_obj}"
                                if saved_points_per_obj > 0
                                else "_random"
                            )
                            observation[f"{save_key_name}_{pixel_key}"] = []

            # Get all frames for this demo to prepare for tracking
            image_dir = data_point / "videos"
            if not image_dir.exists():
                print(f"Data point {data_point} is incomplete")
                continue
            demo_frames = {}
            for save_idx, idx in enumerate(camera_indices):
                video_path = image_dir / f"camera{idx}.mp4"
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"Video {video_path} could not be opened")
                    continue

                if idx not in zed_indices:
                    camera_name = f"cam_{idx}"
                    pixel_key = camera2pixelkey[camera_name]

                    demo_frames[pixel_key], frames = [], []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # crop the image
                        h, w, _ = frame.shape
                        frame = frame[
                            int(h * crop_h[0]) : int(h * crop_h[1]),
                            int(w * crop_w[0]) : int(w * crop_w[1]),
                        ]
                        if (
                            frame.shape[0] != original_img_size[1]
                            or frame.shape[1] != original_img_size[0]
                        ):
                            frame = cv2.resize(frame, original_img_size)

                        demo_frames[pixel_key].append(frame)
                        if (
                            frame.shape[0] != save_img_size[1]
                            or frame.shape[1] != save_img_size[0]
                        ):
                            frame = cv2.resize(frame, save_img_size)
                        frames.append(frame)
                    demo_frames[pixel_key] = np.array(demo_frames[pixel_key])
                    if save_pixels:
                        observation[pixel_key] = np.array(frames)
                else:
                    camera_names = [f"cam_{idx}_left", f"cam_{idx}_right"]
                    pixel_keys = [
                        camera2pixelkey[camera_name] for camera_name in camera_names
                    ]
                    demo_frames[pixel_keys[0]], demo_frames[pixel_keys[1]] = [], []
                    frames_left, frames_right = [], []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        h, w, _ = frame.shape
                        frame_left = frame[:, : w // 2]
                        frame_right = frame[:, w // 2 :]

                        # crop the image
                        h, w, _ = frame_left.shape
                        frame_left = frame_left[
                            int(h * crop_h[0]) : int(h * crop_h[1]),
                            int(w * crop_w[0]) : int(w * crop_w[1]),
                        ]
                        frame_right = frame_right[
                            int(h * crop_h[0]) : int(h * crop_h[1]),
                            int(w * crop_w[0]) : int(w * crop_w[1]),
                        ]

                        if (
                            frame_left.shape[0] != zed_image_size[1]
                            or frame_left.shape[1] != zed_image_size[0]
                        ):
                            frame_left = cv2.resize(frame_left, zed_image_size)
                        if (
                            frame_right.shape[0] != zed_image_size[1]
                            or frame_right.shape[1] != zed_image_size[0]
                        ):
                            frame_right = cv2.resize(frame_right, zed_image_size)

                        demo_frames[pixel_keys[0]].append(frame_left)
                        demo_frames[pixel_keys[1]].append(frame_right)

                        if (
                            frame_left.shape[0] != save_img_size[1]
                            or frame_left.shape[1] != save_img_size[0]
                        ):
                            frame_left = cv2.resize(frame_left, save_img_size)
                        if (
                            frame_right.shape[0] != save_img_size[1]
                            or frame_right.shape[1] != save_img_size[0]
                        ):
                            frame_right = cv2.resize(frame_right, save_img_size)
                        frames_left.append(frame_left)
                        frames_right.append(frame_right)
                    demo_frames[pixel_keys[0]] = np.array(demo_frames[pixel_keys[0]])
                    demo_frames[pixel_keys[1]] = np.array(demo_frames[pixel_keys[1]])
                    if save_pixels:
                        observation[pixel_keys[0]] = np.array(frames_left)
                        observation[pixel_keys[1]] = np.array(frames_right)

            ########################

            # save eef states and gripper states
            state_csv_path = data_point / "states.csv"
            state = read_csv(state_csv_path)
            # Parsing cartesian pose data
            cartesian_states = state["pose_aa"].values
            cartesian_states = np.array(
                [
                    np.array([extract_number(x) for x in pose.strip("[]").split(",")])
                    for pose in cartesian_states
                ],
                dtype=np.float32,
            )
            cartesian_states = np.concatenate(
                [
                    cartesian_states[:, :3],
                    R.from_rotvec(cartesian_states[:, 3:]).as_quat(),
                ],
                axis=-1,
            )
            gripper_states = state["gripper_state"].values.astype(np.float32)
            observation["eef_states"] = cartesian_states.astype(np.float32)
            observation["gripper_states"] = gripper_states.astype(np.float32)

            # actions
            action = state["cmd_pose_aa"].values
            action = np.array(
                [
                    np.array([extract_number(x) for x in pose.strip("[]").split(",")])
                    for pose in action
                ],
                dtype=np.float32,
            )
            # compute delta actions
            T_cmd = np.array([np.eye(4) for _ in range(len(action))])
            T_cmd[:, :3, 3] = action[:, :3]
            T_cmd[:, :3, :3] = R.from_rotvec(action[:, 3:]).as_matrix()
            T_current = np.array([np.eye(4) for _ in range(len(cartesian_states))])
            T_current[:, :3, 3] = cartesian_states[:, :3]
            T_current[:, :3, :3] = R.from_quat(cartesian_states[:, 3:]).as_matrix()
            T_delta = np.array(
                [T_cmd[i] @ np.linalg.inv(T_current[i]) for i in range(len(T_cmd))]
            )
            action = np.concatenate(
                [
                    T_delta[:, :3, 3],
                    R.from_matrix(T_delta[:, :3, :3]).as_quat(),
                ],
                axis=-1,
            )
            future_gripper = np.concatenate(
                [observation["gripper_states"][1:], observation["gripper_states"][-1:]],
                axis=0,
            )
            action = np.concatenate([action, future_gripper[:, None]], axis=-1).astype(
                np.float32
            )

            if not process_points:
                observations.append(observation)
                actions.append(action)
                continue

            # Get camera projections for triangulation
            camera_projections = {}
            for camera_idx in camera_indices:
                camera_names = (
                    [f"cam_{camera_idx}"]
                    if camera_idx not in zed_indices
                    else [f"cam_{camera_idx}_left", f"cam_{camera_idx}_right"]
                )
                for camera_name in camera_names:
                    pixel_key = camera2pixelkey[camera_name]
                    camera_extrinsic = camera_extrinsics[
                        camera_name
                    ]  # robot base in camera frame
                    camera_intrinsic = camera_intrinsics[camera_name]
                    camera_intrinsic = np.column_stack((camera_intrinsic, np.zeros(3)))
                    camera_projections[camera_name] = (
                        camera_intrinsic @ camera_extrinsic
                    )

            # Process each SAVED_POINTS_PER_OBJ value separately
            for save_idx, saved_points_per_obj in enumerate(SAVED_POINTS_PER_OBJ):
                print(
                    f"Processing {save_key_name} with {saved_points_per_obj} points per object ({save_idx + 1}/{len(SAVED_POINTS_PER_OBJ)})"
                )

                save_key_name = "object_tracks"
                save_key_name += (
                    f"_{saved_points_per_obj}"
                    if saved_points_per_obj > 0
                    else "_random"
                )

                # Initialize VLM tracking variables
                vlm_objects = None
                if vlm_mode == "segment":
                    pass
                elif vlm_mode == "segment_depth":
                    vlm_tracker = None
                    points2d, depths = {}, {}
                elif vlm_mode == "point_tracking":
                    point_trackers = None

                # Process each step in the trajectory
                for step_idx in range(len(action)):
                    if step_idx % 10 == 0:
                        print(f"Processing step {step_idx} of demo {demo_idx + 1}")

                    # Get gripper state
                    if save_idx == 0:
                        eef_state = observation["eef_states"][step_idx]
                        gripper_state = observation["gripper_states"][step_idx]

                        eef_pos, eef_ori = eef_state[:3], eef_state[3:]
                        eef_ori = R.from_quat(eef_ori).as_matrix()
                        T_g2b = np.eye(4)
                        T_g2b[:3, :3] = eef_ori
                        T_g2b[:3, 3] = eef_pos

                        # Get robot points (gripper keypoints)
                        points3d_robot = []
                        for idx, Tp in enumerate(extrapoints):
                            if gripper_state > 0 and idx in [0, 1]:
                                Tp = Tp.copy()
                                Tp[1, 3] = 0.015 if idx == 0 else -0.015
                            pt = T_g2b @ Tp  # pt in robot base frame
                            pt = pt[:3, 3]
                            points3d_robot.append(pt[:3])
                        points3d_robot = np.array(points3d_robot)
                        observation["robot_tracks_3d"].append(points3d_robot)

                        current_pose = np.concatenate(
                            [
                                eef_pos,
                                matrix_to_rotation_6d(eef_ori),
                                [gripper_state],
                            ]
                        )

                    # Get current images from all cameras and apply cropping
                    pixel_key_images = {}
                    for camera_idx in camera_indices:
                        camera_names = (
                            [f"cam_{camera_idx}"]
                            if camera_idx not in zed_indices
                            else [f"cam_{camera_idx}_left", f"cam_{camera_idx}_right"]
                        )
                        for camera_name in camera_names:
                            pixel_key = camera2pixelkey[camera_name]
                            image = demo_frames[pixel_key][step_idx]
                            pixel_key_images[pixel_key] = image

                    # Use centralized VLM detection functions for tracking
                    if vlm_mode == "segment_depth":
                        # camera_name = camera_names[0]
                        camera_name = f"cam_{camera_indices[0]}"
                        intrinsic_matrix = camera_intrinsics[camera_name + "_left"]
                        extrinsic_matrix = camera_extrinsics[camera_name + "_left"]

                        ref_pixel_key = camera2pixelkey[camera_name + "_left"]
                        ref_image = pixel_key_images[ref_pixel_key]

                        # compute depth using foundation stereo
                        img0 = ref_image
                        img1 = pixel_key_images[camera2pixelkey[camera_name + "_right"]]
                        ref_depth = foundation_stereo_model.get_depth_map(
                            img0, img1, intrinsic_matrix
                        )

                        (
                            _,
                            points3d_objects,
                            vlm_tracker,
                            vlm_objects,
                            points2d,
                            depths,
                        ) = get_vlm_points_using_segments_depth(
                            ref_image=ref_image,
                            ref_depth=ref_depth,
                            task_description=task_description,
                            vlm_tracker=vlm_tracker,
                            vlm_objects=vlm_objects,
                            is_first_step=(step_idx == 0),
                            num_points_per_obj=saved_points_per_obj,
                            current_pose=current_pose,
                            intrinsic_matrix=intrinsic_matrix,
                            extrinsic_matrix=extrinsic_matrix,
                            sam_predictor=sam_predictor,
                            gemini_client=gemini_client,
                            molmo_client=molmo_client,
                            task_name=task_name,
                            height_limits=crop_h,
                            width_limits=crop_w,
                            points2d=points2d,
                            depths=depths,
                            use_noised_foreground_points=use_noised_foreground_points,
                        )
                    elif vlm_mode == "point_tracking":
                        (
                            _,
                            points3d_objects,
                            vlm_objects,
                            point_trackers,
                        ) = get_vlm_points_using_point_tracking(
                            pixel_key_images=pixel_key_images,
                            task_description=task_description,
                            pixel_keys=[
                                camera2pixelkey[camera_name]
                                for camera_name in camera_names
                            ],
                            camera_projections=camera_projections,
                            vlm_objects=vlm_objects,
                            is_first_step=(step_idx == 0),
                            num_points_per_obj=saved_points_per_obj,
                            current_pose=current_pose,
                            sam_predictor=sam_predictor,
                            gemini_client=gemini_client,
                            molmo_client=molmo_client,
                            mast3r_model=mast3r_model,
                            dust3r_inference=dust3r_inference,
                            task_name=task_name,
                            height_limits=crop_h,
                            width_limits=crop_w,
                            point_trackers=point_trackers,
                            pixelkey2camera={v: k for k, v in camera2pixelkey.items()},
                            use_noised_foreground_points=use_noised_foreground_points,
                        )

                    observation[f"{save_key_name}_3d"].append(points3d_objects)

                    # Project 3D points to 2D for each camera
                    for camera_idx in camera_indices:
                        camera_names = (
                            [f"cam_{camera_idx}"]
                            if camera_idx not in zed_indices
                            else [f"cam_{camera_idx}_left", f"cam_{camera_idx}_right"]
                        )
                        for camera_name in camera_names:
                            pixel_key = camera2pixelkey[camera_name]

                            camera_extrinsic = camera_extrinsics[
                                camera_name
                            ]  # robot base in camera frame
                            camera_intrinsic = camera_intrinsics[camera_name]
                            camera_intrinsic = np.column_stack(
                                (camera_intrinsic, np.zeros(3))
                            )

                            if save_idx == 0:
                                points2d_robot, _ = pixel3d_to_2d(
                                    points3d_robot, camera_intrinsic, camera_extrinsic
                                )
                                observation[f"robot_tracks_{pixel_key}"].append(
                                    points2d_robot
                                )

                            # Project object points
                            points2d_objects = []
                            for obj_pts in points3d_objects:
                                points2d_obj, _ = pixel3d_to_2d(
                                    obj_pts, camera_intrinsic, camera_extrinsic
                                )
                                points2d_objects.append(points2d_obj)
                            observation[f"{save_key_name}_{pixel_key}"].append(
                                np.array(points2d_objects)
                            )

        except:
            print(f"Error demo {demo_idx + 1}, continuing...")
            continue

        # Convert lists to numpy arrays
        observation["robot_tracks_3d"] = np.array(observation["robot_tracks_3d"])

        for saved_points_per_obj in SAVED_POINTS_PER_OBJ:
            save_key_name = "object_tracks"
            save_key_name += (
                f"_{saved_points_per_obj}" if saved_points_per_obj > 0 else "_random"
            )
            observation[f"{save_key_name}_3d"] = np.array(
                observation[f"{save_key_name}_3d"]
            )

        for camera_idx in camera_indices:
            camera_names = (
                [f"cam_{camera_idx}"]
                if camera_idx not in zed_indices
                else [f"cam_{camera_idx}_left", f"cam_{camera_idx}_right"]
            )
            for camera_name in camera_names:
                if save_pixels:
                    observation[pixel_key] = np.array(observation[pixel_key])

                pixel_key = camera2pixelkey[camera_name]
                for saved_points_per_obj in SAVED_POINTS_PER_OBJ:
                    save_key_name = "object_tracks"
                    save_key_name += (
                        f"_{saved_points_per_obj}"
                        if saved_points_per_obj > 0
                        else "_random"
                    )

                    observation[f"{save_key_name}_{pixel_key}"] = np.array(
                        observation[f"{save_key_name}_{pixel_key}"]
                    )
                    # resize to save_img_size
                    if camera_idx in zed_indices:
                        observation[f"{save_key_name}_{pixel_key}"][..., 0] *= (
                            save_img_size[0] / zed_image_size[0]
                        )
                        observation[f"{save_key_name}_{pixel_key}"][..., 1] *= (
                            save_img_size[1] / zed_image_size[1]
                        )
                    else:
                        observation[f"{save_key_name}_{pixel_key}"][..., 0] *= (
                            save_img_size[0] / original_img_size[0]
                        )
                        observation[f"{save_key_name}_{pixel_key}"][..., 1] *= (
                            save_img_size[1] / original_img_size[1]
                        )

                observation[f"robot_tracks_{pixel_key}"] = np.array(
                    observation[f"robot_tracks_{pixel_key}"]
                )
                # resize robot tracks to save_img_size
                if camera_idx in zed_indices:
                    observation[f"robot_tracks_{pixel_key}"][..., 0] *= (
                        save_img_size[0] / zed_image_size[0]
                    )
                    observation[f"robot_tracks_{pixel_key}"][..., 1] *= (
                        save_img_size[1] / zed_image_size[1]
                    )
                else:
                    observation[f"robot_tracks_{pixel_key}"][..., 0] *= (
                        save_img_size[0] / original_img_size[0]
                    )
                    observation[f"robot_tracks_{pixel_key}"][..., 1] *= (
                        save_img_size[1] / original_img_size[1]
                    )

        # save data
        observations.append(observation)
        actions.append(action)

    if len(observations) == 0:
        print(f"No observations for task {task_name}. Not saving")
        continue

    # save data
    save_data_path = SAVE_DATA_PATH / f"{task_name}.pkl"
    save_data = {
        "observations": observations,
        "actions": actions,
        "task_desc": task_description,
        "task_emb": task_emb,
        "camera_intrinsics": camera_intrinsics,
        "camera_extrinsics": camera_extrinsics,
    }
    with open(save_data_path, "wb") as f:
        pkl.dump(save_data, f)

    print(f"Saved to {str(save_data_path)}")

    tasks_stored += 1
    save_task_names.append(task_name)

print(f"Saved {len(save_task_names)} tasks.")
