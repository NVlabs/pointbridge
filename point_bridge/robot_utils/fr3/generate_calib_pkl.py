# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import sys

sys.path.append("../../")

import re
import argparse
import pickle as pkl
from pathlib import Path
import cv2
import numpy as np
from pandas import read_csv
from scipy.spatial.transform import Rotation as R

import torch
from point_bridge.robot_utils.fr3.utils import camera2pixelkey

# Create the parser
parser = argparse.ArgumentParser(
    description="Convert processed robot data into a pkl file"
)

# Add the arguments
parser.add_argument("--data_dir", type=str, help="Path to the data directory")
parser.add_argument(
    "--calib_path", type=str, default=None, help="Path to the calibration file"
)
parser.add_argument("--task_names", nargs="+", type=str, help="List of task names")
parser.add_argument(
    "--num_demos", type=int, default=None, help="Number of demonstrations to process"
)

args = parser.parse_args()
DATA_DIR = Path(args.data_dir)
task_names = args.task_names
NUM_DEMOS = args.num_demos

camera_indices = [8]
original_img_size = (640, 480)
crop_h, crop_w = (0.0, 1.0), (0.0, 1.0)
save_img_size = (640, 480)  # (256, 256)
# zed camera
zed_indices = [8]  # Zed does not have depth
zed_image_size = (1280, 720)

PROCESSED_DATA_PATH = Path(DATA_DIR) / "processed_data"
SAVE_DATA_PATH = Path(DATA_DIR) / "expert_demos" / "franka_env"

if save_img_size is None:
    save_img_size = (
        int(original_img_size[0] * (crop_w[1] - crop_w[0])),
        int(original_img_size[1] * (crop_h[1] - crop_h[0])),
    )


if task_names is None:
    task_names = [x.name for x in PROCESSED_DATA_PATH.iterdir() if x.is_dir()]

SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)


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


for TASK_NAME in task_names:
    DATASET_PATH = Path(f"{PROCESSED_DATA_PATH}/{TASK_NAME}")

    if (SAVE_DATA_PATH / f"{TASK_NAME}.pkl").exists():
        print(f"Data for {TASK_NAME} already exists. Appending to it...")
        input("Press Enter to continue...")
        data = pkl.load(open(SAVE_DATA_PATH / f"{TASK_NAME}.pkl", "rb"))
        observations = data["observations"]
        max_cartesian = data["max_cartesian"]
        min_cartesian = data["min_cartesian"]
        max_gripper = data["max_gripper"]
        min_gripper = data["min_gripper"]
    else:
        observations = []
        max_cartesian, min_cartesian = None, None
        max_gripper, min_gripper = None, None

    dirs = [x for x in DATASET_PATH.iterdir() if x.is_dir()]
    for i, data_point in enumerate(sorted(dirs)):
        print(f"Processing data point {i+1}/{len(dirs)}")

        if NUM_DEMOS is not None and int(str(data_point).split("_")[-1]) >= NUM_DEMOS:
            print(f"Skipping data point {data_point}")
            continue

        observation = {}

        # Process images
        image_dir = data_point / "videos"
        if not image_dir.exists():
            print(f"Data point {data_point} is incomplete")
            continue

        for save_idx, idx in enumerate(camera_indices):
            video_path = image_dir / f"camera{idx}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Video {video_path} could not be opened")
                continue

            if idx not in zed_indices:
                frames = []
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
                        frame.shape[0] != save_img_size[1]
                        or frame.shape[1] != save_img_size[0]
                    ):
                        frame = cv2.resize(frame, save_img_size)
                    frames.append(frame)

                observation[f"pixels{idx}"] = np.array(frames)
            else:
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
                    frames_left.append(frame_left)
                    frames_right.append(frame_right)
                observation[f"pixels{idx}_left"] = np.array(frames_left)
                observation[f"pixels{idx}_right"] = np.array(frames_right)

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

        gripper_states = state["gripper_state"].values.astype(np.float32)
        observation["cartesian_states"] = cartesian_states.astype(np.float32)
        observation["gripper_states"] = gripper_states.astype(np.float32)

        cmd_cartesian_states = state["cmd_pose_aa"].values
        cmd_cartesian_states = np.array(
            [
                np.array([extract_number(x) for x in pose.strip("[]").split(",")])
                for pose in cmd_cartesian_states
            ],
            dtype=np.float32,
        )
        cmd_gripper_states = state["cmd_gripper_state"].values.astype(np.float32)
        observation["cmd_cartesian_states"] = cmd_cartesian_states.astype(np.float32)
        observation["cmd_gripper_states"] = cmd_gripper_states.astype(np.float32)

        # Update max and min cartesian values for normalization
        if max_cartesian is None:
            max_cartesian = np.max(cartesian_states, axis=0)
            min_cartesian = np.min(cartesian_states, axis=0)
        else:
            max_cartesian = np.maximum(max_cartesian, np.max(cartesian_states, axis=0))
            min_cartesian = np.minimum(min_cartesian, np.min(cartesian_states, axis=0))

        # Update max and min gripper values for normalization
        if max_gripper is None:
            max_gripper = np.max(gripper_states)
            min_gripper = np.min(gripper_states)
        else:
            max_gripper = np.maximum(max_gripper, np.max(gripper_states))
            min_gripper = np.minimum(min_gripper, np.min(gripper_states))

        observations.append(observation)

    # Save data to a pickle file
    data = {
        "observations": observations,
        "max_cartesian": max_cartesian,
        "min_cartesian": min_cartesian,
        "max_gripper": max_gripper,
        "min_gripper": min_gripper,
    }
    with open(SAVE_DATA_PATH / f"{TASK_NAME}.pkl", "wb") as f:
        pkl.dump(data, f)

print("Processing complete.")
