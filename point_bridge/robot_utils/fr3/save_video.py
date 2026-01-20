# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import sys

sys.path.append("../../")

import pickle as pkl
import numpy as np
import imageio
import einops
from pathlib import Path
import cv2
from point_bridge.robot_utils.fr3.utils import camera2pixelkey

DATA_DIR = Path("/path/to/pointbridge/expert_demos")
benchmark_name = "fr3_nyu"
save_dir_suffix = "robot_data"

SAVE_DIR = Path(f"./videos_{save_dir_suffix}")
save_points = True
k = 0  # number of track points to plot per frame
traj_indices = [0]  # None means all trajectories

camera_indices = [8]
zed_indices = [8]
original_img_size = (640, 480)
zed_image_size = (672, 448)

camera_names = []
for camera_idx in camera_indices:
    if camera_idx not in zed_indices:
        camera_names.append(f"cam_{camera_idx}")
    else:
        camera_names.append(f"cam_{camera_idx}_left")
        camera_names.append(f"cam_{camera_idx}_right")
pixel_keys = [camera2pixelkey[camera_name] for camera_name in camera_names]


# Get task names
task_names = [
    f.stem for f in DATA_DIR.glob(f"{benchmark_name}_{save_dir_suffix}/*.pkl")
]
task_names = [
    "mug_on_plate",
]
num_tasks = len(task_names)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

for task_name in task_names:
    print(f"Processing task {task_name}...")
    DATA_PATH = DATA_DIR / f"{benchmark_name}_{save_dir_suffix}" / f"{task_name}.pkl"

    # Read data
    try:
        with open(DATA_PATH, "rb") as f:
            data = pkl.load(f)

        if traj_indices is None:
            traj_indices = list(range(len(data["observations"])))

        save_frames = {pixel_key: [] for pixel_key in pixel_keys}
        for traj_idx in traj_indices:
            for pixel_key in pixel_keys:
                images = data["observations"][traj_idx][pixel_key]
                images = np.array(images)

                if save_points:
                    key = "pixels8_left" if "left" in pixel_key else "pixels8_right"
                    point_track_key = f"robot_tracks_{key}"
                    object_track_key = f"object_tracks_128_{key}"

                    point_tracks = data["observations"][traj_idx][point_track_key]
                    point_tracks = np.array(point_tracks)
                    object_tracks = data["observations"][traj_idx][object_track_key]
                    object_tracks = np.array(object_tracks)

                    # Ensure all arrays have the same number of steps
                    min_steps = min(len(point_tracks), len(object_tracks), len(images))
                    point_tracks = point_tracks[:min_steps]
                    object_tracks = object_tracks[:min_steps]
                    images = images[:min_steps]

                    object_tracks = einops.rearrange(
                        object_tracks, "n o p d -> n (o p) d"
                    )

                    # Color for each point
                    num_points = point_tracks.shape[1]
                    colors = np.zeros((num_points, 3))
                    third = num_points // 3
                    colors[:third, 0] = 255
                    colors[third : 2 * third, 1] = 255
                    colors[2 * third :, 2] = 255

                # Plot points on images such that each frame has point tracks
                # from the last k frames
                for i, frame in enumerate(images):
                    if save_points:
                        for j, points in enumerate(point_tracks[max(0, i - k) : i + 1]):
                            for l, point in enumerate(points):
                                point = point.astype(int)
                                point[0] = np.clip(point[0], 0, frame.shape[1])
                                point[1] = np.clip(point[1], 0, frame.shape[0])
                                frame = cv2.circle(
                                    frame, tuple(point), 2, colors[l].tolist(), -1
                                )
                    save_frames[pixel_key].append(frame)

        # Save the video
        for pixel_key in pixel_keys:
            save_frames[pixel_key] = np.array(save_frames[pixel_key]).astype(np.uint8)
            (SAVE_DIR / task_name).mkdir(parents=True, exist_ok=True)
            save_path = SAVE_DIR / task_name / f"{pixel_key}.mp4"
            imageio.mimwrite(save_path, save_frames[pixel_key], fps=20)

    except Exception as e:
        print(f"Error processing {task_name}: {e}")
        continue
