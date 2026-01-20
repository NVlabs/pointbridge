# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import pickle as pkl
import numpy as np
import imageio
import einops
from pathlib import Path
import cv2

DATA_DIR = Path("/path/to/pointbridge/expert_demos")
benchmark_name = "mimiclabs"
save_dir_suffix = ""  # TODO: fill name here

DATA_DIR = DATA_DIR / f"{benchmark_name}_{save_dir_suffix}"
SAVE_DIR = Path(f"./videos/{save_dir_suffix}")
pixel_keys = ["pixels_left", "pixels_right"]
save_points = False
k = 0  # number of track points to plot per frame
TRAJ_INDICES = None

# Get task names
task_names = [f.stem for f in DATA_DIR.iterdir() if f.is_file() and f.suffix == ".pkl"]
num_tasks = len(task_names)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

for task_name in task_names:
    print(f"Processing task {task_name}...")
    CONFIGS = [DATA_DIR / f"{task_name}.pkl"]

    for config_idx, config_path in enumerate(CONFIGS):
        # Read data
        try:
            with open(config_path, "rb") as f:
                data = pkl.load(f)

            if TRAJ_INDICES is None:
                traj_indices = list(range(len(data["observations"])))
            else:
                traj_indices = TRAJ_INDICES

            save_frames = {pixel_key: [] for pixel_key in pixel_keys}
            for traj_idx in traj_indices:
                for pixel_key in pixel_keys:
                    images = data["observations"][traj_idx][pixel_key]
                    images = np.array(images)

                    if save_points:
                        point_track_key = f"robot_tracks_{pixel_key}"
                        object_track_key = f"object_tracks_128_{pixel_key}"

                        point_tracks = data["observations"][traj_idx][point_track_key]
                        point_tracks = np.array(point_tracks)
                        object_tracks = data["observations"][traj_idx][object_track_key]
                        object_tracks = np.array(object_tracks)
                        object_tracks = einops.rearrange(
                            object_tracks, "n o p d -> n (o p) d"
                        )
                        point_tracks = np.concatenate(
                            [point_tracks, object_tracks], axis=1
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
                            for j, points in enumerate(
                                point_tracks[max(0, i - k) : i + 1]
                            ):
                                for l, point in enumerate(points):
                                    point = point.astype(int)
                                    frame = cv2.circle(
                                        frame, tuple(point), 3, colors[l].tolist(), -1
                                    )
                        save_frames[pixel_key].append(frame)

            # Save the video
            for pixel_key in pixel_keys:
                (SAVE_DIR / task_name).mkdir(parents=True, exist_ok=True)
                save_path = SAVE_DIR / f"{task_name}" / f"{config_idx}_{pixel_key}.mp4"
                imageio.mimwrite(save_path, save_frames[pixel_key], fps=20)

        except Exception as e:
            print(f"Error processing {task_name}, {config_path.stem}: {e}")
            continue
