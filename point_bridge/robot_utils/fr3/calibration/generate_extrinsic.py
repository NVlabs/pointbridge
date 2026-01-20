# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
A script which given camera intrinsics computes te robot to camera transformation
for each camera and uses that as extrinsics to save in a calib.pkl file
"""

import cv2
from cv2 import aruco
import json
import numpy as np
import pickle as pkl
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import imageio

# Configuration flags
SAVE_VIDEOS = True
VIDEO_DIR = Path("videos")  # Directory to save debug videos

PATH_DATA_PKL = Path("/path/to/calibration.pkl")
PATH_INTRINSICS = None
SAVE_DIR = Path("calib")
PATH_SAVE_CALIB = SAVE_DIR / "calib.json"
CAM_IDS = [8]
zed_indices = [8]
R2C_TRAJ_IDX = 0
FRAME_FREQ = 1

# Create video directory if saving videos
if SAVE_VIDEOS:
    VIDEO_DIR.mkdir(exist_ok=True, parents=True)


with open(PATH_DATA_PKL, "rb") as f:
    observations = pkl.load(f)["observations"]

if PATH_INTRINSICS is not None and PATH_INTRINSICS.exists():
    print("Using intrinsics from file")
    with open(PATH_INTRINSICS, "rb") as f:
        intrinsics = pkl.load(f)
else:
    print("Using intrinsics from constants")
    from constants import CAMERA_MATRICES, DISTORTION_COEFFICIENTS

    intrinsics = {
        "camera_matrices": {},
        "distortion_coefficients": {},
    }
    for cam_id in CAM_IDS:
        intrinsics["camera_matrices"][f"cam_{cam_id}"] = CAMERA_MATRICES[
            f"cam_{cam_id}"
        ]
        intrinsics["distortion_coefficients"][
            f"cam_{cam_id}"
        ] = DISTORTION_COEFFICIENTS[f"cam_{cam_id}"]

SAVE_DIR.mkdir(exist_ok=True, parents=True)

################################# compute the robot to camera transformation #################################

calibration_dict = {}

for cam_id in CAM_IDS:
    if cam_id in zed_indices:
        images = observations[R2C_TRAJ_IDX][f"pixels{cam_id}_left"][
            ..., ::-1
        ]  # Use left image for shape
        shape = images.shape
        pixels = {
            "left": observations[R2C_TRAJ_IDX][f"pixels{cam_id}_left"][..., ::-1],
            "right": observations[R2C_TRAJ_IDX][f"pixels{cam_id}_right"][..., ::-1],
        }
    else:
        images = observations[R2C_TRAJ_IDX][f"pixels{cam_id}"][..., ::-1]
        shape = images.shape
        pixels = {"left": images}

    # Setup frame collection for video creation
    frame_collections = {}
    if SAVE_VIDEOS:
        for pixel_key in pixels.keys():
            frame_collections[pixel_key] = []
            print(f"Will collect frames for cam_{cam_id}_{pixel_key} video")

    # object point transformations
    object_pos = observations[R2C_TRAJ_IDX]["cartesian_states"][:, :3].copy()
    object_aa = observations[R2C_TRAJ_IDX]["cartesian_states"][:, 3:].copy()
    object_rot_mat = R.from_rotvec(object_aa).as_matrix()
    object_trans = np.zeros(
        (object_pos.shape[0], 4, 4)
    )  # pose of gripper in robot base frame
    object_trans[:, :3, :3] = object_rot_mat
    object_trans[:, :3, 3] = object_pos

    # compute object points
    T_a_g = np.array([[1, 0, 0, 0.0025], [0, 1, 0, 0], [0, 0, 1, 0.0625], [0, 0, 0, 1]])

    object_pts = [T @ T_a_g for T in object_trans]
    object_pts = np.array(object_pts)
    object_points = object_pts[:, :3, 3]

    # Aruco marker detection with Cv2 on pixels
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    image_points = {key: [] for key in pixels.keys()}
    invalid_indices = {key: [] for key in pixels.keys()}
    for idx in range(0, len(pixels["left"]), FRAME_FREQ):
        for pixel_key in pixels.keys():
            frame = pixels[pixel_key][idx].copy()
            corners, ids, rejectedImgPoints = detector.detectMarkers(frame)

            if corners:
                center_img = corners[0].mean(axis=1).flatten()
                image_points[pixel_key].append(center_img)

                # Draw ArUco detection for debugging
                if SAVE_VIDEOS:
                    # Draw detected markers
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    # Draw center point
                    center = tuple(map(int, center_img))
                    cv2.circle(frame, center, 5, (0, 255, 0), -1)
                    # Add text with frame info
                    cv2.putText(
                        frame,
                        f"Frame: {idx}, Detected: {len(corners)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
            else:
                invalid_indices[pixel_key].append(idx)
                # Draw "No Detection" text for debugging
                if SAVE_VIDEOS:
                    cv2.putText(
                        frame,
                        f"Frame: {idx}, No Detection",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            # Collect frame for video creation if saving videos
            if SAVE_VIDEOS and pixel_key in frame_collections:
                # Ensure frame is in correct format (RGB, uint8) for imageio
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                # Convert BGR to RGB for imageio
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif len(frame.shape) == 3 and frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                # Add frame to collection
                frame_collections[pixel_key].append(frame)

    for pixel_key in pixels.keys():
        print(
            f"# invalid frames for {pixel_key}: {len(invalid_indices[pixel_key])}/{len(pixels[pixel_key])}"
        )

    # remove invalid indices from subsampled object points
    object_points = object_points[::FRAME_FREQ]
    object_points_new = {}
    for key in pixels.keys():
        for i in range(len(object_points)):
            if i not in invalid_indices[key]:
                object_points_new[key].append(object_points[i])
        object_points_new[key] = np.array(object_points_new[key])
    object_points = object_points_new

    # convert to numpy float arrays
    object_points = {
        key: np.array(object_points[key].copy()).astype(np.float32)
        for key in pixels.keys()
    }
    image_points = {
        key: np.array(image_points[key].copy()).astype(np.float32)
        for key in pixels.keys()
    }

    # get T_ci_b
    camera_matrix = intrinsics["camera_matrices"][f"cam_{cam_id}"]
    dist_coeffs = intrinsics["distortion_coefficients"][f"cam_{cam_id}"]
    T_ci_b = (
        {key: np.eye(4) for key in pixels.keys()}
        if len(pixels.keys()) > 1
        else np.eye(4)
    )
    for pixel_key in pixels.keys():
        cam_mat = (
            camera_matrix[pixel_key]
            if isinstance(camera_matrix, dict)
            else camera_matrix
        )
        dist_coeff = (
            dist_coeffs[pixel_key] if isinstance(dist_coeffs, dict) else dist_coeffs
        )
        ret, rvec, tvec = cv2.solvePnP(
            object_points[pixel_key],
            image_points[pixel_key],
            cam_mat,
            dist_coeff,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        rot = cv2.Rodrigues(rvec)[0]
        if len(pixels.keys()) > 1:
            T_ci_b[pixel_key][:3, :3] = rot
            T_ci_b[pixel_key][
                :3, 3
            ] = tvec.flatten()  # these are extrinsics (world in camera frame)
        else:
            T_ci_b[:3, :3] = rot
            T_ci_b[:3, 3] = tvec.flatten()

    # save intrinsics and extrinsics in a dictionary
    intrinsics_matrix = intrinsics["camera_matrices"][f"cam_{cam_id}"]
    dist_coeffs = intrinsics["distortion_coefficients"][f"cam_{cam_id}"]
    if len(pixels.keys()) > 1:
        extrinsics_matrix = {
            key: np.linalg.inv(T_ci_b[key]) for key in T_ci_b.keys()
        }  # camera in robot base frame
    else:
        extrinsics_matrix = np.linalg.inv(T_ci_b)

    if len(pixels.keys()) > 1:
        calibration_dict[f"cam_{cam_id}"] = {
            "int": {
                key: intrinsics_matrix[key].tolist() for key in intrinsics_matrix.keys()
            },
            "dist_coeff": {
                key: dist_coeffs[key].tolist() for key in dist_coeffs.keys()
            },
            "ext": {
                key: extrinsics_matrix[key].tolist() for key in extrinsics_matrix.keys()
            },
        }
    else:
        calibration_dict[f"cam_{cam_id}"] = {
            "int": intrinsics_matrix.tolist(),
            "dist_coeff": dist_coeffs.tolist(),
            "ext": extrinsics_matrix.tolist(),
        }

    # Create videos from collected frames
    if SAVE_VIDEOS:
        for pixel_key in frame_collections.keys():
            if len(frame_collections[pixel_key]) > 0:
                video_path = VIDEO_DIR / f"cam_{cam_id}_{pixel_key}_aruco_debug.mp4"
                try:
                    # Write video using imageio at 20 FPS
                    imageio.mimwrite(
                        str(video_path), frame_collections[pixel_key], fps=20
                    )
                    print(
                        f"Video saved for cam_{cam_id}_{pixel_key}: {video_path} ({len(frame_collections[pixel_key])} frames)"
                    )
                except Exception as e:
                    print(f"Error creating video for cam_{cam_id}_{pixel_key}: {e}")
            else:
                print(f"No frames collected for cam_{cam_id}_{pixel_key}")

with open(PATH_SAVE_CALIB, "w") as f:
    json.dump(calibration_dict, f, indent=4)

print(f"Calibration saved to {PATH_SAVE_CALIB}")
if SAVE_VIDEOS:
    print(f"Debug videos saved to {VIDEO_DIR}")
