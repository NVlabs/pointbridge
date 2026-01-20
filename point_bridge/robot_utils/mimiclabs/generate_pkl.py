# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Script to generate pkl files for expert demonstrations from hdf5 files.
The points on the first frame for each demo is obtained by auto keypoint labeling.
All the points are grounded in the robot's base frame.

Points reduced using Farthest Point Sampling.

"""

import sys

sys.path.append("../../")

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import h5py
import json
import pickle as pkl
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import robosuite as suite
import point_bridge
from mimiclabs.mimiclabs.envs.problems import *
from point_bridge.robot_utils.mimiclabs.sample_object_points import extract_object_mesh, sample_points_from_mesh

import torch
from sentence_transformers import SentenceTransformer

from point_bridge.robot_utils.common.mujoco_transforms import MujocoTransforms
from point_bridge.robot_utils.common.franka_gripper_points import extrapoints
from point_bridge.robot_utils.common.utils import (
    farthest_point_sampling,
    transform_points,
    depthimg2Meters,
    pixel2d_to_3d,
)
from point_bridge.robot_utils.mimiclabs.utils import camera2pixelkey, pixelkey2camera, T_robot_base, T_gripper
from point_bridge.robot_utils.common.camera_utils import (
    add_camera_with_offset,
    add_cameras_from_extrinsics,
)

HOME_DIR = Path(point_bridge.__path__[0]).parent
TASK_NAME = "bowl_on_plate" # bowl_on_plate, mug_on_plate, stack_bowls
save_pixels = False
save_dir_suffix = ""
add_real_cam = True # add real camera from camera extrinsics
add_depth_from_real_cam = True

DATASET_PATH = HOME_DIR / f"data/mimicgen_data/{TASK_NAME}"
SAVE_DATA_PATH = HOME_DIR / "expert_demos"
BDDL_DIR = HOME_DIR / "third_party/mimiclabs/mimiclabs/mimiclabs/task_suites/new_task_suite"

benchmark_name = "mimiclabs"
NUM_DEMOS = None
camera_names = ["agentviewleft", "agentview", "robot0_eye_in_hand"]
cam_extrinsics_path = HOME_DIR / "point_bridge/robot_utils/fr3/calibration_matrices/camera_extrinsics.json"

if add_depth_from_real_cam:
    save_dir_suffix += "_camdepth"

num_points_per_obj = 1000
SAVED_POINTS_PER_OBJ = [128]
img_size = (720, 720)  # (width, height)
pixel_keys = [camera2pixelkey[camera_name] for camera_name in camera_names]

if not save_pixels:
    save_dir_suffix += "_no_images"

if add_real_cam:
    camera_names += ["cam_8_left"]
    pixel_keys += ["pixels_cam_8_left"]
    pixelkey2camera["pixels_cam_8_left"] = "cam_8_left"
    camera2pixelkey["cam_8_left"] = "pixels_cam_8_left"

assert img_size[0] == img_size[1], "Image size must be square"

# create save directory
SAVE_DATA_PATH = (
    SAVE_DATA_PATH / f"{benchmark_name}_{save_dir_suffix}"
    if save_dir_suffix != ""
    else SAVE_DATA_PATH / benchmark_name
)
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# load sentence transformer
print("Initializing Sentence Transformer ...")
lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Sentence Transformer initialized")

# default device
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# get task names
task_names = [f.name for f in DATASET_PATH.iterdir() if f.is_dir()]
task_names.sort()
num_tasks = len(task_names)

bodynames = ["gripper0_eef"]

tasks_stored = 0
save_task_names = []
for task_idx, task_name in enumerate(task_names):
    print(f"Processing Task {task_idx+1}/{len(task_names)}: {task_name}")

    demo_path = DATASET_PATH / task_name / "demo/demo.hdf5"
    data = h5py.File(demo_path, "r")["data"]

    controller_configs = suite.load_controller_config(default_controller="OSC_POSE")
    controller_configs["control_delta"] = True
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
        bddl_file_name=str(BDDL_DIR / f"{task_name}.bddl"),
    )

    env = suite.make(**env_kwargs)
    env.reset()

    observations = []
    actions = []

    # filter and get relevant object names
    object_names = {}
    for obj in env.obj_of_interest:
        object_names[obj] = []
        for body_name in env.sim.model.body_names:
            if body_name is None:
                continue
            # hierarchical body names cause issues
            # so check with mesh function to finalize relevant body names
            if obj not in body_name:
                continue
            try:
                mesh = extract_object_mesh(
                    env, body_name, visualize=False, in_base_frame=True
                )
                if mesh is not None:
                    object_names[obj].append(body_name)
            except:
                pass

    object_points = {}
    for saved_points_per_obj in SAVED_POINTS_PER_OBJ:
        assert (
            saved_points_per_obj > 0
        ), "Saved points per object must be greater than 0"
        save_key_name = f"object_tracks_{saved_points_per_obj}"
        object_points[save_key_name] = {}

    for demo_idx, demo_key in enumerate(data.keys()):
        if NUM_DEMOS is not None and demo_idx >= NUM_DEMOS:
            break

        try:
            print(f"Processing demo {demo_idx + 1}: {demo_key}")
            demo_data = data[demo_key]

            # data variables
            observation = {}

            # add data to observation
            observation["states"] = np.array(demo_data["states"], dtype=np.float32)

            CAMERA_POSES, CAMERA_INTRINSICS, ROBOT_BASES = [], [], []
            observation["robot_tracks_3d"] = []
            for saved_points_per_obj in SAVED_POINTS_PER_OBJ:
                save_key_name = f"object_tracks_{saved_points_per_obj}"
                observation[f"{save_key_name}_3d"] = []
            for camera_name in camera_names:
                pixel_key = camera2pixelkey[camera_name]
                observation[f"robot_tracks_{pixel_key}"] = []
                for saved_points_per_obj in SAVED_POINTS_PER_OBJ:
                    save_key_name = f"object_tracks_{saved_points_per_obj}"
                    observation[f"{save_key_name}_{pixel_key}"] = []
                if save_pixels:
                    observation[pixel_key] = []
            observation["eef_states"], observation["gripper_states"] = [], []

            for step_idx in range(len(demo_data["states"])):
                state = demo_data["states"][0]
                if step_idx == 0:
                    model_file = demo_data.attrs["model_file"]
                    env.reset_to({"states": state, "model": model_file})

                    # Add new camera after reset using utility function
                    camera_name = "agentviewleft"
                    offset = np.array([0, -0.12, 0])  # Offset from agentview camera
                    success = add_camera_with_offset(
                        env, camera_name, "agentview", offset
                    )
                    if success:
                        # Reset to the same state after adding camera
                        env.sim.reset()
                        env.sim.set_state_from_flattened(state)
                        env.sim.forward()

                    # add from extrinsics
                    if add_real_cam:
                        (
                            env,
                            _pixel_keys,
                            _pixelkey2camera,
                        ) = add_cameras_from_extrinsics(
                            env,
                            cam_extrinsics_path,
                            T_robot_base,
                            camera_names=["cam_8_left"],
                        )
                        if "pixels_cam_8_left" not in _pixel_keys:
                            pixel_keys.append("pixels_cam_8_left")
                            camera_names.append("cam_8_left")
                            pixelkey2camera["pixels_cam_8_left"] = "cam_8_left"
                            camera2pixelkey["cam_8_left"] = "pixels_cam_8_left"
                else:
                    obs, reward, done, info = env.step(
                        demo_data["actions"][step_idx - 1]
                    )

                # camera metrics (for every step since camera can move if base moves)
                transforms = MujocoTransforms(
                    env, camera_names, img_size[0], gripper_name=bodynames[0]
                )
                transformation_matrices = transforms.transforms
                camera_poses = transformation_matrices[
                    "camera_projection_matrix"
                ]  # camera in world frame
                camera_intrinsics = transforms.camera_intrinsics

                # robot base pose
                robot_base_pos, robot_base_ori = env.sim.data.get_body_xpos(
                    "robot0_base"
                ), env.sim.data.get_body_xmat("robot0_base")
                robot_base = np.eye(4)  # robot base in world frame
                robot_base[:3, :3] = robot_base_ori
                robot_base[:3, 3] = robot_base_pos
                robot_base = robot_base @ T_robot_base  # FR3 robot base in world frame

                CAMERA_POSES.append(camera_poses)
                CAMERA_INTRINSICS.append(camera_intrinsics)
                ROBOT_BASES.append(robot_base)

                # get 3d robot points
                eef_pos, eef_ori = env.sim.data.get_body_xpos(
                    bodynames[0]
                ), env.sim.data.get_body_xmat(bodynames[0])
                T_g2b = np.eye(4)  # gripper in world
                T_g2b[:3, :3] = eef_ori
                T_g2b[:3, 3] = eef_pos
                T_g2b = T_g2b @ T_gripper  # FR3 gripper in world frame
                T_g2b = (
                    np.linalg.inv(robot_base) @ T_g2b
                )  # FR3 gripper in FR3 robot base
                # add point transform
                points3d_robot = []
                for idx, Tp in enumerate(extrapoints):
                    if demo_data["actions"][step_idx, -1] > 0 and idx in [0, 1]:
                        Tp = Tp.copy()
                        Tp[1, 3] = 0.015 if idx == 0 else -0.015
                    pt = T_g2b @ Tp  # pt in FR3 robot base
                    pt = pt[:3, 3]
                    points3d_robot.append(pt[:3])
                points3d_robot = np.array(points3d_robot)
                observation["robot_tracks_3d"].append(points3d_robot)

                # get 3d points for each object
                for save_idx, saved_points_per_obj in enumerate(SAVED_POINTS_PER_OBJ):
                    save_key_name = f"object_tracks_{saved_points_per_obj}"

                    # compute object points
                    points3d_objects = []
                    for object_name in object_names:
                        body_names = object_names[object_name]
                        num_pts = [num_points_per_obj // len(body_names)] * len(
                            body_names
                        )
                        save_num_pts = [saved_points_per_obj // len(body_names)] * len(
                            body_names
                        )
                        if sum(num_pts) < num_points_per_obj:
                            num_pts[-1] += num_points_per_obj - sum(num_pts)
                        if sum(save_num_pts) < saved_points_per_obj:
                            save_num_pts[-1] += saved_points_per_obj - sum(save_num_pts)
                        body_points = []
                        for body_name, num_pt, save_num_pt in zip(
                            body_names, num_pts, save_num_pts
                        ):
                            # object pose in world frame
                            obj_pos, obj_ori = env.sim.data.get_body_xpos(
                                body_name
                            ), env.sim.data.get_body_xmat(body_name)
                            Tobj = np.eye(4)
                            Tobj[:3, :3], Tobj[:3, 3] = obj_ori, obj_pos
                            Tobj = (
                                np.linalg.inv(robot_base) @ Tobj
                            )  # object in FR3 robot base frame

                            if step_idx == 0:
                                mesh = extract_object_mesh(
                                    env,
                                    body_name,
                                    visualize=False,
                                    in_base_frame=True,
                                )
                                points = sample_points_from_mesh(
                                    mesh, n_points=num_pt, surface_only=True
                                )  # in robot base frame

                                # reduce points using Farthest Point Sampling
                                points = farthest_point_sampling(points, save_num_pt)

                                # compute points in FR3 robot base frame
                                points = transform_points(
                                    points, np.linalg.inv(T_robot_base)
                                )

                                # points in object frame
                                pts = transform_points(points, np.linalg.inv(Tobj))
                                object_points[save_key_name][body_name] = pts
                            else:
                                pts = object_points[save_key_name][
                                    body_name
                                ]  # in object frame
                                points = transform_points(
                                    pts, Tobj
                                )  # in FR3 robot base frame
                            body_points.append(points[:, :3])

                        body_points = np.concatenate(body_points, axis=0)
                        points3d_objects.append(body_points)
                    points3d_objects = np.array(points3d_objects)
                    observation[f"{save_key_name}_3d"].append(points3d_objects)

                    # get 2d points
                    for camera_name in camera_names:
                        pixel_key = camera2pixelkey[camera_name]
                        camera_pose = CAMERA_POSES[step_idx][
                            camera_name
                        ]  # camera in world frame
                        robot_base = ROBOT_BASES[
                            step_idx
                        ]  # FR3 robot base in world frame
                        extr = (
                            np.linalg.inv(camera_pose) @ robot_base
                        )  # FR3 robot base in camera frame
                        intr = CAMERA_INTRINSICS[step_idx][camera_name]

                        r, t = extr[:3, :3], extr[:3, 3]
                        r, _ = cv2.Rodrigues(r)

                        # For robot points
                        if save_idx == 0:
                            points, _ = cv2.projectPoints(
                                points3d_robot, r, t, intr, np.zeros(5)
                            )
                            points = points[:, 0]

                            observation[f"robot_tracks_{pixel_key}"].append(points)

                        # For object points
                        points2d_objects = []
                        for obj_pts in points3d_objects:
                            points, _ = cv2.projectPoints(
                                obj_pts, r, t, intr, np.zeros(5)
                            )
                            points = points[:, 0]
                            points2d_objects.append(points)

                        # Convert 2D points back to 3D using depth when depth noising is enabled
                        if add_real_cam and add_depth_from_real_cam:
                            _, depth_img = env.sim.render(
                                img_size[1],
                                img_size[0],
                                camera_name=camera_name,
                                depth=True,
                            )
                            depth_img = depth_img[::-1]
                            depth_img = depthimg2Meters(env, depth_img)

                            points3d_objects_from_depth = []

                            for obj_idx, obj_pts in enumerate(points3d_objects):
                                # Convert 2D points back to 3D using depth for this object
                                obj_points3d_from_depth = []
                                for i, point_2d in enumerate(points2d_objects[obj_idx]):
                                    u, v = int(point_2d[0]), int(point_2d[1])
                                    # Check bounds
                                    if 0 <= u < img_size[0] and 0 <= v < img_size[1]:
                                        depth = depth_img[v, u]
                                        if depth > 0:  # Valid depth
                                            # Convert 2D point back to 3D in camera frame
                                            point_3d_cam = pixel2d_to_3d(
                                                np.array([[u, v]]),
                                                np.array([depth]),
                                                intr,
                                                np.eye(
                                                    4
                                                ),  # Identity since we want camera frame
                                            )
                                            # Transform back to robot base frame
                                            point_3d_robot = transform_points(
                                                point_3d_cam, np.linalg.inv(extr)
                                            )
                                            obj_points3d_from_depth.append(
                                                point_3d_robot[0]
                                            )
                                        else:
                                            # If no valid depth, use original point
                                            obj_points3d_from_depth.append(obj_pts[i])
                                    else:
                                        # If out of bounds, use original point
                                        obj_points3d_from_depth.append(obj_pts[i])

                                points3d_objects_from_depth.append(
                                    np.array(obj_points3d_from_depth)
                                )

                            # Update object tracks 3D with depth-based points
                            observation[f"{save_key_name}_3d"][-1] = np.array(
                                points3d_objects_from_depth
                            )

                        observation[f"{save_key_name}_{pixel_key}"].append(
                            np.array(points2d_objects)
                        )

                # store eef and gripper states
                eef_pos, eef_ori = env.sim.data.get_body_xpos(
                    bodynames[0]
                ), env.sim.data.get_body_xmat(bodynames[0])
                # shift eef to FR3 robot base frame
                T = np.eye(4)
                T[:3, :3] = eef_ori
                T[:3, 3] = eef_pos
                T = T @ T_gripper  # FR3 gripper in world frame
                T = np.linalg.inv(robot_base) @ T  # FR3 gripper in FR3 robot base frame
                eef_pos, eef_ori = T[:3, 3], T[:3, :3]
                eef_quat = R.from_matrix(eef_ori).as_quat()
                observation["eef_states"].append(np.concatenate([eef_pos, eef_quat]))
                observation["gripper_states"].append(demo_data["actions"][step_idx, -1])

                for camera_name in camera_names:
                    pixel_key = camera2pixelkey[camera_name]
                    # save pixels
                    if save_pixels:
                        frame = env.sim.render(
                            img_size[1], img_size[0], camera_name=camera_name
                        )[::-1]
                        observation[pixel_key].append(frame)

            observation["robot_tracks_3d"] = np.array(observation["robot_tracks_3d"])
            for saved_points_per_obj in SAVED_POINTS_PER_OBJ:
                save_key_name = f"object_tracks_{saved_points_per_obj}"
                observation[f"{save_key_name}_3d"] = np.array(
                    observation[f"{save_key_name}_3d"]
                )

            for camera_name in camera_names:
                pixel_key = camera2pixelkey[camera_name]
                observation[f"robot_tracks_{pixel_key}"] = np.array(
                    observation[f"robot_tracks_{pixel_key}"]
                )
                for saved_points_per_obj in SAVED_POINTS_PER_OBJ:
                    save_key_name = f"object_tracks_{saved_points_per_obj}"
                    observation[f"{save_key_name}_{pixel_key}"] = np.array(
                        observation[f"{save_key_name}_{pixel_key}"]
                    )
                if save_pixels:
                    observation[pixel_key] = np.array(observation[pixel_key])
            observation["eef_states"] = np.array(observation["eef_states"])
            observation["gripper_states"] = np.array(observation["gripper_states"])

            # save data
            if reward == 1.0:
                observations.append(observation)
                actions.append(np.array(demo_data["actions"], dtype=np.float32))
            else:
                print(f"Skipping demo {demo_key} because final reward is {reward}")
        except Exception as e:
            print(f"Error processing demo {demo_key}: {e}")
            continue

    # save data
    save_data_path = SAVE_DATA_PATH
    save_data_path.mkdir(parents=True, exist_ok=True)
    save_data_path = save_data_path / f"{task_name}.pkl"
    save_data = {
        "observations": observations,
        "actions": actions,
        "task_desc": env.language_instruction,
        "task_emb": lang_model.encode(env.language_instruction),
        "robot_base": ROBOT_BASES[0],
    }
    with open(save_data_path, "wb") as f:
        pkl.dump(save_data, f)

    print(f"Saved to {str(save_data_path)}")

    env.close()
    tasks_stored += 1
    save_task_names.append(task_name)
