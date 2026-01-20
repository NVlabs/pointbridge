# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np
import open3d as o3d
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.transform import Rotation as R
import robosuite.utils.transform_utils as T
import json


def add_camera_to_env(
    env,
    camera_name: str,
    pos: Union[List[float], np.ndarray],
    quat: Union[List[float], np.ndarray],
    mode: str = "fixed",
) -> bool:
    """
    Add a new camera to the environment after it has been reset to a dataset state.

    Args:
        env: The robosuite environment
        camera_name: Name of the new camera
        pos: Position of the camera [x, y, z]
        quat: Quaternion orientation of the camera [w, x, y, z]
        mode: Camera mode ("fixed" or "track")

    Returns:
        bool: True if camera was added successfully, False if it already exists
    """
    # Check if camera already exists
    existing_cameras = []
    for i in range(env.sim.model.ncam):
        existing_cameras.append(env.sim.model.camera_id2name(i))

    if camera_name in existing_cameras:
        print(f"Camera {camera_name} already exists in model")
        return False

    # Get the current XML
    current_xml = env.sim.model.get_xml()
    root = ET.fromstring(current_xml)
    worldbody = root.find("worldbody")

    # Create new camera element
    new_camera = ET.Element("camera")
    new_camera.set("name", camera_name)
    # new_camera.set("mode", mode)

    # Convert position and quaternion to strings
    pos_str = f"{pos[0]} {pos[1]} {pos[2]}"
    quat_str = f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"

    new_camera.set("pos", pos_str)
    new_camera.set("quat", quat_str)

    # Add camera to worldbody
    worldbody.append(new_camera)

    # Convert back to string and reload the model
    new_xml = ET.tostring(root, encoding="utf8").decode("utf8")
    env.reset_from_xml_string(new_xml)

    print(f"Added new camera: {camera_name}")
    return True


def update_observation_system(env, camera_name: str) -> bool:
    """
    Update the environment's observation system to include the new camera.
    This is a more comprehensive approach that manually adds the camera sensor.

    Args:
        env: The robosuite environment
        camera_name: Name of the camera to add to observations

    Returns:
        bool: True if camera was added to observations successfully
    """
    try:
        # Check if camera exists in model
        cam_id = env.sim.model.camera_name2id(camera_name)

        # Update camera names list if it exists
        if hasattr(env, "camera_names"):
            if camera_name not in env.camera_names:
                env.camera_names.append(camera_name)
                print(f"Added {camera_name} to camera_names")

        # Update camera heights and widths if they exist
        if hasattr(env, "camera_heights"):
            if isinstance(env.camera_heights, list):
                env.camera_heights.append(
                    env.camera_heights[0]
                )  # Use same height as first camera
            else:
                env.camera_heights = [env.camera_heights, env.camera_heights]

        if hasattr(env, "camera_widths"):
            if isinstance(env.camera_widths, list):
                env.camera_widths.append(
                    env.camera_widths[0]
                )  # Use same width as first camera
            else:
                env.camera_widths = [env.camera_widths, env.camera_widths]

        # Manually add camera sensor to observables
        if hasattr(env, "observables"):
            # Create a new camera sensor
            from robosuite.utils.observables import Observable

            def camera_obs(obs_cache):
                return env.sim.render(
                    camera_name=camera_name,
                    width=env.camera_widths[0]
                    if isinstance(env.camera_widths, list)
                    else env.camera_widths,
                    height=env.camera_heights[0]
                    if isinstance(env.camera_heights, list)
                    else env.camera_heights,
                    depth=False,
                )[::-1]

            # Add the new camera observable
            env.observables[f"{camera_name}_image"] = Observable(
                name=f"{camera_name}_image",
                sensor=camera_obs,
                sampling_rate=env.control_freq,
            )

            print(f"Added {camera_name}_image to observables")

        return True

    except Exception as e:
        print(f"Error updating observation system for {camera_name}: {e}")
        return False


def update_camera_config(env, camera_name: str) -> bool:
    """
    Update the environment's camera configuration to include the new camera in observations.

    Args:
        env: The robosuite environment
        camera_name: Name of the camera to add to observations

    Returns:
        bool: True if camera was added to observations successfully
    """
    try:
        # Check if camera exists in model
        cam_id = env.sim.model.camera_name2id(camera_name)

        # Update camera names list if it exists
        if hasattr(env, "camera_names"):
            if camera_name not in env.camera_names:
                env.camera_names.append(camera_name)
                print(f"Added {camera_name} to camera_names")

        # Update camera heights and widths if they exist
        if hasattr(env, "camera_heights"):
            if isinstance(env.camera_heights, list):
                env.camera_heights.append(
                    env.camera_heights[0]
                )  # Use same height as first camera
            else:
                env.camera_heights = [env.camera_heights, env.camera_heights]

        if hasattr(env, "camera_widths"):
            if isinstance(env.camera_widths, list):
                env.camera_widths.append(
                    env.camera_widths[0]
                )  # Use same width as first camera
            else:
                env.camera_widths = [env.camera_widths, env.camera_widths]

        # Force observation update
        if hasattr(env, "_get_observations"):
            # This will force the environment to rebuild its observation sensors
            env._setup_observables()
            print(f"Updated observation system to include {camera_name}")

        return True

    except Exception as e:
        print(f"Error updating camera config for {camera_name}: {e}")
        return False


def get_camera_attributes(env, camera_name: str):
    """
    Get camera position and orientation using the correct attribute names.
    """
    pos = env.sim.data.get_camera_xpos(camera_name)
    mat = env.sim.data.get_camera_xmat(camera_name)
    quat = T.convert_quat(T.mat2quat(mat), to="wxyz")  # T.mat2quat returns xyzw
    return pos, quat


def add_camera_with_offset(
    env,
    camera_name: str,
    base_camera_name: str = "agentview",
    offset: Optional[np.ndarray] = None,
) -> bool:
    """
    Add a new camera based on an existing camera with an offset.

    Args:
        env: The robosuite environment
        camera_name: Name of the new camera
        base_camera_name: Name of the existing camera to base the new one on
        offset: Position offset from the base camera [x, y, z]

    Returns:
        bool: True if camera was added successfully, False otherwise
    """
    # Ensure environment state is properly set
    env.sim.forward()

    # Get base camera position and orientation
    base_pos, base_quat = get_camera_attributes(
        env, base_camera_name
    )  # base_quat is wxyz

    # convert base_pos to robot base frame
    robot_base_pos, robot_base_ori = env.sim.data.get_body_xpos(
        "robot0_base"
    ), env.sim.data.get_body_xmat("robot0_base")
    robot_base = np.eye(4)  # robot base in world frame
    robot_base[:3, :3] = robot_base_ori
    robot_base[:3, 3] = robot_base_pos

    # base_pos is in world frame, convert to robot base frame
    base_pos = np.linalg.inv(robot_base) @ np.concatenate([base_pos, [1]])
    base_pos = base_pos[:3]

    # Apply offset if provided
    if offset is not None:
        new_pos = base_pos + offset
    else:
        new_pos = base_pos

    # convert new pos to world frame
    new_pos = robot_base @ np.concatenate([new_pos, [1]])
    new_pos = new_pos[:3]

    # Add camera to model
    success = add_camera_to_env(
        env, camera_name, new_pos, base_quat
    )  # base_quat is wxyz

    if success:
        # Update observation system to include the new camera
        update_observation_system(env, camera_name)

    return success


def add_cameras_from_extrinsics(
    env, camera_extrinsics_file, T_robot_base, camera_names=None
):
    # robot base in world frame
    robot_base_pos = env.sim.data.get_body_xpos("robot0_base")
    robot_base_ori = env.sim.data.get_body_xmat("robot0_base")
    robot_base = np.eye(4)
    robot_base[:3, 3] = robot_base_pos
    robot_base[:3, :3] = robot_base_ori
    robot_base = robot_base @ T_robot_base  # FR3 robot base in world frame

    # read camera extrinsics file
    camera_extrinsics_json = {}
    camera_extrinsics = {}  # camera in world frame
    pixel_keys, _pixelkey2camera = [], {}
    state = env.sim.get_state().flatten()
    with open(camera_extrinsics_file, "r") as f:
        camera_extrinsics_json = json.load(f)
    for camera_name, extrinsics in camera_extrinsics_json.items():
        if camera_names is not None and camera_name not in camera_names:
            continue
        translation = np.array(
            extrinsics["translation"]
        ).flatten()  # Convert to 1D array
        rotation = np.array(extrinsics["rotation"])
        extrinsic_matrix = np.eye(4, dtype=np.float32)  # camera in FR3 robot base frame
        extrinsic_matrix[:3, :3] = rotation
        extrinsic_matrix[:3, 3] = translation  # + np.array([0.3, 0, 0])

        # match camera axes between sim and real
        T_transform = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )  # sim camera axes in real world frame
        extrinsic_matrix = extrinsic_matrix @ T_transform  # camera in world frame

        extrinsic_matrix = robot_base @ extrinsic_matrix  # camera in world frame
        camera_extrinsics[camera_name] = extrinsic_matrix

        pos = extrinsic_matrix[:3, 3]
        quat = R.from_matrix(extrinsic_matrix[:3, :3]).as_quat()  # quat is xyzw
        quat = T.convert_quat(
            quat, to="wxyz"
        )  # R assumes xyzw and the function needs wxyz
        success = add_camera_to_env(env, camera_name, pos, quat)
        if success:
            # update_observation_system(env, camera_name)
            # Reset to the same state after adding camera
            env.sim.reset()
            env.sim.set_state_from_flattened(state)
            env.sim.forward()

            pixel_keys.append(f"pixels_{camera_name}")
            _pixelkey2camera[f"pixels_{camera_name}"] = camera_name

    return env, pixel_keys, _pixelkey2camera


def get_camera_info(env, camera_name: str) -> Optional[Dict]:
    """
    Get information about a camera in the environment.

    Args:
        env: The robosuite environment
        camera_name: Name of the camera

    Returns:
        Dict with camera info or None if camera doesn't exist
    """
    # Ensure environment state is properly set
    env.sim.forward()

    try:
        cam_id = env.sim.model.camera_name2id(camera_name)
        pos, quat = get_camera_attributes(env, camera_name)

        if pos is None or quat is None:
            return None

        return {"name": camera_name, "id": cam_id, "pos": pos, "quat": quat}
    except Exception as e:
        # Try case-insensitive search
        existing_cameras = list_cameras(env)
        possible_names = [
            camera_name,
            camera_name.lower(),
            camera_name.upper(),
            camera_name.capitalize(),
        ]

        for name in possible_names:
            if name in existing_cameras:
                try:
                    cam_id = env.sim.model.camera_name2id(name)
                    pos, quat = get_camera_attributes(env, name)

                    if pos is not None and quat is not None:
                        return {"name": name, "id": cam_id, "pos": pos, "quat": quat}
                except:
                    continue

        print(f"Error getting camera info for {camera_name}: {e}")
        return None


def list_cameras(env) -> List[str]:
    """
    List all cameras in the environment.

    Args:
        env: The robosuite environment

    Returns:
        List of camera names
    """
    cameras = []
    for i in range(env.sim.model.ncam):
        cameras.append(env.sim.model.camera_id2name(i))
    return cameras
