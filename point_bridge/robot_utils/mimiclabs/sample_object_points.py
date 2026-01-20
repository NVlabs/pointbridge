# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Script to extract a specific object from a mujoco environment, convert it to trimesh,
and sample points from its mesh in either the mujoco world frame or robot base frame.
"""

import sys

sys.path.append("../../")

import os
import numpy as np
import trimesh

import robosuite

if hasattr(robosuite, "__version__") and robosuite.__version__ >= "1.5.0":
    from robosuite.controllers.composite.composite_controller_factory import (
        load_composite_controller_config,
    )
else:
    from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T

from point_bridge.robot_utils.mimiclabs.robosuite_to_trimesh import (
    get_geom_info_from_model,
    get_geom_pose_info_from_data,
    create_trimesh_scene_from_mujoco,
)


def create_full_scene(env):
    """
    Create a trimesh scene containing all objects in the environment.
    """
    mj_geom_model_infos, _ = get_geom_info_from_model(env=env)
    mj_geom_pose_infos = get_geom_pose_info_from_data(
        env=env, geom_model_infos=mj_geom_model_infos
    )

    return create_trimesh_scene_from_mujoco(
        env=env,
        geom_model_infos=mj_geom_model_infos,
        geom_pose_infos=mj_geom_pose_infos,
    )


def extract_object_mesh(env, object_name, visualize=False, in_base_frame=False):
    """
    Extract the trimesh representation of a specific object from the mujoco environment.

    Args:
        env: Robosuite environment
        object_name (str): Name of the object to extract (this should match the body name in mujoco)
        visualize (bool): If True, will display the extracted mesh using trimesh viewer
        in_base_frame (bool): If True, returns mesh in robot base frame. If False, returns in world frame.

    Returns:
        trimesh.Trimesh: The mesh of the requested object in either world frame or robot base frame
    """
    # Get all geom information from mujoco
    mj_geom_model_infos, mj_body_to_geom_info_inds = get_geom_info_from_model(env=env)
    mj_geom_pose_infos = get_geom_pose_info_from_data(
        env=env, geom_model_infos=mj_geom_model_infos, in_base_frame=in_base_frame
    )

    # Check if the object exists
    if object_name not in mj_body_to_geom_info_inds:
        raise ValueError(
            f"Object {object_name} not found in environment. Available objects: {list(mj_body_to_geom_info_inds.keys())}"
        )

    # Extract only the meshes corresponding to our target object
    object_geom_indices = mj_body_to_geom_info_inds[object_name]
    object_geoms = [mj_geom_model_infos[i] for i in object_geom_indices]
    object_poses = [mj_geom_pose_infos[i] for i in object_geom_indices]

    # Create a scene with just this object
    object_scene = create_trimesh_scene_from_mujoco(
        env=env,
        geom_model_infos=object_geoms,
        geom_pose_infos=object_poses,
    )

    # Combine all meshes into a single mesh
    meshes = list(object_scene.geometry.values())
    if len(meshes) == 0:
        raise ValueError(f"No valid meshes found for object {object_name}")
    elif len(meshes) == 1:
        object_mesh = meshes[0]
    else:
        object_mesh = trimesh.util.concatenate(meshes)

    # Visualize if requested
    if visualize:
        object_scene.show()

    return object_mesh


def sample_points_from_mesh(mesh, n_points, surface_only=True):
    """
    Sample points from a mesh. Points will be in the same frame as the input mesh.

    Args:
        mesh (trimesh.Trimesh): The mesh to sample from
        n_points (int): Number of points to sample
        surface_only (bool): If True, samples points only from the surface.
                           If False, samples points from the volume.

    Returns:
        np.ndarray: Array of shape (n_points, 3) containing the sampled points
    """
    if surface_only:
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
    else:
        points = trimesh.sample.volume_mesh(mesh, n_points)

    return points


def visualize_points_with_scene(
    env, points, point_color=[1.0, 0.0, 0.0, 1.0], point_radius=0.002
):
    """
    Visualize points overlaid on the full scene.

    Args:
        env: Robosuite environment
        points (np.ndarray): Points to visualize, shape (N, 3)
        point_color (list): RGBA color for points
        point_radius (float): Radius for spheres representing points
    """
    # Create full scene
    scene = create_full_scene(env)

    # Create small spheres for each point
    for point in points:
        sphere = trimesh.creation.icosphere(radius=point_radius)
        sphere.apply_translation(point)
        sphere.visual.face_colors = point_color
        scene.add_geometry(sphere)

    # Show scene
    scene.show()


def main():
    TASK_NAME = "PnPCounterToSink"
    object_name = "obj_main"

    # create env
    env = create_env(
        env_name=TASK_NAME,
        render_onscreen=False,
        seed=10,  # None,
    )
    env.reset()

    # Dump model XML to file for debugging
    xml_path = f"{TASK_NAME}_model.xml"
    with open(xml_path, "w") as f:
        f.write(env.sim.model.get_xml())
    print(f"Dumped model XML to {xml_path}")

    # Extract object mesh - try both world frame and base frame
    try:
        # Get mesh in world frame and sample points
        world_mesh = extract_object_mesh(
            env, object_name, visualize=False, in_base_frame=False
        )
        world_points = sample_points_from_mesh(
            world_mesh, n_points=1000, surface_only=True
        )
        import ipdb

        ipdb.set_trace()
        print(f"Sampled {len(world_points)} points in world frame")

        # Visualize points overlaid on full scene
        print("Showing points overlaid on full scene (points in red):")
        visualize_points_with_scene(env, world_points, point_color=[1.0, 0.0, 0.0, 1.0])

        # Get mesh in base frame and sample points
        base_mesh = extract_object_mesh(
            env, object_name, visualize=False, in_base_frame=True
        )
        base_points = sample_points_from_mesh(
            base_mesh, n_points=1000, surface_only=True
        )
        print(f"Sampled {len(base_points)} points in base frame")

        # Visualize base frame points (need to transform back to world frame for visualization)
        base_pos = env.robots[0].base_pos
        base_rot = env.robots[0].base_ori
        if base_rot.shape[0] == 4:
            base_rot = T.quat2mat(base_rot)
        base_pose = T.make_pose(base_pos, base_rot)
        base_points_homog = np.column_stack([base_points, np.ones(len(base_points))])
        world_points_from_base = (base_pose @ base_points_homog.T).T[:, :3]

        print(
            "Showing base frame points transformed back to world frame (points in blue):"
        )
        visualize_points_with_scene(
            env, world_points_from_base, point_color=[0.0, 0.0, 1.0, 1.0]
        )

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
