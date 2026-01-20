# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Minimal example to sync robosuite world to trimesh for visualization.

NOTE: mujoco version should be 3.2 or higher.
NOTE: pyglet version tested is 1.5.23
"""
import os
import json
import random
import string
import colorsys
import numpy as np
import traceback
import xml.etree.ElementTree as ET

import mujoco
import trimesh

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import string_to_array

# environment name
ENV_NAME = "Coffee_D0"


def get_fake_name(existing_names):
    """
    Make a random name, ensuring it is not included in the list of existing_names.
    """
    while True:
        new_name = "".join(random.choices(string.ascii_letters, k=8))
        if new_name not in existing_names:
            return new_name


def convert_world_pose_to_base_frame(pose, base_pose_inv=None, env=None):
    """
    Converts pose in mujoco world frame to base frame.
    """
    assert (base_pose_inv is not None) or (env is not None)
    assert (base_pose_inv is None) or (env is None)
    if base_pose_inv is None:
        assert len(env.robots) == 1, "only one robot supported for now"
        base_pos = env.robots[0].base_pos
        base_rot = env.robots[0].base_ori
        if base_rot.shape[0] == 4:
            # hack to be robust to robosuite version - this is sometimes a rotation matrix and sometimes a quaternion
            base_rot = T.quat2mat(base_rot)
        base_pose_inv = T.pose_inv(T.make_pose(base_pos, base_rot))
    return np.matmul(base_pose_inv, pose)


def read_byte_string_until_zero(byte_string, start_index=0):
    """
    Read byte string until 0 termination and convert result to string.
    """
    result = bytearray()
    for i in range(start_index, len(byte_string)):
        byte = byte_string[i]
        if byte == 0:
            break
        result.append(byte)
    return result.decode("utf-8")


def get_visual_info_from_model(env):
    """
    Parse mujoco model to get texture and material information.

    Returns:
        texture_infos (dict): maps texture name to dictionary of attributes corresponding to that texture
            in the MJCF file.

        material_infos (dict): maps material name to dictionary of attributes corresponding to that material
            in the MJCF file.

        material_name_to_texture_name (dict): maps material name to a corresponding texture name, or None if no texture

        material_name_to_rgba (dict): maps material name to rgba (only if no texture, else None)
    """

    # get xml string
    xml_str = env.sim.model.get_xml()

    # parse xml string to get texture and material info
    # based on https://github.com/ARISE-Initiative/robosuite/blob/v1.3/robosuite/renderers/nvisii/parser.py
    xml_root = ET.fromstring(xml_str)

    # texture name to attributes
    texture_infos = dict()

    # material name to attributes
    material_infos = dict()

    # material name to texture name
    material_name_to_texture_name = dict()

    # material name to rgba
    material_name_to_rgba = dict()

    # note: we ignore textures without a name (since they will not be used as a material)
    for texture in xml_root.iter("texture"):
        texture_name = texture.get("name")
        if texture_name is not None:
            texture_infos[texture_name] = texture.attrib

    for material in xml_root.iter("material"):
        material_name = material.get("name")
        texture_name = material.get("texture")
        rgba = material.get("rgba")
        # assert (material_name is not None) and (texture_name is not None)
        if texture_name is not None:
            material_name_to_texture_name[material_name] = texture_name
            material_name_to_rgba[material_name] = None
        else:
            material_name_to_texture_name[material_name] = None
            # assert rgba is not None
            if rgba is not None:
                rgba = list(string_to_array(rgba))
            material_name_to_rgba[material_name] = rgba

        material_infos[material_name] = material.attrib

    return (
        texture_infos,
        material_infos,
        material_name_to_texture_name,
        material_name_to_rgba,
    )


def get_geom_info_from_model(
    env, exclude_geom_prefixes=("robot", "gripper", "mobilebase"), verbose=False
):  # "mount"
    """
    Parse mujoco model from robosuite environment to get geom information that should be imported
    into curobo.

    Returns:
        geom_model_infos (list of dict): dict per geom after filtering out geoms that
            are purely visual or ones that correspond to the robot model (since those should be excluded
            from curobo).

        body_to_geom_info_inds (dict): dictionary mapping body name to list of indices
            into @geom_model_infos, to  easily keep track of groups of geoms that correspond to a body.
    """

    # construct useful variables from mujoco model
    ngeoms = env.sim.model._model.ngeom

    # geom name to ID
    geom_name_to_id = env.sim.model._geom_name2id

    # geom ID to name
    geom_id_to_name = env.sim.model._geom_id2name

    # geom ID to type (e.g. mesh, box, sphere, etc...)
    geom_id_to_type = env.sim.model._model.geom_type

    # geom type to string and vice-versa, for supported geom types in our conversion
    geom_type_to_string = {
        # mujoco.mjtGeom.mjGEOM_PLANE : "plane",
        mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
        mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
        mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
        mujoco.mjtGeom.mjGEOM_BOX: "box",
        mujoco.mjtGeom.mjGEOM_MESH: "mesh",
    }
    geom_string_to_type = {geom_type_to_string[k]: k for k in geom_type_to_string}

    # geom ID to size
    geom_id_to_size = env.sim.model._model.geom_size

    # geom ID to body ID
    geom_id_to_body_id = env.sim.model._model.geom_bodyid

    # geom ID to group (1 means visual, does not participate in collisions)
    geom_id_to_group = env.sim.model._model.geom_group

    # geom ID to mesh ID (-1 if no mesh)
    geom_id_to_data_id = env.sim.model._model.geom_dataid

    # material ID for each geom (-1 if no material)
    geom_id_to_mat_id = env.sim.model._model.geom_matid

    # rgba for each geom (only used when material ID is -1)
    geom_id_to_rgba = env.sim.model._model.geom_rgba

    # body name to ID
    body_name_to_id = env.sim.model._body_name2id

    # body ID to name
    body_id_to_name = env.sim.model._body_id2name

    # mesh name to ID
    mesh_name_to_id = env.sim.model._mesh_name2id

    # mesh ID to name
    mesh_id_to_name = env.sim.model._mesh_id2name

    # mesh ID to scale
    mesh_id_to_scale = env.sim.model._model.mesh_scale

    # mesh ID to offsets
    mesh_id_to_pos_offset = env.sim.model._model.mesh_pos
    mesh_id_to_quat_offset = env.sim.model._model.mesh_quat

    # mesh ID to index in asset paths
    mesh_id_to_path_adr = env.sim.model._model.mesh_pathadr

    # get texture and material mappings between ID and name - useful for linking geoms to corresponding materials and textures
    tex_names, tex_name2id, tex_id2name = env.sim.model._extract_mj_names(
        env.sim.model._model.name_texadr,
        env.sim.model._model.ntex,
        mujoco.mjtObj.mjOBJ_TEXTURE,
    )
    mat_names, mat_name2id, mat_id2name = env.sim.model._extract_mj_names(
        env.sim.model._model.name_matadr,
        env.sim.model._model.nmat,
        mujoco.mjtObj.mjOBJ_MATERIAL,
    )

    # Parse XML to get mesh reference frame information
    xml_str = env.sim.model.get_xml()
    xml_root = ET.fromstring(xml_str)

    # Create a mapping from mesh name to reference frame info
    mesh_name_to_ref_info = {}
    for mesh in xml_root.iter("mesh"):
        mesh_name = mesh.get("name")
        assert (
            mesh_name is not None
        ), "Found mesh asset without a name attribute. All meshes must have names."
        ref_pos = mesh.get("refpos")
        ref_quat = mesh.get("refquat")
        ref_info = {
            "refpos": string_to_array(ref_pos) if ref_pos is not None else np.zeros(3),
            "refquat": string_to_array(ref_quat)
            if ref_quat is not None
            else np.array([1, 0, 0, 0]),
        }
        mesh_name_to_ref_info[mesh_name] = ref_info

    # all asset paths (byte string, 0-termination for each string)
    all_asset_paths = env.sim.model._model.paths

    # list of dictionaries, one dict per geom
    geom_model_infos = []

    # body name to list of indices in @geom_model_infos
    body_to_geom_info_inds = dict()

    # all geom names in mujoco
    all_mujoco_geom_names = set([geom_id_to_name[geom_id] for geom_id in range(ngeoms)])
    all_mujoco_geom_names.discard(None)

    # store fake geom names we make for geoms that do not have names
    all_geom_names = set(all_mujoco_geom_names)

    # iterate over all geoms
    geom_model_info_ind = 0
    for geom_id in range(ngeoms):
        geom_name = geom_id_to_name[geom_id]
        geom_group = geom_id_to_group[geom_id]
        geom_type = geom_id_to_type[geom_id]
        geom_size = geom_id_to_size[geom_id]
        mat_id = geom_id_to_mat_id[geom_id]
        body_id = geom_id_to_body_id[geom_id]
        body_name = body_id_to_name[body_id]

        # skip visual geoms (no collision)
        if geom_group == 1:
            if verbose:
                print("skip geom {} since it is visual".format(geom_name))
            continue

        # skip geoms that correspond to robot
        if (geom_name is not None) and any(
            geom_name.startswith(geom_prefix) for geom_prefix in exclude_geom_prefixes
        ):
            if verbose:
                print("skip geom {} since it belongs to robot".format(geom_name))
            continue

        # maybe make a fake geom name
        geom_name_is_fake = False
        if geom_name is None:
            geom_name = get_fake_name(all_geom_names)
            all_geom_names.add(geom_name)
            geom_name_is_fake = True

        # check geom type to understand the geometry
        assert geom_type in geom_type_to_string, "got unknown geom type: {}".format(
            geom_type
        )
        geom_type_name = geom_type_to_string[geom_type]

        # record visual attributes for this geom
        if mat_id != -1:
            geom_material = mat_id2name[mat_id]
            geom_rgba = None
        else:
            geom_material = None
            geom_rgba = geom_id_to_rgba[geom_id]

        # maybe record mesh asset path
        geom_mesh_id = None
        geom_mesh_name = None
        geom_mesh_scale = None
        geom_mesh_pos_offset = None
        geom_mesh_quat_offset = None
        geom_mesh_path = None
        geom_mesh_ref_pos = None
        geom_mesh_ref_quat = None
        if geom_type_name == "mesh":
            # get mesh id and name
            geom_mesh_id = geom_id_to_data_id[geom_id]
            assert geom_mesh_id != -1
            geom_mesh_name = mesh_id_to_name[geom_mesh_id]
            geom_mesh_scale = list(mesh_id_to_scale[geom_mesh_id])
            # assert (len(geom_mesh_scale) == 3) and (geom_mesh_scale[0] == 1.) and (geom_mesh_scale[1] == 1.) and (geom_mesh_scale[2] == 1.)
            geom_mesh_pos_offset = list(mesh_id_to_pos_offset[geom_mesh_id])
            geom_mesh_quat_offset = list(mesh_id_to_quat_offset[geom_mesh_id])

            # read mesh path from asset bytestring
            geom_mesh_path_start_ind = mesh_id_to_path_adr[geom_mesh_id]
            geom_mesh_path = read_byte_string_until_zero(
                byte_string=all_asset_paths, start_index=geom_mesh_path_start_ind
            )

            # Get reference frame info for this mesh
            ref_info = mesh_name_to_ref_info[geom_mesh_name]
            geom_mesh_ref_pos = ref_info["refpos"].tolist()
            geom_mesh_ref_quat = ref_info["refquat"].tolist()

        # remember body this geom corresponds to
        if body_name not in body_to_geom_info_inds:
            body_to_geom_info_inds[body_name] = []
        body_to_geom_info_inds[body_name].append(geom_model_info_ind)

        geom_model_infos.append(
            dict(
                geom_name=geom_name,
                geom_id=int(geom_id),
                body_name=body_name,
                body_id=int(body_id),
                geom_group=int(geom_group),
                geom_type=int(geom_type),
                geom_type_name=geom_type_name,
                geom_size=list(geom_size),
                geom_mesh_id=int(geom_mesh_id) if geom_mesh_id is not None else None,
                geom_mesh_name=geom_mesh_name,
                geom_mesh_scale=geom_mesh_scale,
                geom_mesh_pos_offset=geom_mesh_pos_offset,
                geom_mesh_quat_offset=geom_mesh_quat_offset,
                geom_mesh_path=geom_mesh_path,
                geom_mesh_ref_pos=geom_mesh_ref_pos,
                geom_mesh_ref_quat=geom_mesh_ref_quat,
                geom_material=geom_material,
                geom_rgba=list(geom_rgba),
                geom_name_is_fake=geom_name_is_fake,
            )
        )
        geom_model_info_ind += 1

    return geom_model_infos, body_to_geom_info_inds


def get_geom_pose_info_from_data(env, geom_model_infos, in_base_frame=False):
    """
    Parse mujoco data from robosuite environment to get geom information that should be imported
    into curobo. Returns a list of dictionaries (one per geom) that has the pose of each geom with
    respect to the robot base frame, since this is what curobo expects. Each pose is position (3-dim)
    and rotation (4-dim) expressed as wxyz quaternion.

    Curobo Obstacle size information can be found here:

    https://gitlab-master.nvidia.com/srl/curobo/curobo/-/blob/bala/documentation_705/src/curobo/geom/types.py?ref_type=heads#L279
    """

    # get robot base pose in world frame - needed for conversions
    assert len(env.robots) == 1, "only one robot supported for now"
    if in_base_frame:
        # base_pos, base_rot = env.robots[
        #     0
        # ].composite_controller.get_controller_base_pose("right")
        base_pos, base_rot = env.sim.data.get_body_xpos(
            "robot0_base"
        ), env.sim.data.get_body_xmat("robot0_base")
        base_pose_inv = T.pose_inv(T.make_pose(base_pos, base_rot))
    else:
        base_pose_inv = np.eye(4)

    geom_pose_infos = []
    for geom_info in geom_model_infos:
        # read attributes we need
        geom_name = geom_info["geom_name"]
        geom_id = geom_info["geom_id"]
        geom_name_is_fake = geom_info["geom_name_is_fake"]

        # get geom pose in world frame
        geom_pos = np.array(env.sim.data.geom_xpos[geom_id])
        geom_rot = np.array(env.sim.data.geom_xmat[geom_id].reshape((3, 3)))
        geom_pose = T.make_pose(geom_pos, geom_rot)

        # if geom_info["geom_mesh_pos_offset"] is not None:
        if geom_info["geom_type_name"] == "mesh":
            # correct the pose of the mesh
            # see: https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-mesh

            # First apply the mesh offset transformation if it exists
            if geom_info["geom_mesh_pos_offset"] is not None:
                geom_offset = T.make_pose(
                    geom_info["geom_mesh_pos_offset"],
                    T.quat2mat(
                        T.convert_quat(
                            np.array(geom_info["geom_mesh_quat_offset"]), to="xyzw"
                        )
                    ),
                )
                geom_pose = np.matmul(geom_pose, T.pose_inv(geom_offset))

            # Then apply the reference frame transformation
            ref_pos = geom_info["geom_mesh_ref_pos"]
            ref_quat = geom_info["geom_mesh_ref_quat"]
            ref_pose = T.make_pose(
                ref_pos, T.quat2mat(T.convert_quat(np.array(ref_quat), to="xyzw"))
            )
            geom_pose = np.matmul(geom_pose, T.pose_inv(ref_pose))

        # convert to robot base frame
        geom_pose_in_base = convert_world_pose_to_base_frame(
            pose=geom_pose, base_pose_inv=base_pose_inv
        )
        geom_pos, geom_rot = (
            geom_pose_in_base[..., :3, 3],
            geom_pose_in_base[..., :3, :3],
        )
        geom_quat = T.convert_quat(T.mat2quat(geom_rot), to="wxyz")
        final_geom_pose = np.concatenate([geom_pos, geom_quat])

        geom_pose_infos.append(
            dict(
                geom_name=geom_name,
                geom_id=geom_id,
                geom_pose=final_geom_pose.tolist(),
                # geom_pose=geom_pose,
                geom_name_is_fake=geom_name_is_fake,
            )
        )

    return geom_pose_infos


def get_object_frame_points(mesh_object, num_points=1000):
    """
    Get points on a mesh in the object's local frame.

    Args:
        mesh_object (trimesh.Trimesh): The mesh object to sample points from
        num_points (int): Number of points to sample on the surface

    Returns:
        np.ndarray: Array of shape (num_points, 3) containing points in object frame
    """
    # Sample points on the surface of the mesh
    points = mesh_object.sample(num_points)

    # The points are already in the object's local frame since we're using the raw mesh
    # without any transformations applied
    return points


def create_trimesh_scene_from_mujoco(env, geom_model_infos, geom_pose_infos):
    """
    Creates trimesh scene from mujoco world. Takes outputs of
    @get_geom_info_from_model and @get_geom_pose_info_from_data, which parse
    the current mujoco scene into a format that is easily intepreted by this function.
    """
    assert len(geom_model_infos) == len(geom_pose_infos)
    num_geoms = len(geom_model_infos)

    # create an entity per geom
    mesh_obstacles = []
    cuboid_obstacles = []
    capsule_obstacles = []
    cylinder_obstacles = []
    sphere_obstacles = []

    # housekeeping for convenience
    obstacles_by_name = dict()

    colors = [
        colorsys.hsv_to_rgb(h, s=1.0, v=0.8)
        for h in np.linspace(0, 1, num_geoms, endpoint=False)
    ]
    for ind in range(num_geoms):
        geom_model_info = geom_model_infos[ind]
        geom_pose_info = geom_pose_infos[ind]

        geom_name = geom_model_info["geom_name"]
        assert geom_pose_info["geom_name"] == geom_name

        geom_type_name = geom_model_info["geom_type_name"]
        geom_size = geom_model_info["geom_size"]
        geom_pose = geom_pose_info["geom_pose"]
        geom_pose = T.make_pose(
            geom_pose[:3],
            T.quat2mat(T.convert_quat(np.array(geom_pose[3:]), to="xyzw")),
        )
        geom_color = np.append(colors[ind], 1.0)

        if geom_type_name == "box":
            obstacle = trimesh.creation.box(
                extents=[2.0 * geom_size[0], 2.0 * geom_size[1], 2.0 * geom_size[2]],
            )
            obstacle.apply_transform(geom_pose)
            obstacle.visual.face_colors = geom_color[:3]
            cuboid_obstacles.append(obstacle)
        elif geom_type_name == "mesh":
            mesh_file = geom_model_info["geom_mesh_path"]
            mesh_scale = geom_model_info["geom_mesh_scale"]
            obstacle = trimesh.load_mesh(mesh_file)
            obstacle.apply_scale(mesh_scale)
            obstacle.apply_transform(geom_pose)
            # obstacle.visual.face_colors = geom_color[:3]
            mesh_obstacles.append(obstacle)
        elif geom_type_name == "capsule":
            obstacle = trimesh.creation.capsule(
                radius=geom_size[0],
                height=2.0 * geom_size[1],
            )
            obstacle.apply_transform(geom_pose)
            obstacle.visual.face_colors = geom_color[:3]
            capsule_obstacles.append(obstacle)
        elif geom_type_name == "cylinder":
            obstacle = trimesh.creation.cylinder(
                radius=geom_size[0],
                height=(2.0 * geom_size[1]),
            )
            obstacle.apply_transform(geom_pose)
            obstacle.visual.face_colors = geom_color[:3]
            cylinder_obstacles.append(obstacle)
        elif geom_type_name == "sphere":
            obstacle = trimesh.creation.icosphere(
                radius=geom_size[0],
            )
            obstacle.apply_transform(geom_pose)
            obstacle.visual.face_colors = geom_color[:3]
            sphere_obstacles.append(obstacle)
        else:
            raise Exception("got invalid geom type: {}".format(geom_type_name))

        # store the obstacle object by geom name
        obstacles_by_name[geom_name] = obstacle

    # construct trimesh scene from the obstacles
    all_meshes = (
        mesh_obstacles
        + cuboid_obstacles
        + capsule_obstacles
        + cylinder_obstacles
        + sphere_obstacles
    )
    scene = trimesh.Scene(all_meshes)

    return scene
