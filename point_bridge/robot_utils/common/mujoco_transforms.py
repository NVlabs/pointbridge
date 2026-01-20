# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np


class MujocoTransforms:
    def __init__(
        self, env, camera_names, height, width=None, gripper_name="gripper0_eef"
    ):
        self._env = env
        self._camera_names = camera_names
        self._height = height
        self._width = width if width is not None else height
        self._gripper_name = gripper_name

    def create_projection_matrix(self, translation, rotation):
        pose = np.zeros((4, 4))
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        pose[3, 3] = 1.0

        # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
        camera_axis_correction = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        pose = pose @ camera_axis_correction
        return pose

    def get_camera_projection_matrix(self, camera_name):
        """
        Returns camera pose. This is not extrinsics
        """
        pos = self._env.sim.data.cam_xpos[
            self._env.sim.model.camera_name2id(camera_name)
        ]
        rot = self._env.sim.data.cam_xmat[
            self._env.sim.model.camera_name2id(camera_name)
        ].reshape((3, 3))

        projection_matrix = self.create_projection_matrix(pos, rot)
        return projection_matrix

    def get_robot_base_projection(self):
        pos = self._env.sim.data.get_body_xpos("robot0_base")
        rot = self._env.sim.data.get_body_xmat("robot0_base").reshape((3, 3))
        projection_matrix = self.create_projection_matrix(pos, rot)
        return projection_matrix

    def get_robot_gripper_projection(self, gripper_name="gripper0_eef"):
        pos = self._env.sim.data.get_body_xpos(gripper_name)
        rot = self._env.sim.data.get_body_xmat(gripper_name).reshape((3, 3))
        projection_matrix = self.create_projection_matrix(pos, rot)
        return projection_matrix

    def get_camera_intrinsics(self, camera_name):
        """
        Link: https://github.com/openai/mujoco-py/issues/271
        """
        fovy = self._env.sim.model.cam_fovy[
            self._env.sim.model.camera_name2id(camera_name)
        ]
        fovy_radians = np.radians(fovy)

        # Calculate aspect ratio
        aspect_ratio = self._width / self._height

        # For rectangular images, we need to calculate focal lengths differently
        # The vertical FOV (fovy) is given, so we calculate fy first
        fy = 0.5 * self._height / np.tan(fovy_radians / 2)

        # For the horizontal focal length, we need to consider the aspect ratio
        # The horizontal FOV (fovx) is related to the vertical FOV by: tan(fovx/2) = aspect_ratio * tan(fovy/2)
        fovx_radians = 2 * np.arctan(aspect_ratio * np.tan(fovy_radians / 2))
        fx = 0.5 * self._width / np.tan(fovx_radians / 2)

        camera_intrinsics = np.array(
            [[fx, 0, self._width / 2], [0, fy, self._height / 2], [0, 0, 1]]
        )
        return camera_intrinsics

    @property
    def camera_projection_matrix(self):
        projection_matrix = {}
        for cam in self._camera_names:
            projection_matrix[cam] = self.get_camera_projection_matrix(cam)
        return projection_matrix

    @property
    def camera_intrinsics(self):
        camera_matrix = {}
        for cam in self._camera_names:
            camera_matrix[cam] = self.get_camera_intrinsics(cam)
        return camera_matrix

    @property
    def transforms(self):
        # get camera projection matrix
        camera_projection_matrix = self.camera_projection_matrix
        # get robot base projection matrix
        robot_base_projection_matrix = self.get_robot_base_projection()
        # get robot gripper projection matrix
        robot_gripper_projection_matrix = self.get_robot_gripper_projection(
            self._gripper_name
        )
        return {
            "camera_projection_matrix": camera_projection_matrix,  # camera to world
            "robot_base_projection_matrix": robot_base_projection_matrix,
            "robot_gripper_projection_matrix": robot_gripper_projection_matrix,
        }

    def transformation_matrix_from_projection_matrix(self, dict):
        camera_projection_matrix = dict["camera_projection_matrix"]
        robot_base_projection_matrix = dict["robot_base_projection_matrix"]
        robot_gripper_projection_matrix = dict["robot_gripper_projection_matrix"]

        # camera to robot base
        camera_to_robot_base = {}
        for cam_name in camera_projection_matrix.keys():
            camera_to_robot_base[cam_name] = (
                np.linalg.inv(camera_projection_matrix[cam_name])
                @ robot_base_projection_matrix
            )

        # camera to robot gripper
        camera_to_robot_gripper = {}
        for cam_name in camera_projection_matrix.keys():
            camera_to_robot_gripper[cam_name] = (
                np.linalg.inv(camera_projection_matrix[cam_name])
                @ robot_gripper_projection_matrix
            )

        return {
            "camera2robot_base": camera_to_robot_base,  # robot to camera
            "camera2robot_gripper": camera_to_robot_gripper,  # robot gripper to camera
        }
