# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Client script for FoundationStereo ZMQ client that communicates with the server on the same host.
The script uses shared memory to minimize data transfer latency.
"""

import cv2
import zmq
import time
import numpy as np
from multiprocessing import shared_memory


class FoundationStereoZMQClient:
    def __init__(self, server_address="localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{server_address}")

    def get_msg_for_shm(self, left_image, right_image, intrinsic_matrix):
        concat_image = np.concatenate([left_image, right_image], axis=1)

        # Create shared memory for image
        shm = shared_memory.SharedMemory(create=True, size=concat_image.nbytes)
        shm_np = np.ndarray(
            concat_image.shape, dtype=concat_image.dtype, buffer=shm.buf
        )
        np.copyto(shm_np, concat_image)

        # Create shared memory for intrinsic matrix
        intrinsic_matrix_shm = shared_memory.SharedMemory(
            create=True, size=intrinsic_matrix.nbytes
        )
        intrinsic_matrix_shm_np = np.ndarray(
            intrinsic_matrix.shape,
            dtype=intrinsic_matrix.dtype,
            buffer=intrinsic_matrix_shm.buf,
        )
        np.copyto(intrinsic_matrix_shm_np, intrinsic_matrix)

        # Prepare depth map shared memory buffer (grayscale output)
        depth_shape = (
            concat_image.shape[0],
            concat_image.shape[1] // 2,
        )  # depth for left image
        depth_dtype = np.dtype(np.float32)
        depth_shm = shared_memory.SharedMemory(
            create=True, size=np.prod(depth_shape) * np.dtype(depth_dtype).itemsize
        )

        # Send metadata & shared memory names to server
        msg = {
            "shm_name": shm.name,
            "shape": concat_image.shape,
            "dtype": concat_image.dtype.name,
            "intrinsic_matrix_shm_name": intrinsic_matrix_shm.name,
            "intrinsic_matrix_shape": intrinsic_matrix.shape,
            "intrinsic_matrix_dtype": intrinsic_matrix.dtype.name,
            "depth_shm_name": depth_shm.name,
            "depth_shape": depth_shape,
            "depth_dtype": depth_dtype.name,
        }

        return msg, shm, (depth_shm, depth_shape, depth_dtype)

    def get_depth_map(self, left_image, right_image, intrinsic_matrix):
        msg, shm, (depth_shm, depth_shape, depth_dtype) = self.get_msg_for_shm(
            left_image, right_image, intrinsic_matrix
        )

        # Send message to server
        self.socket.send_json(msg)

        # Wait for server to respond when done
        self.socket.recv_string()

        # Read depth map result from shared memory
        depth_np = np.ndarray(depth_shape, dtype=depth_dtype, buffer=depth_shm.buf)
        depth_map = depth_np.copy()  # copy out to avoid shared memory dependencies

        # Cleanup shared memory
        shm.close()
        shm.unlink()
        depth_shm.close()
        depth_shm.unlink()

        return depth_map


if __name__ == "__main__":
    left_path = "/home/siddhanth/github/FoundationStereo/assets/left.png"
    right_path = "/home/siddhanth/github/FoundationStereo/assets/right.png"

    # read images
    left_image = cv2.imread(left_path)[:, :, ::-1]  # BGR to RGB
    right_image = cv2.imread(right_path)[:, :, ::-1]  # BGR to RGB

    # resize
    scale = 0.5
    left_image = cv2.resize(
        left_image, (int(left_image.shape[1] * scale), int(left_image.shape[0] * scale))
    )
    right_image = cv2.resize(
        right_image,
        (int(right_image.shape[1] * scale), int(right_image.shape[0] * scale)),
    )

    intrinsic_matrix = np.eye(3)

    client = FoundationStereoZMQClient(server_address="localhost:60000")
    while True:
        print("Sending prompt...")
        start_time = time.time()
        depth_map = client.get_depth_map(left_image, right_image, intrinsic_matrix)
        print(f"Time taken: {time.time() - start_time} seconds")
        print(depth_map.shape)
