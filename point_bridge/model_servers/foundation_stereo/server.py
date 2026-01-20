# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import sys
import zmq
import numpy as np
from multiprocessing import shared_memory

import cv2
import torch
import onnxruntime as ort
from onnx_tensorrt import tensorrt_engine
import tensorrt as trt


def get_onnx_model(pretrained_model_path):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = ort.InferenceSession(
        pretrained_model_path,
        sess_options=session_options,
        providers=["CUDAExecutionProvider"],
    )
    return model


def get_engine_model(pretrained_model_path):
    with open(pretrained_model_path, "rb") as file:
        engine_data = file.read()
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
        engine_data
    )
    engine = tensorrt_engine.Engine(engine)
    return engine


def initialize_foundation_stereo_trt():
    """Initialize the FoundationStereo TensorRT model with pretrained weights."""

    # FoundationStereo imports
    FS_DIR = "/home_shared/grail_siddhant/github/FoundationStereo"
    sys.path.append(FS_DIR)

    pretrained_model_path = f"{FS_DIR}/pretrained_models/foundation_stereo.plan"

    # load model
    if pretrained_model_path.endswith(".onnx"):
        model = get_onnx_model(pretrained_model_path)
    elif pretrained_model_path.endswith(".engine") or pretrained_model_path.endswith(
        ".plan"
    ):
        model = get_engine_model(pretrained_model_path)
    else:
        raise ValueError(f"Unsupported model file extension: {pretrained_model_path}")

    args = dict(
        height=448,
        width=672,
        baseline=0.12,
        pretrained_model_path=pretrained_model_path,
        remove_invisible=True,
    )

    return model, args


def depth_from_foundation_stereo_trt(
    model, left_img, right_img, intrinsic_matrix, args
):
    """
    Get depth from FoundationStereo model in the img0 frame.
    """
    H, W = left_img.shape[:2]
    if H != args["height"] or W != args["width"]:
        left_img = cv2.resize(
            left_img, (args["width"], args["height"]), interpolation=cv2.INTER_LINEAR
        )
        right_img = cv2.resize(
            right_img, (args["width"], args["height"]), interpolation=cv2.INTER_LINEAR
        )

    # convert to torch tensor
    left_img = (
        torch.as_tensor(left_img.copy()).float()[None].permute(0, 3, 1, 2).contiguous()
    )
    right_img = (
        torch.as_tensor(right_img.copy()).float()[None].permute(0, 3, 1, 2).contiguous()
    )

    # run model
    torch.cuda.synchronize()
    if args["pretrained_model_path"].endswith(".onnx"):
        left_disp = model.run(
            None, {"img0": left_img.numpy(), "img1": right_img.numpy()}
        )[0]
    else:
        left_disp = model.run([left_img.numpy(), right_img.numpy()])[0]
    torch.cuda.synchronize()

    # left_disp = left_disp.squeeze().cpu().numpy() # HxW
    left_disp = left_disp.reshape(args["height"], args["width"])

    if args["remove_invisible"]:
        yy, xx = np.meshgrid(
            np.arange(left_disp.shape[0]), np.arange(left_disp.shape[1]), indexing="ij"
        )
        us_right = xx - left_disp
        invalid = us_right < 0
        left_disp[invalid] = np.inf

    depth = intrinsic_matrix[0, 0] * args["baseline"] / left_disp

    if H != args["height"] or W != args["width"]:
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

    return depth


class FoundationStereoZMQServer:
    def __init__(self, host="*", port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self.port = port

        # Initialize FoundationStereo components
        self.model, self.args = initialize_foundation_stereo_trt()

    def process_request(self, msg_parts):
        shm_name = msg_parts["shm_name"]
        shape = tuple(msg_parts["shape"])
        dtype = np.dtype(msg_parts["dtype"])
        intrinsic_matrix_shm_name = msg_parts["intrinsic_matrix_shm_name"]
        intrinsic_matrix_shape = tuple(msg_parts["intrinsic_matrix_shape"])
        intrinsic_matrix_dtype = np.dtype(msg_parts["intrinsic_matrix_dtype"])
        depth_shm_name = msg_parts["depth_shm_name"]
        depth_shm_shape = tuple(msg_parts["depth_shape"])
        depth_shm_dtype = np.dtype(msg_parts["depth_dtype"])

        # # Access shared memory buffer for input image
        shm = shared_memory.SharedMemory(name=shm_name)
        concat_img = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        left_img = concat_img[:, : concat_img.shape[1] // 2]
        right_img = concat_img[:, concat_img.shape[1] // 2 :]

        intrinsic_matrix_shm = shared_memory.SharedMemory(
            name=intrinsic_matrix_shm_name
        )
        intrinsic_matrix_shm_np = np.ndarray(
            intrinsic_matrix_shape,
            dtype=intrinsic_matrix_dtype,
            buffer=intrinsic_matrix_shm.buf,
        )
        intrinsic_matrix = intrinsic_matrix_shm_np.copy()

        # Process depth map
        depth_map = depth_from_foundation_stereo_trt(
            self.model,
            left_img,
            right_img,
            intrinsic_matrix,
            self.args,
        )

        # Write depth map to shared memory
        depth_shm = shared_memory.SharedMemory(name=depth_shm_name)
        depth_np = np.ndarray(
            depth_shm_shape, dtype=depth_shm_dtype, buffer=depth_shm.buf
        )
        np.copyto(depth_np, depth_map)

        # Cleanup input shm reference
        shm.close()
        depth_shm.close()

        # Notify client that we're done
        self.socket.send_string("done")

    def start(self):
        print(f"ZMQ FoundationStereo Server listening on port {self.port}")
        while True:
            # Receive shared memory names and shape info
            msg_parts = self.socket.recv_json()

            # Process and respond
            self.process_request(msg_parts)


if __name__ == "__main__":
    server = FoundationStereoZMQServer(port=60000)
    server.start()
