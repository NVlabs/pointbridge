# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Efficient robot environment server for real-time control over 10Gigabit ethernet.
Uses ZeroMQ REQ/REP pattern with MessagePack serialization and JPEG compression.
"""

import zmq
import msgpack
import msgpack_numpy as m
import numpy as np
import cv2
import io
import time
import gym
import franka_env
from typing import Dict, Any, Tuple

# Enable MessagePack numpy support
m.patch()


class RobotZMQServer:
    def __init__(
        self,
        host="*",
        port=5555,
        jpeg_quality=85,
        env_name="Franka-v1",
        cam_ids=[2, 4, 5, 6],
        zed_ids=[8],
        height=640,
        width=480,
        use_robot=False,
        use_gt_depth=False,
        side="left",
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self.port = port
        self.jpeg_quality = jpeg_quality

        # Set socket options for better performance
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second timeout

        # Initialize robot environment
        self.env = gym.make(
            env_name,
            cam_ids=cam_ids,
            zed_ids=zed_ids,
            height=height,
            width=width,
            use_robot=use_robot,  # True,  # Fixed: was eval
            use_gt_depth=use_gt_depth,  # True,
            side=side,
        )
        self.env.reset()
        # self.env = None

        # Pre-allocate buffers for efficiency
        self._jpeg_encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        self._jpeg_decode_params = [cv2.IMREAD_COLOR]

    def _compress_image(self, image: np.ndarray) -> bytes:
        """Compress image using JPEG with specified quality."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR to RGB if needed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Encode as JPEG
        success, encoded_img = cv2.imencode(".jpg", image, self._jpeg_encode_params)
        if not success:
            raise ValueError("Failed to encode image as JPEG")

        return encoded_img.tobytes()

    def _decompress_image(self, image_data: bytes) -> np.ndarray:
        """Decompress JPEG image data."""
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image data")
        return image

    def _serialize_observation(self, obs: Dict[str, Any]) -> bytes:
        """Serialize observation dict with efficient compression for images."""
        serialized_obs = {}

        for key, value in obs.items():
            if "pixels" in key and isinstance(value, np.ndarray):
                # Compress image data
                serialized_obs[key] = {
                    "type": "image",
                    "data": self._compress_image(value),
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                }
            elif isinstance(value, np.ndarray):
                # Direct numpy array serialization
                serialized_obs[key] = {
                    "type": "array",
                    "data": value,
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                }
            else:
                # Scalar or other types
                serialized_obs[key] = {"type": "scalar", "data": value}

        return msgpack.packb(serialized_obs, use_bin_type=True)

    def _deserialize_action(self, action_data: bytes) -> np.ndarray:
        """Deserialize action from client."""
        data = msgpack.unpackb(action_data, raw=False)
        if data["type"] != "array":
            raise ValueError("Expected array type for action")
        return data["data"]

    def process_action(self, action: np.ndarray) -> bytes:
        """Process robot action and return serialized observation."""
        try:
            # Step the environment

            observation, reward, done, _, info = self.env.step(action)

            # Serialize and return observation
            return self._serialize_observation(observation)

        except Exception as e:
            # Return error information
            error_obs = {"error": str(e), "done": True}
            return self._serialize_observation(error_obs)

    def start(self):
        """Start the server main loop."""
        print(f"Robot ZMQ Server listening on port {self.port}")
        print(f"JPEG quality: {self.jpeg_quality}")

        while True:
            try:
                # Receive action from client
                action_data = self.socket.recv()

                if action_data == b"reset":
                    observation_data = self.env.reset()
                    observation_data = self._serialize_observation(observation_data)
                else:
                    action = self._deserialize_action(action_data)
                    observation_data = self.process_action(action)

                # Send observation back
                self.socket.send(observation_data)

            except zmq.Again:
                print("Timeout waiting for client request")
                continue
            except Exception as e:
                print(f"Error processing request: {e}")
                # Send error response
                error_obs = {"error": str(e), "done": True}
                error_data = self._serialize_observation(error_obs)
                try:
                    self.socket.send(error_data)
                except:
                    pass
                continue

    def close(self):
        """Close server resources."""
        self.socket.close()
        self.context.term()
        self.env.close()


if __name__ == "__main__":
    use_robot = True
    side = "left"
    cam_ids = [8]
    zed_ids = [8]
    width, height = 672, 448
    use_gt_depth = False

    server = RobotZMQServer(
        port=7999,
        jpeg_quality=85,
        env_name="Franka-v1",
        cam_ids=cam_ids,
        zed_ids=zed_ids,
        height=height,
        width=width,
        use_robot=use_robot,
        use_gt_depth=use_gt_depth,
        side=side,
    )

    try:
        server.start()
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.close()
