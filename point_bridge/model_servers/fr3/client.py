# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Efficient robot environment client for real-time control over 10Gigabit ethernet.
Uses ZeroMQ REQ/REP pattern with MessagePack serialization and JPEG decompression.
"""

import zmq
import msgpack
import msgpack_numpy as m
import numpy as np
import cv2
import time
from typing import Dict, Any, Optional

# Enable MessagePack numpy support
m.patch()


class RobotZMQClient:
    def __init__(self, server_address="localhost:5555", timeout_ms=1000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{server_address}")

        # Set socket options for better performance
        # No need to set timeout for client side because the robot could take longer to execute the action.
        self.socket.setsockopt(zmq.LINGER, 0)
        # self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        # self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

        # Pre-allocate decode parameters
        self._jpeg_decode_params = [cv2.IMREAD_COLOR]

    def _serialize_action(self, action: np.ndarray) -> bytes:
        """Serialize action for sending to server."""
        action_data = {
            "type": "array",
            "data": action,
            "shape": action.shape,
            "dtype": str(action.dtype),
        }
        return msgpack.packb(action_data, use_bin_type=True)

    def _decompress_image(self, image_data: bytes) -> np.ndarray:
        """Decompress JPEG image data."""
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image data")
        return image

    def _deserialize_observation(self, obs_data: bytes) -> Dict[str, Any]:
        """Deserialize observation from server."""
        serialized_obs = msgpack.unpackb(obs_data, raw=False)
        observation = {}

        for key, value in serialized_obs.items():
            if value["type"] == "image":
                # Decompress image data
                image = self._decompress_image(value["data"])
                observation[key] = image
            elif value["type"] == "array":
                # Direct numpy array
                observation[key] = value["data"]
            else:
                # Scalar or other types
                observation[key] = value["data"]

        return observation

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Send action to robot server and receive observation.

        Args:
            action: 7-dimensional numpy array representing robot action

        Returns:
            Dictionary containing observation data with 'pixels' keys as images
            and other keys as numpy arrays
        """
        try:
            # Serialize action
            action_data = self._serialize_action(action)

            # Send action to server
            self.socket.send(action_data)

            # Receive observation from server
            obs_data = self.socket.recv()

            # Deserialize observation
            observation = self._deserialize_observation(obs_data)

            return observation

        except zmq.Again:
            raise TimeoutError("Request timed out - server may be unavailable")
        except Exception as e:
            raise RuntimeError(f"Communication error: {e}")

    def reset(self) -> Dict[str, Any]:
        """
        Reset the robot environment.
        Note: This requires server support for reset functionality.
        """
        # Send the text "reset" to the server
        self.socket.send(b"reset")
        obs_data = self.socket.recv()
        observation = self._deserialize_observation(obs_data)
        return observation

    def close(self):
        """Close client resources."""
        self.socket.close()
        self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import numpy as np

    # Test client
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    with RobotZMQClient(server_address="100.94.192.118:7999") as client:
        print("Testing robot client...")

        print("Resetting environment...")
        obs = client.reset()
        print(f"Reset observation keys: {list(obs.keys())}")

        # Test a few steps
        times = []
        num_steps = 5  # 100
        for i in range(num_steps):
            print(f"Step {i+1}...")
            start_time = time.time()

            # Send random action
            action = np.random.uniform(-0.1, 0.1, 7)
            obs = client.step(action)

            end_time = time.time()
            print(f"  Response time: {(end_time - start_time)*1000:.2f}ms")
            print(f"  Observation keys: {list(obs.keys())}")
            times.append(end_time - start_time)

            # Check for images
            for key, value in obs.items():
                if "pixels" in key and isinstance(value, np.ndarray):
                    print(f"  {key} shape: {value.shape}")

            time.sleep(0.1)  # Small delay between steps

        print(f"Average step time: {np.mean(times)*1000:.2f}ms")
        print(f"Max step time: {np.max(times)*1000:.2f}ms")
        print(f"Min step time: {np.min(times)*1000:.2f}ms")
        print(f"Std step time: {np.std(times)*1000:.2f}ms")
        print(f"Max frequency: {1000/np.min(times):.2f}Hz")
        print(f"Min frequency: {1000/np.max(times):.2f}Hz")
        print(f"Std frequency: {1000/np.std(times):.2f}Hz")
