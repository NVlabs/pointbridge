# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import zmq


class MolmoZMQClient:
    def __init__(self, server_address="localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{server_address}")

    def send_prompt(self, text_prompt, image_path):
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Send multipart message [text, image]
        self.socket.send_multipart([text_prompt.encode(), image_data])
        return self.socket.recv_string()


if __name__ == "__main__":
    prompt = "Mark a point on the bowl. Only look at the region in front of the robot."
    image_path = "image.png"

    client = MolmoZMQClient(server_address="localhost:45000")
    while True:
        print("Sending prompt...")
        response = client.send_prompt(prompt, image_path)
        print("VLM Response:", response)
