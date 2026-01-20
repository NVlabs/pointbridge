# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import sys
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class PointTracker:
    def __init__(
        self,
        pixel_keys,
        device,
        **kwargs,
    ):
        """
        Initialize the Point Tracking Class for tracking key points using CoTracker.

        Parameters:
        -----------
        pixel_keys : list
            List of pixel keys to track points for.

        device : str
            The device to use for computation, either 'cpu' or 'cuda' (for GPU acceleration).

        cotracker_checkpoint : str
            Path to the CoTracker checkpoint file.

        """

        self.pixel_keys = pixel_keys
        self.device = device

        # Initialize CoTracker models for each pixel key
        self.cotracker = {}
        for pixel_key in self.pixel_keys:
            # Import CoTracker here to avoid circular imports
            cotracker_path = os.path.join(
                os.path.dirname(__file__), "../../third_party/co-tracker/"
            )
            sys.path.append(cotracker_path)

            from cotracker.predictor import CoTrackerOnlinePredictor

            self.cotracker[pixel_key] = CoTrackerOnlinePredictor(
                checkpoint=f"{cotracker_path}/checkpoints/scaled_online.pth",
                window_len=16,
            ).to(device)

        # Initialize tracking dictionaries
        self.tracks = {pixel_key: None for pixel_key in self.pixel_keys}
        self.object_tracks = {pixel_key: None for pixel_key in self.pixel_keys}

        # Initialize image lists
        self.image_list = {
            f"{pixel_key}": torch.tensor([]).to(self.device)
            for pixel_key in self.pixel_keys
        }

        # in case image is cropped and resized
        self.original_image_size = None
        self.current_image_size = None
        self.crop_ratios = None

    def add_to_image_list(self, image, pixel_key):
        """
        Add an image to the image list for finding key points.

        Parameters:
        -----------
        image : np.ndarray
            The image to add to the image list. This image must be in RGB format.
        """

        key = f"{pixel_key}"

        transformed = (
            torch.from_numpy(image.astype(np.uint8)).permute(2, 0, 1).float() / 255
        )

        # We only want to track the last 16 images so pop the first one off if we have more than 16
        if self.image_list[key].shape[0] > 0 and self.image_list[key].shape[1] == 16:
            self.image_list[key] = self.image_list[key][:, 1:]

        # If it is the first image you want to repeat until the whole array is full
        # Otherwise it will just add the new image to the end of the array
        while self.image_list[key].shape[0] == 0 or self.image_list[key].shape[1] < 16:
            self.image_list[key] = torch.cat(
                (
                    self.image_list[key],
                    transformed.unsqueeze(0).unsqueeze(0).clone().to(self.device),
                ),
                dim=1,
            )

    def reset_episode(self):
        """
        Reset the image list and tracking data for finding key points.
        """

        self.image_list = {
            f"{pixel_key}": torch.tensor([]).to(self.device)
            for pixel_key in self.pixel_keys
        }
        self.tracks = {pixel_key: None for pixel_key in self.pixel_keys}
        self.object_tracks = {pixel_key: None for pixel_key in self.pixel_keys}

    def track_points(
        self,
        pixel_key,
        last_n_frames=1,
        is_first_step=False,
        one_frame=True,
        init_points=None,
    ):
        """
        Track the key points in the current image using the CoTracker model.

        Parameters:
        -----------
        pixel_key : str
            The pixel key to track points for.

        last_n_frames : int
            Number of frames to track back.

        is_first_step : bool
            Whether or not this is the first step in the episode.

        one_frame : bool
            Whether to track only one frame.

        init_points : torch.Tensor, optional
            Initial points to track if provided.
        """

        # Track object points using CoTracker
        # if init_points is not None:
        if is_first_step and init_points is not None:
            # Initialize CoTracker with provided points
            init_points_tensor = torch.tensor(init_points).float().to(self.device)

            # Set frame index for CoTracker
            init_points_tensor[:, 0] = self.cotracker[pixel_key].model.window_len - 2

            self.cotracker[pixel_key](
                video_chunk=self.image_list[pixel_key][0, 0].unsqueeze(0).unsqueeze(0),
                is_first_step=True,
                add_support_grid=True,
                queries=init_points_tensor[None].to(self.device),
            )
            self.object_tracks[pixel_key] = init_points_tensor
            self.num_tracked_points = init_points.shape[0]
        else:
            # Continue tracking with CoTracker
            tracks, _ = self.cotracker[pixel_key](
                self.image_list[pixel_key], one_frame=one_frame
            )
            # Remove the support points - keep only the tracked points
            tracks = tracks[:, :, 0 : self.num_tracked_points, :]
            self.object_tracks[pixel_key] = tracks.clone()

        self.tracks[pixel_key] = self.object_tracks[pixel_key]

    def get_points(self, pixel_key, last_n_frames=1):
        """
        Get the tracked points for the current frame organized in a dictionary.

        Parameters:
        -----------
        pixel_key : str
            The pixel key to get points for.

        last_n_frames : int
            The number of frames to look back in the episode.

        Returns:
        --------
        points_dict : dict
            Dictionary containing 'object' point arrays.
        """
        points_dict = {}

        if self.tracks[pixel_key] is None:
            return points_dict

        # Get object points if available
        if self.object_tracks[pixel_key] is not None:
            num_object_points = self.object_tracks[pixel_key].shape[2]
            object_points = torch.zeros((last_n_frames, num_object_points, 2))
            for frame_num in range(last_n_frames):
                frame_idx = -1 * (last_n_frames - frame_num)
                for point in range(num_object_points):
                    x = self.object_tracks[pixel_key][0, frame_idx, point][0]
                    y = self.object_tracks[pixel_key][0, frame_idx, point][1]
                    object_points[frame_num, point] = torch.tensor([x, y])
            points_dict["object"] = object_points

        return points_dict

    def plot_image(self, pixel_key, last_n_frames=1):
        """
        Plot the image with the key points overlaid on top of it. Running this will slow down your tracking, but it's good for debugging.

        Parameters:
        -----------
        pixel_key : str
            The pixel key to plot points for.

        last_n_frames : int
            The number of frames to look back in the episode.

        Returns:
        --------
        img_list : list
            A list of images with the key points overlaid on top of them.
        """

        img_list = []

        for frame_num in range(last_n_frames):
            frame_idx = -1 * (last_n_frames - frame_num)
            curr_image = (
                self.image_list[pixel_key][0, frame_idx]
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                * 255
            )

            fig, ax = plt.subplots(1)
            ax.imshow(curr_image.astype(np.uint8))

            if self.tracks[pixel_key] is not None:
                rainbow = plt.get_cmap("rainbow")
                # Generate n evenly spaced colors from the colormap
                colors = [
                    rainbow(i / self.tracks[pixel_key].shape[2])
                    for i in range(self.tracks[pixel_key].shape[2])
                ]

                for idx, coord in enumerate(self.tracks[pixel_key][0, frame_idx]):
                    ax.add_patch(
                        patches.Circle(
                            (coord[0].cpu(), coord[1].cpu()),
                            5,
                            facecolor=colors[idx],
                            edgecolor="black",
                        )
                    )
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img_list.append(img.copy())
            plt.close()

        return img_list
