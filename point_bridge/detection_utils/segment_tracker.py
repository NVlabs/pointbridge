# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
import yaml
import cv2
import torch
import numpy as np
from PIL import Image


def get_root_dir(cfg_path):
    with open(cfg_path, "r") as f:
        local_cfg = yaml.safe_load(f)
    return local_cfg["root_dir"]
PB_DIR = get_root_dir(os.path.join(os.path.dirname(__file__), "../cfgs/local.yaml"))


class SegmentTracker:
    def __init__(
        self,
        device: str = "cuda",
        checkpoint: str = f"{PB_DIR}/third_party/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        **kwargs,
    ):
        """
        Initialize the Segment Tracking Class for tracking objects using SAM-2 camera predictor.

        Parameters:
        -----------
        device : str
            The device to use for computation, either 'cpu' or 'cuda' (for GPU acceleration).
        checkpoint : str
            Path to the SAM-2 checkpoint file.
        model_cfg : str
            Path to the SAM-2 model configuration file.
        """
        self.device = device
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg

        # Initialize SAM-2 camera predictor
        self._init_sam2_predictor()

        # Initialize tracking state
        self.is_initialized = False
        self.predictors = {}  # Dictionary to store predictors for each pixel key
        self.object_ids = {}  # Dictionary to store object IDs for each pixel key
        self.object_names = {}  # Dictionary to store object names for each pixel key

    def _init_sam2_predictor(self):
        """Initialize SAM-2 camera predictor."""
        try:
            # Add SAM-2 path to system path
            sam2_path = os.path.join(
                os.path.dirname(__file__),
                "../../third_party/segment-anything-2-real-time/",
            )
            sys.path.append(sam2_path)

            from sam2.build_sam import build_sam2_camera_predictor

            print("Initializing SAM-2 camera predictor...")
            self.base_predictor = build_sam2_camera_predictor(
                self.model_cfg,
                self.checkpoint,
                device=self.device,
                vos_optimized=True,
            )
            print("SAM-2 camera predictor loaded successfully!")

        except Exception as e:
            print(f"Error initializing SAM-2 camera predictor: {e}")
            raise

    def reset(
        self,
        pixel_key_images: Dict[str, np.ndarray],
        object_names: List[str],
        object_points: Dict[str, Dict[str, np.ndarray]],
    ):
        """
        Reset the tracker and initialize SAM-2 for each pixel key with the provided objects and points.

        Parameters:
        -----------
        pixel_key_images : Dict[str, np.ndarray]
            Dictionary mapping pixel keys to RGB images (H, W, 3) in uint8 format.
        object_names : List[str]
            List of object names to track.
        object_points : Dict[str, Dict[str, np.ndarray]]
            Dictionary mapping pixel keys to dictionaries of object names to point arrays.
            Each point array should be of shape (N, 2) where N is the number of points per object.
            Example: {
                'left_camera': {
                    'cup': np.array([[100, 200], [150, 250]]),
                    'bottle': np.array([[300, 400]])
                },
                'right_camera': {
                    'cup': np.array([[120, 220], [170, 270]]),
                    'bottle': np.array([[320, 420]])
                }
            }
        """
        print("Resetting segment tracker...")

        # Clear previous state
        self.predictors = {}
        self.object_ids = {}
        self.object_names = {}
        self.is_initialized = False

        # Initialize for each pixel key
        for pixel_key, image in pixel_key_images.items():
            print(f"Initializing tracker for pixel key: {pixel_key}")

            # Convert image to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR if image is uint8 and has typical BGR values
                if image.dtype == np.uint8:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
            else:
                raise ValueError(
                    f"Image for {pixel_key} must be RGB with shape (H, W, 3)"
                )

            # Create a new predictor instance for this pixel key
            sam2_path = os.path.join(
                os.path.dirname(__file__),
                "../../third_party/segment-anything-2-real-time/",
            )
            sys.path.append(sam2_path)
            from sam2.build_sam import build_sam2_camera_predictor

            predictor = build_sam2_camera_predictor(
                self.model_cfg, self.checkpoint, device=self.device
            )

            # Initialize with first frame
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.load_first_frame(image_rgb)

                # Get points for this pixel key
                if pixel_key not in object_points:
                    print(f"Warning: No points provided for pixel key {pixel_key}")
                    continue

                pixel_points = object_points[pixel_key]
                obj_id_counter = 1
                pixel_object_ids = {}
                pixel_object_names = {}

                # Add prompts for each object
                for obj_name in object_names:
                    if obj_name in pixel_points:
                        points = pixel_points[obj_name]

                        # Ensure points are in correct format
                        if isinstance(points, np.ndarray):
                            if points.ndim == 1:
                                points = points.reshape(1, 2)
                            elif points.ndim == 2 and points.shape[1] == 2:
                                pass  # Already correct format
                            else:
                                raise ValueError(
                                    f"Points for {obj_name} must be of shape (N, 2)"
                                )
                        else:
                            raise ValueError(
                                f"Points for {obj_name} must be a numpy array"
                            )

                        # Convert to float32
                        points = points.astype(np.float32)

                        # Add positive labels for all points
                        labels = np.ones(len(points), dtype=np.int32)

                        print(
                            f"  Adding {len(points)} points for object '{obj_name}' (ID: {obj_id_counter})"
                        )

                        # Add new prompt to predictor
                        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                            frame_idx=0,
                            points=points,
                            labels=labels,
                            obj_id=obj_id_counter,
                        )

                        # Store object mapping
                        pixel_object_ids[obj_name] = obj_id_counter
                        pixel_object_names[obj_id_counter] = obj_name
                        obj_id_counter += 1
                    else:
                        print(
                            f"  Warning: No points provided for object '{obj_name}' in {pixel_key}"
                        )

                # Store predictor and mappings
                self.predictors[pixel_key] = predictor
                self.object_ids[pixel_key] = pixel_object_ids
                self.object_names[pixel_key] = pixel_object_names

        self.is_initialized = True
        print("Segment tracker reset and initialized successfully!")

    def track(
        self,
        pixel_key_images: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Track objects in the provided images for each pixel key.

        Parameters:
        -----------
        pixel_key_images : Dict[str, np.ndarray]
            Dictionary mapping pixel keys to RGB images (H, W, 3) in uint8 format.

        Returns:
        --------
        masks : Dict[str, Dict[str, np.ndarray]]
            Dictionary mapping pixel keys to dictionaries of object names to binary masks.
            Each mask is a boolean array of shape (H, W).
            Example: {
                'left_camera': {
                    'cup': np.array([[True, False, ...], ...]),
                    'bottle': np.array([[False, True, ...], ...])
                },
                'right_camera': {
                    'cup': np.array([[True, False, ...], ...]),
                    'bottle': np.array([[False, True, ...], ...])
                }
            }
        """
        if not self.is_initialized:
            raise RuntimeError("Segment tracker not initialized. Call reset() first.")

        masks = {}

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for pixel_key, image in pixel_key_images.items():
                if pixel_key not in self.predictors:
                    print(f"Warning: No predictor found for pixel key {pixel_key}")
                    continue

                # Convert image to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    if image.dtype == np.uint8:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image_rgb = image
                else:
                    raise ValueError(
                        f"Image for {pixel_key} must be RGB with shape (H, W, 3)"
                    )

                predictor = self.predictors[pixel_key]
                pixel_object_ids = self.object_ids[pixel_key]
                pixel_object_names = self.object_names[pixel_key]

                # Track objects in current frame
                out_obj_ids, out_mask_logits = predictor.track(image_rgb)
                out_mask_logits = (out_mask_logits > 0.0).bool().cpu().numpy()

                # Extract masks for each object
                pixel_masks = {}
                for obj_name, obj_id in pixel_object_ids.items():
                    # Find mask for this object
                    mask = np.zeros(
                        (image_rgb.shape[0], image_rgb.shape[1]), dtype=bool
                    )

                    for i, tracked_obj_id in enumerate(out_obj_ids):
                        if tracked_obj_id == obj_id:
                            mask = out_mask_logits[i]

                            # Ensure mask is 2D
                            if mask.ndim > 2:
                                mask = mask.squeeze()
                            break

                    pixel_masks[obj_name] = mask

                masks[pixel_key] = pixel_masks

        return masks

    def get_tracked_objects(self) -> Dict[str, List[str]]:
        """
        Get the list of tracked objects for each pixel key.

        Returns:
        --------
        tracked_objects : Dict[str, List[str]]
            Dictionary mapping pixel keys to lists of tracked object names.
        """
        tracked_objects = {}
        for pixel_key, object_ids in self.object_ids.items():
            tracked_objects[pixel_key] = list(object_ids.keys())
        return tracked_objects

    def is_object_tracked(self, pixel_key: str, object_name: str) -> bool:
        """
        Check if a specific object is being tracked for a given pixel key.

        Parameters:
        -----------
        pixel_key : str
            The pixel key to check.
        object_name : str
            The name of the object to check.

        Returns:
        --------
        bool
            True if the object is being tracked, False otherwise.
        """
        if pixel_key in self.object_ids:
            return object_name in self.object_ids[pixel_key]
        return False

    def generate_mask_for_image(
        self,
        image: np.ndarray,
        object_points: Dict[str, np.ndarray],
        object_names: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Generate masks for objects in a single image using SAM-2.

        Parameters:
        -----------
        image : np.ndarray
            RGB image of shape (H, W, 3) in uint8 format.
        object_points : Dict[str, np.ndarray]
            Dictionary mapping object names to point arrays of shape (N, 2).
        object_names : List[str]
            List of object names to generate masks for.

        Returns:
        --------
        masks : Dict[str, np.ndarray]
            Dictionary mapping object names to binary masks of shape (H, W).
        """
        # Convert image to PIL for SAM-2
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype == np.uint8:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            raise ValueError("Image must be RGB with shape (H, W, 3)")

        pil_image = Image.fromarray(image_rgb)

        # Initialize SAM-2 image predictor
        sam2_path = os.path.join(
            os.path.dirname(__file__),
            "../../third_party/segment-anything-2-real-time/",
        )
        sys.path.append(sam2_path)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        image_predictor = SAM2ImagePredictor(
            build_sam2(self.model_cfg, self.checkpoint, device=self.device)
        )

        masks = {}

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            image_predictor.set_image(pil_image)

            for obj_name in object_names:
                if obj_name not in object_points:
                    print(f"Warning: No points provided for object '{obj_name}'")
                    continue

                points = object_points[obj_name]

                # Ensure points are in correct format
                if isinstance(points, np.ndarray):
                    if points.ndim == 1:
                        points = points.reshape(1, 2)
                    elif points.ndim == 2 and points.shape[1] == 2:
                        pass  # Already correct format
                    else:
                        raise ValueError(
                            f"Points for {obj_name} must be of shape (N, 2)"
                        )
                else:
                    raise ValueError(f"Points for {obj_name} must be a numpy array")

                # Convert to float32
                points = points.astype(np.float32)

                # Add positive labels for all points
                labels = np.ones(len(points), dtype=np.int32)

                # Generate masks using SAM-2
                sam_masks, sam_scores, _ = image_predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=True,
                )

                # Select the best mask (highest score)
                best_mask_idx = np.argmax(sam_scores)
                best_mask = sam_masks[best_mask_idx]

                # Convert to numpy if needed
                if hasattr(best_mask, "cpu"):
                    mask_np = best_mask.cpu().numpy()
                else:
                    mask_np = best_mask

                masks[obj_name] = mask_np.astype(bool)

        return masks
