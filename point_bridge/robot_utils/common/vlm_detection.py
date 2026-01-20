# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np
import einops

from point_bridge.robot_utils.common.utils import (
    triangulate_points,
    farthest_point_sampling,
    pixel2d_to_3d,
    transform_points,
    rotation_6d_to_matrix,
)
from point_bridge.robot_utils.common.franka_gripper_points import extrapoints
from point_bridge.detection_utils.utils import detect_segments
from point_bridge.detection_utils.segment_tracker import SegmentTracker
from point_bridge.detection_utils.point_tracker import PointTracker
import cv2


def sample_points_from_mask(mask, num_points=1000):
    """
    Sample points from a binary mask using random sampling.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W)
        num_points (int): Number of points to sample

    Returns:
        np.ndarray: Array of shape (num_points, 2) with (x, y) coordinates
    """

    #########################################################
    # Reduce mask size inwards by 20%
    # Find contours, shrink them, and fill new mask
    mask_uint8 = mask.astype(np.uint8)
    percent_shrink = 0.2
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    shrunken_mask = np.zeros_like(mask_uint8)
    for cnt in contours:
        if len(cnt) < 3:
            continue
        # Compute centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        # Move each point towards centroid
        cnt_shrunk = cnt.astype(np.float32)
        cnt_shrunk[:, 0, 0] = cx + (1 - percent_shrink) * (cnt_shrunk[:, 0, 0] - cx)
        cnt_shrunk[:, 0, 1] = cy + (1 - percent_shrink) * (cnt_shrunk[:, 0, 1] - cy)
        cnt_shrunk = cnt_shrunk.astype(np.int32)
        cv2.drawContours(shrunken_mask, [cnt_shrunk], -1, 1, thickness=-1)
    mask = shrunken_mask.astype(bool)
    #########################################################

    points = np.argwhere(mask)  # points as (h, w)
    if len(points) == 0:
        raise ValueError(f"Empty mask detected, returning {num_points} zero points")

    # Sample points if we have more than needed
    if len(points) > num_points:
        points = points[np.random.choice(points.shape[0], num_points, replace=False)]

    # Convert to (x, y) format
    points = points[..., ::-1]  # points as (w, h) or (x, y)

    # randomly repeat points if we have fewer points than requested
    if len(points) < num_points:
        points = np.concatenate(
            [
                points,
                points[
                    np.random.choice(
                        points.shape[0], num_points - len(points), replace=True
                    )
                ],
            ],
            axis=0,
        )

    return points


def get_vlm_points_using_segments_depth(
    ref_image,
    ref_depth,
    task_description,
    vlm_tracker,
    vlm_objects,
    is_first_step,
    num_points_per_obj,
    current_pose,
    intrinsic_matrix,
    extrinsic_matrix,
    sam_predictor,
    gemini_client,
    molmo_client,
    task_name,
    height_limits,
    width_limits,
    points2d=None,
    depths=None,
    scale_factor=0.5,
    use_noised_foreground_points=True,
):
    """
    Get VLM-based object masks and 3D points using depth information.
    Generates both image and ground truth depth for the first camera only,
    runs detect_segments to get object masks, randomly samples 1000 points
    from each mask, projects to 3D using depth and camera parameters,
    and applies farthest point sampling to reduce to desired number of points.

    Args:
        ref_image (np.ndarray): Reference camera image
        ref_depth (np.ndarray): Reference camera depth
        task_description (str): Task description for VLM detection
        vlm_tracker (SegmentTracker): VLM tracker instance
        vlm_objects (list): List of detected object names
        is_first_step (bool): Whether this is the first step (reset)
        num_points_per_obj (int): Number of points per object
        current_pose (np.ndarray): Current robot pose
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix
        extrinsic_matrix (np.ndarray): Camera extrinsic matrix
        sam_predictor: SAM predictor
        gemini_client: Gemini client
        molmo_client: Molmo client
        task_name (str): Task name for saving
        points2d (dict): Dictionary storing 2D points from previous step
        depths (dict): Dictionary storing depths from previous step
        scale_factor (float): Factor to scale down reference image for tracking (default: 0.5)

    Returns:
        tuple: (points3d_robot, points3d_objects, vlm_tracker, vlm_objects, points2d, depths)
    """

    # Store original image dimensions for scaling back later
    original_height, original_width = ref_image.shape[:2]

    # Create scaled image for tracking
    scaled_height = int(original_height * scale_factor)
    scaled_width = int(original_width * scale_factor)
    scaled_ref_image = cv2.resize(
        ref_image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR
    )

    if is_first_step or vlm_tracker is None:
        # Perform object detection using detect_segments
        data = detect_segments(
            image=ref_image,
            task_description=task_description,
            gemini_client=gemini_client,
            molmo_client=molmo_client,
            sam_predictor=sam_predictor,
            save_path=f"./tmp/{task_name}",
            height_limits=height_limits,
            width_limits=width_limits,
            use_noised_foreground_points=use_noised_foreground_points,
        )
        objects, masks = (
            data["objects"],
            data["masks"],
        )

        print(f"VLM detected objects: {objects}")

        # Check if we have any valid objects with masks
        if not objects:
            print("Warning: No objects detected by VLM, returning empty results")
            return [], [], vlm_tracker, vlm_objects, {}, {}

        # Store detected objects for future use
        vlm_objects = objects

        # Initialize segment tracker for tracking masks
        vlm_tracker = SegmentTracker(device="cuda")

        # Initialize tracker with the detected objects and their masks
        ref_object_points = {}
        for obj in objects:
            if obj in masks:
                # Sample a few points from the detected mask for initialization
                ref_points = sample_points_from_mask(masks[obj], num_points=5)

                # Adjust coordinates for cropping offset
                h, w = ref_image.shape[:2]
                h1 = int(h * height_limits[0])
                w1 = int(w * width_limits[0])
                ref_points[:, 0] += w1  # Add x offset
                ref_points[:, 1] += h1  # Add y offset

                # Convert to float for scaling operations
                ref_points = ref_points.astype(np.float32)

                # Scale points to match scaled image size for tracker initialization
                ref_points[:, 0] *= scale_factor  # Scale x coordinates
                ref_points[:, 1] *= scale_factor  # Scale y coordinates

                ref_object_points[obj] = ref_points

        # Initialize tracker with reference camera using scaled image
        pixel_key_images = {"pixels": scaled_ref_image}  # Use a default key
        all_object_points = {"pixels": ref_object_points}
        vlm_tracker.reset(pixel_key_images, objects, all_object_points)

        # Initialize storage for points and depths
        points2d = {}
        depths = {}

    # Get current masks using the tracker
    if vlm_tracker is None:
        print("Warning: No VLM tracker available, returning empty results")
        return [], [], vlm_tracker, vlm_objects, {}, {}

    # Track masks using the tracker with scaled image
    pixel_key_images = {"pixels": scaled_ref_image}  # Use a default key
    masks = vlm_tracker.track(pixel_key_images)
    current_masks = masks.get("pixels", {})

    # Process each detected object
    points3d_objects = []
    for obj in vlm_objects:
        if obj not in current_masks:
            print(
                f"Warning: Object {obj} not found in current masks, using previous points"
            )
            # Use points from previous step
            prev_points2d = points2d[obj]
            prev_depths = depths[obj]

            # Project 2D points to 3D using depth and camera parameters
            points3d = pixel2d_to_3d(
                prev_points2d,
                prev_depths,
                intrinsic_matrix,
                extrinsic_matrix,
            )  # in FR3 robot base frame

            # Apply farthest point sampling to reduce to desired number of points
            if len(points3d) > num_points_per_obj:
                points3d = farthest_point_sampling(points3d, num_points_per_obj)
            elif len(points3d) < num_points_per_obj:
                # randomly repeat points if we have fewer points than requested
                points3d = np.concatenate(
                    [
                        points3d,
                        points3d[
                            np.random.choice(
                                points3d.shape[0],
                                num_points_per_obj - len(points3d),
                                replace=True,
                            )
                        ],
                    ],
                    axis=0,
                )

            points3d_objects.append(points3d)
            continue

        mask = current_masks[obj]

        # Randomly sample 1000 points from the mask
        try:
            points2d_obj = sample_points_from_mask(mask, num_points=1000)

            # Scale points back up to original image size
            points2d_obj_scaled = points2d_obj.copy()
            points2d_obj_scaled[:, 0] = (
                points2d_obj_scaled[:, 0] / scale_factor
            )  # x coordinates
            points2d_obj_scaled[:, 1] = (
                points2d_obj_scaled[:, 1] / scale_factor
            )  # y coordinates

            # Ensure coordinates are within bounds and convert to integers for depth indexing
            points2d_obj_scaled[:, 0] = np.clip(
                points2d_obj_scaled[:, 0], 0, original_width - 1
            )
            points2d_obj_scaled[:, 1] = np.clip(
                points2d_obj_scaled[:, 1], 0, original_height - 1
            )
            points2d_obj_int = points2d_obj_scaled.astype(int)

            # Get corresponding depths for these points using original coordinates
            depths_obj = ref_depth[points2d_obj_int[:, 1], points2d_obj_int[:, 0]]

            # Store the scaled points for potential reuse in next step
            points2d[obj] = points2d_obj_scaled
            depths[obj] = depths_obj

        except:
            print(f"Warning: No valid points for object {obj}, using previous points")
            # Use points from previous step (these should already be in original scale)
            points2d_obj_scaled = points2d[obj]
            depths_obj = depths[obj]

        # Project 2D points to 3D using depth and camera parameters
        points3d = pixel2d_to_3d(
            points2d_obj_scaled,
            depths_obj,
            intrinsic_matrix,
            extrinsic_matrix,
        )  # in FR3 robot base frame

        # Apply farthest point sampling to reduce to desired number of points
        if len(points3d) > num_points_per_obj:
            points3d = farthest_point_sampling(points3d, num_points_per_obj)
        if len(points3d) < num_points_per_obj:
            # randomly repeat points if we have fewer points than requested
            points3d = np.concatenate(
                [
                    points3d,
                    points3d[
                        np.random.choice(
                            points3d.shape[0],
                            num_points_per_obj - len(points3d),
                            replace=True,
                        )
                    ],
                ],
                axis=0,
            )

        points3d_objects.append(points3d)
    points3d_objects = np.array(points3d_objects)  # (N, O, D)

    # Get robot points (same as get_gt_points)
    eef_pos, eef_ori = current_pose[:3], current_pose[3:9]
    eef_ori = rotation_6d_to_matrix(eef_ori)
    T_g2b = np.eye(4)  # FR3 gripper in FR3 robot base
    T_g2b[:3, :3] = eef_ori
    T_g2b[:3, 3] = eef_pos

    points3d_robot = []
    gripper_state = current_pose[-1]
    for idx, Tp in enumerate(extrapoints):
        if gripper_state > 0 and idx in [0, 1]:
            Tp = Tp.copy()
            Tp[1, 3] = 0.015 if idx == 0 else -0.015
        pt = T_g2b @ Tp  # pt in FR3 robot base
        pt = pt[:3, 3]
        points3d_robot.append(pt[:3])
    points3d_robot = np.array(points3d_robot)  # (N, D)

    return points3d_robot, points3d_objects, vlm_tracker, vlm_objects, points2d, depths


def get_vlm_points_using_point_tracking(
    pixel_key_images,
    task_description,
    pixel_keys,
    camera_projections,
    vlm_objects,
    is_first_step,
    num_points_per_obj,
    current_pose,
    sam_predictor,
    gemini_client,
    molmo_client,
    mast3r_model,
    dust3r_inference,
    task_name,
    height_limits,
    width_limits,
    pixelkey2camera,
    point_trackers=None,
    image_percentage_for_tracking=1.0,
    use_noised_foreground_points=True,
):
    """
    Get VLM-based object points using point tracking.
    Uses detect_segments to obtain object masks during reset (is_first_step == True),
    then uses PointTracker to track points across frames.
    Samples points from masks, computes multiview correspondence, and triangulates to get 3D points.

    Args:
        pixel_key_images (dict): Dictionary mapping pixel keys to images
        task_description (str): Task description for VLM detection
        pixel_keys (list): List of pixel keys
        camera_projections (dict): Dictionary mapping camera names to projection matrices
        vlm_objects (list): List of detected object names
        is_first_step (bool): Whether this is the first step (reset)
        num_points_per_obj (int): Number of points per object
        current_pose (np.ndarray): Current robot pose
        sam_predictor: SAM predictor
        gemini_client: Gemini client
        molmo_client: Molmo client
        mast3r_model: Mast3r model
        dust3r_inference: Dust3r inference function
        task_name (str): Task name for saving
        point_trackers (dict): Dictionary of PointTracker instances for each pixel key

    Returns:
        tuple: (points3d_robot, points3d_objects, vlm_objects, point_trackers)
    """

    # Initialize point trackers if not provided
    if point_trackers is None:
        point_trackers = {}
        for pixel_key in pixel_keys:
            point_trackers[pixel_key] = PointTracker(
                pixel_keys=[pixel_key],
                device="cuda",
            )

    # First step: detect objects and initialize trackers
    if is_first_step or vlm_objects is None:
        # Use the first camera as reference for VLM detection
        ref_pixel_key = pixel_keys[0]
        ref_image = pixel_key_images[ref_pixel_key]

        # Perform object detection using detect_segments
        data = detect_segments(
            image=ref_image,
            task_description=task_description,
            gemini_client=gemini_client,
            molmo_client=molmo_client,
            sam_predictor=sam_predictor,
            save_path=f"./tmp/{task_name}",
            height_limits=height_limits,
            width_limits=width_limits,
            use_noised_foreground_points=use_noised_foreground_points,
        )
        objects, masks = data["objects"], data["masks"]

        print(f"VLM detected objects: {objects}")

        # Check if we have any valid objects with masks
        if not objects:
            print("Warning: No objects detected by VLM, returning empty results")
            vlm_objects = []
            return [], [], vlm_objects, point_trackers

        # Store detected objects for future use
        vlm_objects = objects

        # Generate initial points for each object using farthest point sampling
        all_initial_points = []
        for object_name in objects:
            # Sample points from VLM mask for reference camera
            mask = masks[object_name]
            ref_points = sample_points_from_mask(mask, num_points=1000)

            # Adjust coordinates for cropping offset
            h, w = ref_image.shape[:2]
            h1 = int(h * height_limits[0])
            w1 = int(w * width_limits[0])
            ref_points[:, 0] += w1  # Add x offset
            ref_points[:, 1] += h1  # Add y offset

            # Apply FPS to get desired number of points
            if len(ref_points) > num_points_per_obj:
                # Add third dimension for FPS (z=1)
                points_3d = np.column_stack((ref_points, np.ones(len(ref_points))))
                points_3d = farthest_point_sampling(points_3d, num_points_per_obj)
                ref_points = points_3d[:, :2]  # Remove third dimension
            elif len(ref_points) < num_points_per_obj:
                # Pad with zeros
                padding = np.zeros((num_points_per_obj - len(ref_points), 2))
                ref_points = np.concatenate([ref_points, padding], axis=0)

            all_initial_points.append(ref_points)

        # Concatenate all object points for multiview correspondence
        if all_initial_points:
            all_initial_points = np.concatenate(all_initial_points, axis=0)

            # Resize frames to square for multiview correspondence
            square_size = 512  # Use square size for multiview
            square_frames = []
            for pixel_key in pixel_keys:
                frame = pixel_key_images[pixel_key]
                frame = cv2.resize(frame, (square_size, square_size))
                square_frames.append(frame)

            # Scale points to square image size
            h, w = ref_image.shape[:2]
            scale_x = square_size / w
            scale_y = square_size / h
            ref_points_square = all_initial_points.copy().astype(np.float32)
            ref_points_square[:, 0] *= scale_x
            ref_points_square[:, 1] *= scale_y

            # Compute multiview correspondence for all cameras
            print(f"Computing multiview correspondence for {len(objects)} objects...")
            all_points = compute_multiview_correspondence(
                mast3r_model=mast3r_model,
                dust3r_inference_func=dust3r_inference,
                images=square_frames,  # Use square frames
                ref_image_idx=0,  # First camera is reference
                ref_points=ref_points_square,
                device="cuda",
            )

            # Scale points back to original image size and distribute to cameras
            initial_points_per_camera = {}
            for i, pixel_key in enumerate(pixel_keys):
                points = all_points[i].copy()
                points[:, 0] /= scale_x
                points[:, 1] /= scale_y
                initial_points_per_camera[pixel_key] = points

            # Initialize point trackers with all object points
            for pixel_key in pixel_keys:
                initial_points = initial_points_per_camera[pixel_key]

                # Resize image for tracking if image_percentage_for_tracking < 1.0
                original_image = pixel_key_images[pixel_key]
                if image_percentage_for_tracking < 1.0:
                    h, w = original_image.shape[:2]
                    new_h = int(h * image_percentage_for_tracking)
                    new_w = int(w * image_percentage_for_tracking)
                    tracking_image = cv2.resize(original_image, (new_w, new_h))

                    # Scale initial points to resized image coordinates
                    initial_points[
                        :, 0
                    ] *= image_percentage_for_tracking  # x coordinates
                    initial_points[
                        :, 1
                    ] *= image_percentage_for_tracking  # y coordinates
                else:
                    tracking_image = original_image

                # Add third dimension (z=0) for PointTracker
                init_points_3d = np.column_stack(
                    (np.zeros(len(initial_points)), initial_points)
                )

                # Reset episode and add frame to tracker
                point_trackers[pixel_key].reset_episode()
                point_trackers[pixel_key].add_to_image_list(tracking_image, pixel_key)
                point_trackers[pixel_key].track_points(
                    pixel_key, is_first_step=True, init_points=init_points_3d
                )
                # Call track_points again to get the initial tracked points
                point_trackers[pixel_key].track_points(pixel_key)

    # Track points for current step
    points2d_objects_per_camera = {}

    # Track points for each camera
    for pixel_key in pixel_keys:
        # Resize image for tracking if image_percentage_for_tracking < 1.0
        original_image = pixel_key_images[pixel_key]
        if image_percentage_for_tracking < 1.0:
            h, w = original_image.shape[:2]
            new_h = int(h * image_percentage_for_tracking)
            new_w = int(w * image_percentage_for_tracking)
            tracking_image = cv2.resize(original_image, (new_w, new_h))
        else:
            tracking_image = original_image

        point_trackers[pixel_key].add_to_image_list(tracking_image, pixel_key)
        point_trackers[pixel_key].track_points(pixel_key)

        # Get tracked points
        tracked_points = point_trackers[pixel_key].get_points(pixel_key)["object"][0]

        # Rescale points back to original image coordinates if image was resized
        if image_percentage_for_tracking < 1.0:
            h, w = original_image.shape[:2]
            tracked_points[:, 0] /= image_percentage_for_tracking  # x coordinates
            tracked_points[:, 1] /= image_percentage_for_tracking  # y coordinates

        points2d_objects_per_camera[pixel_key] = tracked_points

    # Triangulate 2D points to 3D for each object
    points3d_objects = []

    # Distribute tracked points among objects (assuming equal distribution)
    if vlm_objects and len(points2d_objects_per_camera) > 0:
        # Get points from first camera to determine number of points per object
        first_camera_key = list(points2d_objects_per_camera.keys())[0]
        total_points = len(points2d_objects_per_camera[first_camera_key])
        points_per_obj = total_points // len(vlm_objects)

        for i, object_name in enumerate(vlm_objects):
            # Collect points and projection matrices for triangulation
            points_list = []
            projection_matrices = []

            start_idx = i * points_per_obj
            end_idx = (
                start_idx + points_per_obj if i < len(vlm_objects) - 1 else total_points
            )

            for pixel_key in pixel_keys:
                if pixel_key in points2d_objects_per_camera:
                    points2d = points2d_objects_per_camera[pixel_key][start_idx:end_idx]

                    # Get camera parameters
                    camera_name = pixelkey2camera[pixel_key]
                    projection_matrix = camera_projections[camera_name]

                    projection_matrices.append(projection_matrix)
                    points_list.append(points2d)

            # Triangulate points
            points3d = triangulate_points(projection_matrices, points_list)[:, :3]

            # Apply FPS if needed
            if len(points3d) > num_points_per_obj:
                points3d = farthest_point_sampling(points3d, num_points_per_obj)
            elif len(points3d) < num_points_per_obj:
                # randomly repeat points if we have fewer points than requested
                points3d = np.concatenate(
                    [
                        points3d,
                        points3d[
                            np.random.choice(
                                points3d.shape[0],
                                num_points_per_obj - len(points3d),
                                replace=True,
                            )
                        ],
                    ],
                    axis=0,
                )

            points3d_objects.append(points3d)

    # Ensure all objects have the same number of points
    max_points = max(len(obj_points) for obj_points in points3d_objects)
    for i in range(len(points3d_objects)):
        if len(points3d_objects[i]) < max_points:
            # Pad with zeros
            padding = np.zeros((max_points - len(points3d_objects[i]), 3))
            points3d_objects[i] = np.concatenate([points3d_objects[i], padding], axis=0)
        elif len(points3d_objects[i]) > max_points:
            # Truncate to max_points
            points3d_objects[i] = points3d_objects[i][:max_points]

    points3d_objects = np.array(points3d_objects)

    # Get robot points (same as get_gt_points)
    eef_pos, eef_ori = current_pose[:3], current_pose[3:9]
    eef_ori = rotation_6d_to_matrix(eef_ori)
    T_g2b = np.eye(4)  # FR3 gripper in FR3 robot base
    T_g2b[:3, :3] = eef_ori
    T_g2b[:3, 3] = eef_pos

    points3d_robot = []
    gripper_state = current_pose[-1]
    for idx, Tp in enumerate(extrapoints):
        if gripper_state > 0 and idx in [0, 1]:
            Tp = Tp.copy()
            Tp[1, 3] = 0.015 if idx == 0 else -0.015
        pt = T_g2b @ Tp  # pt in FR3 robot base
        pt = pt[:3, 3]
        points3d_robot.append(pt[:3])
    points3d_robot = np.array(points3d_robot)  # (N, D)

    return points3d_robot, points3d_objects, vlm_objects, point_trackers
