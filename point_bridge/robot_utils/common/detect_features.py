# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import re
import cv2
import base64
import numpy as np

from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

from google.genai import types
from sklearn.cluster import KMeans


def encode_image(image_path):
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def extract_objects(text):
    # Regular expression pattern to match the objects - more robust with whitespace handling
    pattern = r"<objects>\s*\[(.*?)\]\s*</objects>"

    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # Process each match
    all_objects = []
    for match in matches:
        # Split the objects and strip whitespace
        objects = [obj.strip() for obj in match.split(",")]
        # all_objects.extend(objects)
        all_objects.extend([obj.lower() for obj in objects])

    # keep only unique objects in order
    all_objects = list(dict.fromkeys(all_objects))

    return all_objects


def extract_coordinates(message):
    """
    Extract coordinates from a message that may contain single or multiple points.

    Args:
        message (str): Message containing coordinates in either format:
            - Single point: <point x="..." y="...">
            - Multiple points: x1, y1, x2, y2, ...

    Returns:
        tuple or list: Single point as (x, y) or multiple points as [(x1, y1), (x2, y2), ...]
    """
    # Try to match XML point format with variable number of points
    # This handles both <point x="..." y="..."> and <point x1="..." y1="..." x2="..." y2="...">
    # Also handles <points x1="..." y1="..." x2="..." y2="..."> format
    xml_point_pattern = r"<(?:point|points)\s+([^>]+)>"
    xml_match = re.search(xml_point_pattern, message)

    if xml_match:
        attributes = xml_match.group(1)

        # Check if it's the old single point format
        single_point_attr_pattern = r'x="(\d+\.?\d*)"\s+y="(\d+\.?\d*)"'
        single_attr_match = re.search(single_point_attr_pattern, attributes)

        if single_attr_match:
            # Single point format
            x = float(single_attr_match.group(1))
            y = float(single_attr_match.group(2))
            return x, y

        # Check for multiple points format (x1, y1, x2, y2, ...)
        multi_point_attr_pattern = r'x(\d+)="(\d+\.?\d*)"\s+y\1="(\d+\.?\d*)"'
        multi_attr_matches = re.findall(multi_point_attr_pattern, attributes)

        if multi_attr_matches:
            # Create list of (point_num, x, y) tuples and sort by point number
            point_data = [
                (int(match[0]), float(match[1]), float(match[2]))
                for match in multi_attr_matches
            ]
            point_data.sort(key=lambda x: x[0])  # Sort by point number

            # Only return the first point
            first_point = point_data[0]
            return (first_point[1], first_point[2])  # Return as tuple (x, y)

    # Try to match multiple points format (x1, y1, x2, y2, ...)
    # Look for a sequence of numbers that could be coordinates
    multi_point_pattern = (
        r"(\d+\.?\d*)\s*,\s*(\d+\.?\d*)(?:\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*))*"
    )
    multi_matches = re.findall(multi_point_pattern, message)

    if multi_matches:
        # Flatten the matches and convert to floats
        coords = []
        for match in multi_matches:
            # Each match is a tuple of 2 or 4 numbers
            if len(match) >= 2:
                x = float(match[0])
                y = float(match[1])
                coords.append((x, y))

                # If there are 4 numbers, add the second point
                if len(match) >= 4 and match[2] and match[3]:
                    x2 = float(match[2])
                    y2 = float(match[3])
                    coords.append((x2, y2))

        if coords:
            # Only return the first point
            return coords[0]

    # Try to match the format "x1= y1= x2= y2="
    x1y1x2y2_pattern = r"x1\s*=\s*(\d+\.?\d*)\s*y1\s*=\s*(\d+\.?\d*)\s*x2\s*=\s*(\d+\.?\d*)\s*y2\s*=\s*(\d+\.?\d*)"
    x1y1x2y2_match = re.search(x1y1x2y2_pattern, message)

    if x1y1x2y2_match:
        x1 = float(x1y1x2y2_match.group(1))
        y1 = float(x1y1x2y2_match.group(2))
        # Only return the first point
        return (x1, y1)

    return None


def extract_image_number(response: str) -> int:
    """
    Extracts the image number from a string formatted as <image>X</image>

    Args:
        response: The text response containing the image tag

    Returns:
        Integer value of the image number (1-100) if found, otherwise None
    """
    match = re.search(r"<image>(\d+)</image>", response)
    return int(match.group(1)) if match else None


def remove_noise_from_mask(binary_mask, min_area=100, kernel_size=(3, 3)):
    """
    Removes noise from a binary mask by applying morphological operations
    and filtering small connected components.

    Parameters:
        binary_mask (numpy.ndarray): Input binary mask (values: 0 or 255).
        min_area (int): Minimum area of connected components to keep.
        kernel_size (tuple): Size of the structuring element for morphology.

    Returns:
        numpy.ndarray: Cleaned binary mask.
    """
    # Ensure the mask is binary
    # _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    binary_mask = (binary_mask * 255).astype(np.uint8)

    # Perform morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened_mask, connectivity=8
    )

    # Create an empty mask for cleaned components
    cleaned_mask = np.zeros_like(binary_mask)

    # Keep only components with area >= min_area
    for i in range(1, num_labels):  # Skip background label (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 1  # 255

    return cleaned_mask


def dust3r_images(images, img_norm, square_ok=False):
    """
    convert all images in a list to proper input format for DUSt3R
    """

    imgs = []
    for image in images:
        img = Image.fromarray(image)
        W, H = img.size
        assert W == H, "image is not square"
        cx, cy = W // 2, H // 2
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        imgs.append(
            dict(
                img=img_norm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )

    return imgs


def detect_keypoints(
    gemini_client,
    molmo_client,
    sam_predictor,
    image,
    pixel_key,
    # task_name,
    height_limits,
    width_limits,
    dift_correspondence_feature_func,
    num_points_per_obj,
    save_path="./tmp",
    task_description="task",
    object_pos=None,
    object_labels=None,
):
    """
    Use Gemini for reasoning and Molmo for pointing to objects in the image.

    Args:
        gemini_client: Gemini client for reasoning
        molmo_client: Molmo ZMQ client for pointing
        sam_predictor: SAM2 predictor for segmentation
        image (np.ndarray): Input image
        task_description (str): Description of the task
        height_limits (tuple): Height limits for cropping the image
        width_limits (tuple): Width limits for cropping the image
        dift_correspondence_feature_func: Function for computing DIFT features
        num_points_per_obj (int): Number of keypoints to detect per object

    Returns:
        dict: Data for the task
    """

    # create tmp directory for saving data
    task_save_path = Path(str(save_path))
    # task_description = name2desc[task_name]

    # crop image
    h, w = image.shape[:2]
    h1, h2 = int(h * height_limits[0]), int(h * height_limits[1])
    w1, w2 = int(w * width_limits[0]), int(w * width_limits[1])
    cropped_image = image[h1:h2, w1:w2]

    # save image
    Image.fromarray(image).save(task_save_path / "image.png")
    cropped_image_save_path = task_save_path / "cropped_image.png"
    Image.fromarray(cropped_image).save(cropped_image_save_path)

    # find objects required for the task
    gemini_prompt = f"I want to perform the task of '{task_description}'. \
            Given me the order of the tasks needed to be performed to accomplish this. Once you \
            have created an order of tasks, list down all the objects I need to interact with \
            in order to accomplish the tasks. List the object in the order they are interacted \
            with and provide your answer in the format '<objects>[objectA, objectB, ...]</objects>'. \
            Ignore objects like hand, finger, etc. Reason about parts of an object a robot interacts with. For example, to turn off a stove, \
            the knob is the part of the stove that the robot interacts with. Also, to open a microwave, \
            the handle is the part of the microwave that the robot interacts with. Also, to open a door, \
            the handle is the part of the door that the robot interacts with. Also consider the \
            image provided and reason about the objects in the image. Only consider objects in front \
            of the robot and on the same table top as the top. Also include the placement position of the object \
            when asked to put or place the object. For example, to put a bowl on the table, the placement position is the table top. To place \
            an object on the stove top, the placement position is the stove top."
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[gemini_prompt],
        config={
            "temperature": 0.0,  # Adjust as needed
            "top_p": 0.95,
        },
    )
    all_objects = extract_objects(response.text)
    print(f"Objects: {all_objects}")

    # find points for each object
    points = {}
    for object_name in all_objects:
        if object_pos is not None:
            object_present = any(
                [obj.lower() in object_name.lower() for obj in object_labels]
            )
            if object_present:
                x, y = object_pos
                x = (x - w1) / (w2 - w1) * 100
                y = (y - h1) / (h2 - h1) * 100
                object_pos = (x, y)
                points[object_name] = object_pos
                print(f"Object point for {object_name}: {object_pos}")
                continue

        prompt = f"Mark a point on the {object_name}."
        image_path = str(cropped_image_save_path)
        output = molmo_client.send_prompt(prompt, image_path)
        point = extract_coordinates(output)

        points[object_name] = point
        print(f"Molmo point for {object_name}: {point}")

        # If no point found, skip this object
        if point is None:
            print(f"Warning: No point found for {object_name}, skipping...")
            continue

    # save response in txt file
    with open(task_save_path / "response.txt", "w") as f:
        f.write(response.text)

    # save separate image for each object
    for object_name, point in points.items():
        if point == None or point == "None":
            continue
        x, y = point
        x, y = int(x * cropped_image.shape[1] / 100), int(
            y * cropped_image.shape[0] / 100
        )

        save_img = cv2.circle(cropped_image.copy(), (x, y), 5, (0, 255, 0), -1)
        cv2.imwrite(
            str(task_save_path / f"molmo_points_{object_name}.png"),
            save_img,
        )

    # SAM2 + Gemini for segmentation
    MASKS, MASKED_IMAGES = {}, {}
    for idx, obj in enumerate(all_objects):
        point = np.array(points[obj])
        point[0] = point[0] * cropped_image.shape[1] // 100
        point[1] = point[1] * cropped_image.shape[0] // 100

        point_coords = np.array([point])

        ############### with foreground points ################
        point_labels = [1]  # 1 for foreground, 0 for background
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam_predictor.set_image(cropped_image.copy())
            sam_masks, sam_scores, sam_logits = sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        sorted_ind = np.argsort(sam_scores)[::-1]
        sam_masks = sam_masks[sorted_ind]
        sam_scores = sam_scores[sorted_ind]
        sam_logits = sam_logits[sorted_ind]

        masks = []
        masked_images = []
        masked_images_paths = []
        for i, mask in enumerate(sam_masks):
            mask = remove_noise_from_mask(mask)
            masked_image = np.array(cropped_image) * mask[..., None]
            masked_image = masked_image.astype(np.uint8)
            path = task_save_path / f"masked_image_{obj}_{len(masks)}_foreground.png"
            Image.fromarray(masked_image).save(path)
            masks.append(mask)
            masked_images.append(masked_image)
            masked_images_paths.append(path)

        ############## with point + noise ################
        num_points = 5
        # point_labels = [1] * num_points  # 1 for foreground, 0 for background
        point_noise = np.random.randint(-10, 10, (num_points, 2))
        point_coords = point_coords + point_noise
        point_coords[:, 0] = np.clip(point_coords[:, 0], 0, cropped_image.shape[1] - 1)
        point_coords[:, 1] = np.clip(point_coords[:, 1], 0, cropped_image.shape[0] - 1)

        for point_idx in range(len(point_coords)):
            point_coord = [point_coords[point_idx]]
            point_label = [1]  # 1 for foreground, 0 for background

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sam_predictor.set_image(cropped_image.copy())
                sam_masks, sam_scores, sam_logits = sam_predictor.predict(
                    point_coords=point_coord,
                    point_labels=point_label,
                    multimask_output=True,
                )

            sorted_ind = np.argsort(sam_scores)[::-1]
            sam_masks = sam_masks[sorted_ind]
            sam_scores = sam_scores[sorted_ind]
            sam_logits = sam_logits[sorted_ind]

            for i, mask in enumerate(sam_masks):
                masked_image = np.array(cropped_image) * mask[..., None]
                masked_image = masked_image.astype(np.uint8)
                path = (
                    task_save_path
                    / f"masked_image_{obj}_{len(masks)}_foreground_noise.png"
                )
                Image.fromarray(masked_image).save(path)
                masks.append(mask)
                masked_images.append(masked_image)
                masked_images_paths.append(path)

        # convert images to b64 for Gemini
        b64_images = []
        for path in masked_images_paths:
            b64_images.append(
                types.Part.from_bytes(
                    data=Path(path).read_bytes(), mime_type="image/png"
                )
            )

        text_prompt = f"Following are images containing results of semantic segmentation \
                        for the object '{obj}'. From the provided images, choose the one \
                        that best represents the object. If none of the images are correct, \
                        choose the image that is closest to the object. Some instructions that \
                        you must follow are: \
                        1) Choose the image that contains the entire object. For instance, \
                        for the object 'moka pot', choose the image that contains the entire \
                        moka pot. For the object 'black bowl', choose the image that contains \
                        the entire black bowl. For the object 'stove burner', choose the image \
                        that contains the entire stove burner. \
                        2) When dealing with objects like drawer handle or door handle,\
                        choose the image containing the handle along with the drawer or door.\
                        3) The images your are provided with are the results of semantic segmentation. \
                        The background is colored in black and the object is colored. \
                        Take this into consideration while making your decision and do not blindly \
                        choose the image where the object is part of the background. \
                        \
                        The images are provided in order [1, 2, 3, ...]. Give your answer \
                        in the format '<image>number</image>'."
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[text_prompt, *b64_images],
            config={
                "temperature": 0.0,  # Adjust as needed
                "top_p": 0.95,
            },
        )

        # extract image number and save image
        image_num = extract_image_number(response.text)
        Image.fromarray(masked_images[image_num - 1]).save(
            task_save_path / f"final_image_{obj}.png"
        )
        MASKS[obj] = masks[image_num - 1]
        MASKED_IMAGES[obj] = masked_images[image_num - 1]

    ######################## Generate key points with DIFT features ########################

    shape = cropped_image.shape
    image_pil = Image.fromarray(cropped_image)
    dift_feat = dift_correspondence_feature_func(image_pil, pixel_key)
    dift_feat = torch.nn.Upsample(size=shape[:2], mode="bilinear", align_corners=True)(
        dift_feat
    )[0]
    dift_feat = dift_feat.cpu().numpy()

    POINTS = {}
    pts_image = image.copy()
    pts_image_cropped = cropped_image.copy()
    for object_name in all_objects:
        mask = MASKS[object_name]

        # get dift features for mask region
        ind_h, ind_w = np.where(mask > 0)
        features = []
        for h, w in zip(ind_h, ind_w):
            features.append(dift_feat[:, h, w])

        # Run kmeans
        kmeans = KMeans(
            n_clusters=num_points_per_obj, random_state=0, max_iter=1000
        ).fit(features)
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        # Get idx of elements in features closest to each cluster center
        cluster_idx = []
        for center in cluster_centers:
            dists = np.linalg.norm(features - center, axis=1)
            cluster_idx.append(np.argmin(dists))

        pts = []
        for idx in cluster_idx:
            h, w = ind_h[idx], ind_w[idx]
            cv2.circle(pts_image_cropped, (w, h), 5, (0, 255, 0), -1)
            h, w = h + h1, w + w1
            cv2.circle(pts_image, (w, h), 5, (0, 255, 0), -1)
            pts.append([w, h])
        POINTS[object_name] = np.array(pts)

    # save image
    Image.fromarray(pts_image).save(task_save_path / f"{pixel_key}_keypoints.png")
    Image.fromarray(pts_image_cropped).save(
        task_save_path / f"{pixel_key}_keypoints_cropped.png"
    )

    # return data
    data = {
        f"{pixel_key}": image,
        f"{pixel_key}_cropped": cropped_image,
        "objects": all_objects,
        "points": POINTS,
        # "height_limits": [h1, h2],
        # "width_limits": [w1, w2],
    }
    return data

    # for camera in other_cameras:
    #     other_pixel_key = camera2pixelkey[camera]
    #     other_camera_image = other_camera_images[camera]
    #     save_data[other_pixel_key] = other_camera_image
    # with open(task_save_path / f"data.pkl", "wb") as f:
    #     pkl.dump(save_data, f)


def compute_multiview_correspondence(
    mast3r_model,
    dust3r_inference_func,
    images,
    ref_image_idx,
    ref_points,
    device="cuda",
):
    """
    Compute multi-view correspondence between images using MAST3R and DUSt3R.

    Args:
        mast3r_model: MAST3R model for multi-view correspondence
        dust3r_inference_func: Function for running DUSt3R inference
        images (list): List of images (images must be square)
        ref_image_idx (int): Index of the reference image
        ref_points (np.ndarray): Reference points for correspondence on reference image.

    Returns:
        List of points for each image in images
    """

    ImgNorm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dust3r_imgs = dust3r_images(images, ImgNorm, square_ok=True)

    points = [ref_points]
    images_indices = [i for i in range(len(images)) if i != ref_image_idx]
    for image_idx in images_indices:
        imgs = [dust3r_imgs[ref_image_idx], dust3r_imgs[image_idx]]

        output = dust3r_inference_func(
            [tuple(imgs)], mast3r_model, device, batch_size=1, verbose=False
        )

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]

        desc1, desc2 = (
            pred1["desc"].squeeze(0).detach(),
            pred2["desc"].squeeze(0).detach(),
        )

        out_coords = np.zeros(ref_points.shape)
        for idx, coord in enumerate(ref_points):
            h, w = int(coord[1]), int(coord[0])
            # curr_image_shape = dust3r_imgs[i]["true_shape"][0]

            src_vec = desc1[h, w][None, None].clone()
            trg_vec = desc2

            src_vec /= src_vec.norm(dim=-1, keepdim=True)
            trg_vec /= trg_vec.norm(dim=-1, keepdim=True)
            cos_map = (src_vec * trg_vec).sum(dim=-1).cpu().numpy()

            max_hw = np.unravel_index(cos_map.argmax(), cos_map.shape)
            out_coords[idx, 1], out_coords[idx, 0] = int(max_hw[0]), int(max_hw[1])
        points.append(out_coords)
    return points


def compute_language_features(language_model, prompts_list):
    """
    Compute language features for all prompts using a language model.

    Args:
        language_model: Language model for computing features
        prompts_list (list): List of prompts

    Returns:
        np.ndarray: Language features for each prompt
    """
    lang_features = np.array([language_model.encode(prompt) for prompt in prompts_list])
    return lang_features


def compute_semantic_features(
    images,
    points,
    pixel_keys,
    dift_correspondence_feature_func,
):
    """
    Compute semantic features for each points in the images.

    Args:
        images (list): List of images
        points (list): List of points on each image
        pixel_keys (list): List of pixel keys for each image
        dift_correspondence_feature_func: Function for computing DIFT features

    Returns:
        list: List of semantic features for each object
    """
    semantic_features = []
    for image, point, pixel_key in zip(images, points, pixel_keys):
        shape = image.shape
        image_pil = Image.fromarray(image)
        dift_feat = dift_correspondence_feature_func(image_pil, pixel_key)
        dift_feat = torch.nn.Upsample(
            size=shape[:2], mode="bilinear", align_corners=True
        )(dift_feat)[0]
        dift_feat = dift_feat.cpu().numpy()

        point_feat = np.array([dift_feat[:, int(pt[1]), int(pt[0])] for pt in point])
        semantic_features.append(point_feat)
    return semantic_features
