# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import re
import os
import sys
import cv2
import yaml
import shutil
import base64
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from google.genai import types


# Read root directory from local.yaml
def get_root_dir(cfg_path):
    with open(cfg_path, "r") as f:
        local_cfg = yaml.safe_load(f)
    return local_cfg["root_dir"]


PB_DIR = get_root_dir(os.path.join(os.path.dirname(__file__), "../cfgs/local.yaml"))


def init_models_for_vlm_detection():
    sys.path.append(f"{PB_DIR}/third_party/mast3r")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from google import genai
    from google.genai import types
    from model_servers.molmo.client import MolmoZMQClient
    from sentence_transformers import SentenceTransformer
    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference as dust3r_inference

    # SAM2
    print("Initializing SAM2 ...")
    checkpoint = f"{PB_DIR}/third_party/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    # gemini client
    print("Initializing Gemini client ...")
    gemini_client = genai.Client(
        api_key="<add API key here>"
    )  # TODO: add API key 

    # molmo client
    print("Initializing Molmo client ...")
    molmo_client = MolmoZMQClient(server_address="localhost:45000")

    # load sentence transformer
    print("Initializing Sentence Transformer ...")
    lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # mast3r
    print("Initializing MAST3R ...")
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    mast3r_model = AsymmetricMASt3R.from_pretrained(model_name).to("cuda")

    return (
        sam_predictor,
        gemini_client,
        molmo_client,
        lang_model,
        mast3r_model,
        dust3r_inference,
    )


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


def detect_segments(
    image,
    task_description,
    gemini_client,
    molmo_client,
    sam_predictor,
    height_limits=None,
    width_limits=None,
    save_path="./tmp",
    object_pos=None,
    object_labels=None,
    use_noised_foreground_points=True,
):
    """
    Use Gemini for reasoning and Molmo for pointing to objects in the image.

    Args:
        image (np.ndarray): Input image (RGB)
        task_description (str): Description of the task
        gemini_client: Gemini client for reasoning
        molmo_client: Molmo ZMQ client for pointing
        sam_predictor: SAM2 predictor for segmentation
        height_limits (tuple): Height limits for cropping the image
        width_limits (tuple): Width limits for cropping the image

    Returns:
        dict: Data for the task
    """

    # convert image to BGR
    image = image[..., ::-1]

    # create tmp directory for saving data
    task_save_path = Path(str(save_path))
    if task_save_path.exists():
        shutil.rmtree(task_save_path)
    task_save_path.mkdir(parents=True, exist_ok=True)
    # save image
    cv2.imwrite(str(task_save_path / "image.png"), image)

    # crop image
    if height_limits is not None and width_limits is not None:
        h, w = image.shape[:2]
        h1, h2 = (
            (int(h * height_limits[0]), int(h * height_limits[1]))
            if height_limits is not None
            else (0, h)
        )
        w1, w2 = (
            (int(w * width_limits[0]), int(w * width_limits[1]))
            if width_limits is not None
            else (0, w)
        )
        image = image[h1:h2, w1:w2]
    # save cropped image
    cropped_image_save_path = task_save_path / "cropped_image.png"
    cv2.imwrite(str(cropped_image_save_path), image)

    gemini_prompt = f"I want to perform the task of '{task_description}'. \
            Given me the order of the tasks needed to be performed to accomplish this. Once you \
            have created an order of tasks, list down all the objects I need to interact with \
            in order to accomplish the tasks. List the object in the order they are interacted \
            with and provide your answer in the format '<objects>[objectA, objectB, ...]</objects>'. \
            Ignore the agent performing the task such as hands, robots, etc. Also consider the \
            image provided and reason about the objects in the image. \
            \
            Make sure to consider accurate object part. For instance, when opening a drawer, consider \
            the drawer instead of just the handle. Similarly, when opening a door, consider \
            the whole door instead of just the handle. For a mug, consider the whole mug instead of just the handle. \
            \
            Pay close attention to the description of the task and only include objects that are \
            relevant to the task. For example, if the task is to close a drawer, do not include \
            the bowl on the table in the list of objects. \
            \
            Also, do not include objects that are not present in the image. For example, if the image \
            does not contain a bowl, do not include the bowl in the list of objects."
    # convert images to b64 for Gemini
    b64_images = []
    b64_images.append(
        types.Part.from_bytes(
            data=Path(cropped_image_save_path).read_bytes(), mime_type="image/png"
        )
    )
    # query Gemini
    print("Querying Gemini ...")
    import time

    start_time = time.time()
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[gemini_prompt, *b64_images],
        config={
            "temperature": 0.0,  # Adjust as needed
            "top_p": 0.95,
        },
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    all_objects = extract_objects(response.text)
    print(f"Objects: {all_objects}")

    # Remove repetitive object names
    prompt = f"I want to perform the task of '{task_description}'. \
               I have the following objects that are relevant to the task: {all_objects}. \
               Given the set of object, give me the minimal set of most accurate objects that does not contain any repetitive objects. \
               For example, if you have cabinet and cabinet drawer, give me the drawer only. \
               Another example is if you have cabinet and cabinet top, give me only the cabinet top only. \
               Give your answer in the format '<objects>[objectA, objectB, ...]</objects>'."
    print("Querying Gemini 2 ...")
    start_time = time.time()
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config={
            "temperature": 0.0,  # Adjust as needed
            "top_p": 0.95,
        },
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    all_objects = extract_objects(response.text)
    print(f"Refined objects: {all_objects}")

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
        print("Querying Molmo ...")
        start_time = time.time()
        output = molmo_client.send_prompt(prompt, image_path)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
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

        # Handle single point (now always a tuple)
        if isinstance(point, tuple):
            x, y = point
            x, y = int(x * image.shape[1] / 100), int(y * image.shape[0] / 100)

            save_img = cv2.circle(image.copy(), (x, y), 5, (0, 255, 0), -1)
            cv2.imwrite(
                str(task_save_path / f"molmo_points_{object_name}.png"),
                save_img,
            )

    # SAM2 + Gemini for segmentation
    MASKS, MASKED_IMAGES = {}, {}
    for idx, obj in enumerate(all_objects):
        # Skip objects that don't have valid points
        if obj not in points or points[obj] is None:
            print(f"Skipping {obj} - no valid point found")
            continue

        point_data = points[obj]

        # Handle single point (now always a tuple)
        if isinstance(point_data, tuple):
            x, y = point_data
            x_pixel = x * image.shape[1] // 100
            y_pixel = y * image.shape[0] // 100
            point_coords = np.array([[x_pixel, y_pixel]])
        else:
            print(f"Skipping {obj} - invalid point format")
            continue

        ############### with foreground points ################
        point_labels = [1]  # 1 for foreground, 0 for background
        print("Querying SAM inference ...")
        start_time = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam_predictor.set_image(image.copy())
            sam_masks, sam_scores, sam_logits = sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        sorted_ind = np.argsort(sam_scores)[::-1]
        sam_masks = sam_masks[sorted_ind]
        # only take higher score masks
        sam_masks = sam_masks[:1]

        masks = []
        masked_images = []
        masked_images_paths = []
        for i, mask in enumerate(sam_masks):
            mask = remove_noise_from_mask(mask)
            masked_image = np.array(image) * mask[..., None]
            masked_image = masked_image.astype(np.uint8)
            path = task_save_path / f"masked_image_{obj}_{len(masks)}_foreground.png"
            Image.fromarray(masked_image).save(path)
            masks.append(mask)
            masked_images.append(masked_image)
            masked_images_paths.append(path)

        if use_noised_foreground_points:
            ############## with point + noise ################
            num_points = 3
            height, width = image.shape[0], image.shape[1]
            noise_x = np.random.uniform(
                -0.05 * width, 0.05 * width, size=(num_points, 1)
            )
            noise_y = np.random.uniform(
                -0.05 * height, 0.05 * height, size=(num_points, 1)
            )
            point_noise = np.concatenate([noise_x, noise_y], axis=1).astype(int)
            # add noise to point coordinates
            point_coords = point_coords + point_noise
            point_coords[:, 0] = np.clip(point_coords[:, 0], 0, image.shape[1] - 1)
            point_coords[:, 1] = np.clip(point_coords[:, 1], 0, image.shape[0] - 1)

            for point_idx in range(len(point_coords)):
                point_coord = [point_coords[point_idx]]
                point_label = [1]  # 1 for foreground, 0 for background

                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    sam_predictor.set_image(image.copy())
                    sam_masks, sam_scores, sam_logits = sam_predictor.predict(
                        point_coords=point_coord,
                        point_labels=point_label,
                        multimask_output=True,
                    )

                sorted_ind = np.argsort(sam_scores)[::-1]
                sam_masks = sam_masks[sorted_ind]
                # only take higher score masks
                sam_masks = sam_masks[:1]

                for i, mask in enumerate(sam_masks):
                    masked_image = np.array(image) * mask[..., None]
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
            # add cropped image to b64 images
            b64_images.append(
                types.Part.from_bytes(
                    data=Path(cropped_image_save_path).read_bytes(),
                    mime_type="image/png",
                )
            )
            for path in masked_images_paths:
                b64_images.append(
                    types.Part.from_bytes(
                        data=Path(path).read_bytes(), mime_type="image/png"
                    )
                )

            text_prompt = f"Following are images containing results of semantic segmentation \
                            for the object '{obj}'. From the provided images, choose the one \
                            that best represents the object. The zero-th image depicts the whole scene and \
                            the images after that depict the segmented objects. If none of the images are correct, \
                            choose the image that is closest to the object. Some instructions that \
                            you must follow are: \
                            1) Choose the image that contains the entire object. For instance, \
                            for the object 'oven', choose the image that contains the entire \
                            oven. For the object 'black bowl', choose the image that contains \
                            the entire black bowl. For the object 'stove burner', choose the image \
                            that contains the entire stove burner. \
                            2) When dealing with objects like drawer handle or door handle,\
                            choose the image containing the whole drawer or door instead of just the handle.\
                            3) The images your are provided with are the results of semantic segmentation. \
                            The background is colored in black and the object is colored. \
                            Take only the foreground object into consideration while making your decision and \
                            ignore the black background regions. \
                            \
                            The images are provided in order [0, 1, 2, 3, ...]. The 0-th image is the whole scene so do not \
                            return 0 as your answer. Give your answer \
                            in the format '<image>number</image>'. If all images are identical, \
                            then choose the image number 1 and return in the same '<image>number</image>' format."
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[text_prompt, *b64_images],
                config={
                    "temperature": 0.0,  # Adjust as needed
                    "top_p": 0.95,
                },
            )
            # save response in txt file
            with open(task_save_path / f"verification_response_{obj}.txt", "w") as f:
                f.write(response.text)

            # extract image number and save image
            image_num = extract_image_number(response.text)

        else:
            image_num = 1

        Image.fromarray(masked_images[image_num - 1]).save(
            task_save_path / f"final_image_{obj}.png"
        )
        MASKS[obj] = masks[image_num - 1]
        MASKED_IMAGES[obj] = masked_images[image_num - 1]

    # return data - only include objects that have valid masks
    valid_objects = list(MASKS.keys())
    data = {
        "objects": valid_objects,
        "masks": MASKS,
        "masked_images": MASKED_IMAGES,
    }
    return data


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

            src_vec = desc1[h, w][None, None].clone()
            trg_vec = desc2

            src_vec /= src_vec.norm(dim=-1, keepdim=True)
            trg_vec /= trg_vec.norm(dim=-1, keepdim=True)
            cos_map = (src_vec * trg_vec).sum(dim=-1).cpu().numpy()

            max_hw = np.unravel_index(cos_map.argmax(), cos_map.shape)
            out_coords[idx, 1], out_coords[idx, 0] = int(max_hw[0]), int(max_hw[1])
        points.append(out_coords)
    return points
