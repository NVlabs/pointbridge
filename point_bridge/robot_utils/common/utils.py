# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import torch
import numpy as np
import open3d as o3d


def pixel2d_to_3d_torch(points2d, depths, intrinsic_matrix, extrinsic_matrix):
    """
    Backproject 2D pixel coordinates to 3D points using depth (PyTorch version).
    
    Args:
        points2d (torch.Tensor): 2D pixel coordinates, shape (N, 2) with (u, v) format
        depths (torch.Tensor): Depth values for each pixel, shape (N,)
        intrinsic_matrix (array-like): 3x3 camera intrinsic matrix
        extrinsic_matrix (array-like): 4x4 transformation from world to camera frame
        
    Returns:
        torch.Tensor: 3D points in world frame, shape (N, 3)
        
    Process:
        1. Convert pixel coordinates to normalized camera coordinates using intrinsics
        2. Multiply by depth to get 3D points in camera frame
        3. Transform to world frame using inverse of extrinsic matrix
    """
    intrinsic_matrix = torch.tensor(intrinsic_matrix).float().to(depths.device)
    extrinsic_matrix = torch.tensor(extrinsic_matrix).float().to(depths.device)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    x = (points2d[:, 0] - cx) / fx
    y = (points2d[:, 1] - cy) / fy
    points3d = torch.stack((x * depths, y * depths, depths), dim=1)  # in camera frame
    points3d = torch.cat(
        (points3d, torch.ones((len(points2d), 1)).to(depths.device)), dim=1
    )
    points3d = (torch.linalg.inv(extrinsic_matrix) @ points3d.T).T  # world frame
    return points3d[..., :3]


def pixel2d_to_3d(points2d, depths, intrinsic_matrix, extrinsic_matrix):
    """
    Backproject 2D pixel coordinates to 3D points using depth (NumPy version).
    
    Args:
        points2d (array-like): 2D pixel coordinates, shape (N, 2) in (u, v) format
        depths (array-like): Depth values for each pixel, shape (N,)
        intrinsic_matrix (np.ndarray): 3x3 camera intrinsic matrix containing:
            - fx, fy: focal lengths
            - cx, cy: principal point
        extrinsic_matrix (np.ndarray): 4x4 transformation from world to camera frame
        
    Returns:
        np.ndarray: 3D points in world frame, shape (N, 3)
        
    Note:
        This is a fundamental operation in Point-Bridge for converting VLM-detected
        pixels with stereo depth to 3D points in robot base frame.
        
    Example:
        # Get 3D position of detected object center
        pixel = np.array([[320, 240]])  # Image center
        depth = np.array([1.5])  # 1.5 meters
        point_3d = pixel2d_to_3d(pixel, depth, K, T_world_to_cam)
    """
    points2d = np.array(points2d)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    x = (points2d[:, 0] - cx) / fx
    y = (points2d[:, 1] - cy) / fy
    points_3d = np.column_stack((x * depths, y * depths, depths))  # in camera frame
    points_3d = np.concatenate([points_3d, np.ones((len(points2d), 1))], axis=1)
    points_3d = (np.linalg.inv(extrinsic_matrix) @ points_3d.T).T  # world frame
    return points_3d[..., :3]


def pixel3d_to_2d(points3d, intrinsic_matrix, extrinsic_matrix):
    """
    Project 3D points to 2D pixel coordinates.
    
    Args:
        points3d (array-like): 3D points in world frame, shape (N, 3)
        intrinsic_matrix (np.ndarray): 3x3 camera intrinsic matrix
        extrinsic_matrix (np.ndarray): 4x4 transformation from world to camera frame
        
    Returns:
        tuple: (points2d, depths)
            - points2d (np.ndarray): 2D pixel coordinates, shape (N, 2)
            - depths (np.ndarray): Depth values for each point, shape (N,)
            
    Inverse of pixel2d_to_3d. Useful for:
        - Visualizing 3D points on image
        - Checking if 3D points are visible in camera
    """
    points3d = np.array(points3d)
    points3d = np.concatenate([points3d, np.ones((len(points3d), 1))], axis=1)
    points3d = (extrinsic_matrix @ points3d.T).T  # camera frame
    depth = points3d[:, 2]
    points2d = (intrinsic_matrix @ points3d.T).T
    points2d = points2d / points2d[:, 2][:, None]
    return points2d[..., :2], depth


def triangulate_points(P, points):
    """
    Triangulate a batch of points from a variable number of camera views.

    Parameters:
    P: list of 3x4 projection matrices for each camera (currently world2camera transform)
    points: list of Nx2 arrays of normalized image coordinates for each camera

    Returns:
    Nx4 array of homogeneous 3D points
    """
    num_views = len(P)
    assert num_views > 1, "At least 2 cameras are required for triangulation"

    num_points = points[0].shape[0]
    A = np.zeros((num_points, num_views * 2, 4))

    for idx in range(num_views):
        # Set up the linear system for each point
        A[:, idx * 2] = points[idx][:, 0, np.newaxis] * P[idx][2] - P[idx][0]
        A[:, idx * 2 + 1] = points[idx][:, 1, np.newaxis] * P[idx][2] - P[idx][1]

    # Solve the system using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1, :]

    # Normalize the homogeneous coordinates
    X = X / X[:, 3:]

    return X


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_cols != 3:
        raise Exception(f"matrix A is not Nx3, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_cols != 3:
        raise Exception(f"matrix B is not Nx3, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.T @ Bm

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A.T + centroid_B.T

    return R, t


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation to rotation matrix
    using Gram-Schmidt orthogonalization.

    Args:
        d6: 6D rotation representation, of shape (..., 6)

    Returns:
        Batch of rotation matrices of shape (..., 3, 3)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]

    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """
    Converts rotation matrices to 6D rotation representation
    by dropping the last row.

    Args:
        matrix: Batch of rotation matrices of shape (..., 3, 3)

    Returns:
        6D rotation representation, of shape (..., 6)
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))


def farthest_point_sampling(points, num_points):
    """
    Downsample point cloud using Farthest Point Sampling (FPS).
    
    Args:
        points (np.ndarray): Input point cloud, shape (N, 3)
        num_points (int): Desired number of points
        
    Returns:
        np.ndarray: Downsampled point cloud, shape (num_points, 3)
        
    FPS iteratively selects points that are farthest from already selected points,
    resulting in a more uniform distribution than random sampling.
    
    This is used in Point-Bridge to:
        - Downsample VLM-extracted object points to consistent size (128 points)
        - Reduce full scene point clouds for faster processing
        
    Note: Uses Open3D implementation for efficiency.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.farthest_point_down_sample(num_points)
    return np.asarray(pcd.points)


def transform_points(points, pose_matrix):
    """
    Transform points from frame X to frame Y using a pose matrix.

    Args:
        points: Points in frame X. Array of shape (N, 3) containing 3D points in frame X
        pose_matrix: Pose X in frame Y. Array of shape (4, 4) representing the transformation matrix from frame X to frame Y

    Returns:
        Points in frame Y. Array of shape (N, 3) containing the transformed points in frame Y
    """
    points = np.array(points)
    pose_matrix = np.array(pose_matrix)

    # Ensure inputs have correct shapes
    if points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {points.shape}")
    if pose_matrix.shape != (4, 4):
        raise ValueError(f"Pose matrix must have shape (4, 4), got {pose_matrix.shape}")

    # Convert points to homogeneous coordinates
    points_homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)

    # Apply transformation
    transformed_points_homogeneous = (pose_matrix @ points_homogeneous.T).T

    # Convert back to 3D coordinates
    transformed_points = transformed_points_homogeneous[:, :3]

    return transformed_points


def transform_poses(poses, pose_matrix):
    """
    Transform poses from frame X to frame Y using a pose matrix.

    Args:
        poses: Poses in frame X. Array of shape (N, 4, 4).
        pose_matrix: Pose X in frame Y. Array of shape (4, 4) representing the transformation matrix from frame X to frame Y

    Returns:
        Poses in frame Y. Array of shape (N, 4, 4).
    """

    poses = np.array(poses)
    pose_matrix = np.array(pose_matrix)

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Poses must have shape (N, 4, 4), got {poses.shape}")
    if pose_matrix.shape != (4, 4):
        raise ValueError(f"Pose matrix must have shape (4, 4), got {pose_matrix.shape}")

    # Compose each pose with the transformation
    transformed_poses = np.matmul(pose_matrix[None, ...], poses)
    return transformed_poses


def depthimg2Meters(env, depth):
    extent = env.sim.model.stat.extent
    near = env.sim.model.vis.map.znear * extent
    far = env.sim.model.vis.map.zfar * extent
    image = near / (1 - depth * (1 - near / far))
    return image
