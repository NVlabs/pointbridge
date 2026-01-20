# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
From: https://github.com/YanjieZe/3D-Diffusion-Policy/blob/master/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py

Modified for this codebase.
"""

import torch
import torch.nn as nn
from termcolor import cprint


class PointNetEncoderXYZ(nn.Module):
    """
    PointNet encoder for processing 3D point clouds in Point-Bridge.
    
    This encoder processes unordered point clouds into fixed-size feature vectors
    using the PointNet architecture with LayerNorm for better training stability.
    
    Architecture:
        1. Per-point MLP: 3 → 64 → 128 → 256
        2. Max pooling across points (permutation invariant)
        3. Final projection to output dimension
        
    The key property is permutation invariance: f({p1, p2, p3}) = f({p2, p1, p3})
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        obj_features_dim: int = 384,
        **kwargs,
    ):
        """
        Initialize PointNet encoder.
        
        Args:
            in_channels (int): Input feature dimension per point (default: 3 for xyz)
            out_channels (int): Output feature dimension (default: 1024)
            use_layernorm (bool): Whether to use LayerNorm in MLP layers
            final_norm (str): Type of normalization for final projection ("layernorm" or "none")
            use_projection (bool): Whether to use final projection layer
            obj_features_dim (int): Dimension of object features for conditioning (not used in current implementation)
            
        Note:
            - In Point-Bridge, this processes both robot points and object points
            - Robot points: (num_robot_points, 3) → (repr_dim,)
            - Object points: (num_objects * num_points_per_obj, 3) → (repr_dim,)
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), "cyan")

        self.obj_features_dim = obj_features_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        input_size = block_channel[-1]
        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(input_size, out_channels), nn.LayerNorm(out_channels)
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(input_size, out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)

    def forward(self, x):
        """
        Forward pass through PointNet encoder.
        
        Args:
            x (torch.Tensor): Input point cloud, shape (batch_size, num_points, in_channels)
                or (num_points, in_channels) for single sample
                
        Returns:
            torch.Tensor: Point cloud features, shape (batch_size, out_channels)
                or (out_channels,) for single sample
                
        Process:
            1. Per-point MLP transforms each point independently
            2. Max pooling aggregates across all points (achieves permutation invariance)
            3. Final projection to desired output dimension
            
        Example:
            # Process robot keypoints
            robot_points = torch.randn(8, 3)  # 8 keypoints
            features = encoder(robot_points)  # (512,) feature vector
            
            # Process object points
            object_points = torch.randn(128, 3)  # 128 points per object
            features = encoder(object_points)  # (512,) feature vector
        """
        x = self.mlp(x)  # Apply per-point MLP
        x = torch.max(x, -2)[0]  # Max pool across points
        x = self.final_projection(x)  # Final projection
        return x

    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()

    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()
