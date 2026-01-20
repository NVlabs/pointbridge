# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import einops
import numpy as np
from collections import deque

import torch
from torch import nn

from torchvision import transforms as T

from point_bridge import utils
from point_bridge.agent.networks.rgb_modules import ResnetEncoder
from point_bridge.agent.networks.policy_head import (
    DiffusionHead,
    DeterministicHead,
)
from point_bridge.agent.networks.gpt import GPT, GPTConfig
from point_bridge.agent.networks.mlp import MLP
from point_bridge.agent.networks.dp3_encoder import PointNetEncoderXYZ


class Actor(nn.Module):
    """
    Point-Bridge actor network that predicts actions from encoded observations.
    
    Architecture:
        Encoded features → Insert action tokens → GPT Transformer → Action Head → Actions
        
    The actor uses a GPT-style transformer to process temporal sequences of observation
    features and predict actions. Action tokens are inserted at regular intervals to
    mark where actions should be predicted (similar to BERT's [CLS] token).
    
    Supports two policy heads:
        - Deterministic: Direct action prediction with MSE loss
        - Diffusion: Diffusion-based action generation for multimodal tasks
        
    Action chunking: Can predict multiple future actions (num_queries) at once
    for smoother execution via temporal aggregation.
    """
    
    def __init__(
        self,
        repr_dim,
        act_dim,
        hidden_dim,
        policy_head="deterministic",
        num_feat_per_step=1,
        num_queries=1,
        action_mode="pose",
        num_robot_points=9,
        device="cuda",
    ):
        """
        Initialize the Actor network.
        
        Args:
            repr_dim (int): Dimension of encoded observation features (default: 512)
            act_dim (int): Dimension of action space (10 for pose mode: pos(3) + rot_6d(6) + gripper(1))
            hidden_dim (int): Hidden dimension for transformer and action head (default: 512)
            policy_head (str): Type of action head ("deterministic" or "diffusion")
            num_feat_per_step (int): Number of features per timestep (e.g., 3 for robot + object + proprio)
            num_queries (int): Number of future actions to predict (action chunking)
            action_mode (str): Action representation ("pose" or "points")
            num_robot_points (int): Number of robot keypoints
            device (str): Device for computation
        """
        super().__init__()

        self._policy_head = policy_head
        self._repr_dim = repr_dim
        self._act_dim = act_dim
        self._num_feat_per_step = num_feat_per_step
        self._action_mode = action_mode
        self._num_robot_points = num_robot_points

        # For action chunking: predict multiple future actions
        self._num_queries = num_queries

        # Learnable action token inserted to mark action prediction locations
        self._action_token = nn.Parameter(torch.randn(1, 1, 1, repr_dim))

        # GPT model
        self._policy = GPT(
            GPTConfig(
                block_size=65,
                input_dim=repr_dim,
                output_dim=hidden_dim,
                n_layer=8,
                n_head=4,
                n_embd=hidden_dim,
                dropout=0.1,
                causal=True,
            )
        )

        if policy_head == "diffusion":
            act_dim = self._act_dim // self._num_queries
            assert act_dim * self._num_queries == self._act_dim
            self._pred_horizon = self._num_queries
            self._action_head = DiffusionHead(
                input_size=hidden_dim,
                output_size=act_dim,
                obs_horizon=1,
                pred_horizon=self._pred_horizon,
                hidden_size=hidden_dim,
                num_layers=2,
                device=device,
                loss_coef=10.0,
                num_diffusion_iters=100,
                num_inference_iters=100,
            )

        elif policy_head == "deterministic":
            self._action_head = DeterministicHead(
                input_size=hidden_dim,
                output_size=self._act_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                num_heads=4,
                max_seq_len=100,
                dropout=0.1,
                loss_coef=1.0,
                sep_gripper_loss=False,
                num_queries=self._num_queries,
            )

        self.apply(utils.weight_init)

    def forward(self, obs, num_prompt_feats, stddev, action=None, **kwargs):
        """
        Forward pass to predict actions from encoded observations.
        
        Args:
            obs (torch.Tensor): Encoded observation features, shape (B, T, D)
                where B = batch size, T = sequence length, D = repr_dim
            num_prompt_feats (int): Number of prompt features (e.g., language tokens) at start of sequence
            stddev (float): Standard deviation for action noise (used in stochastic policies)
            action (torch.Tensor, optional): Ground-truth actions for training, shape (B, T, act_dim)
            
        Returns:
            If action is None (inference):
                torch.Tensor: Predicted actions, shape (B, T, num_queries * act_dim)
            If action is provided (training):
                tuple: (predicted_actions, [loss_dict])
                
        Process:
            1. Separate prompt features (e.g., language) from observation features
            2. Insert action tokens at regular intervals to mark prediction locations
            3. Process sequence through GPT transformer
            4. Extract features at action token positions
            5. Pass through action head to get action predictions
        """
        B, T, D = obs.shape

        # Separate prompt features (e.g., language) from observation features
        prompt, obs = obs[:, :num_prompt_feats], obs[:, num_prompt_feats:]

        # Insert action token at each self._num_feat_per_step interval
        # This creates: [feat1, feat2, feat3, action_token, feat1, feat2, feat3, action_token, ...]
        obs = obs.view(B, -1, self._num_feat_per_step, obs.shape[-1])
        action_token = self._action_token.repeat(B, obs.shape[1], 1, 1)
        obs = torch.cat([obs, action_token], dim=-2).view(B, -1, D)

        # Prepend prompt features back
        obs = torch.cat([prompt, obs], dim=1)

        # Process through GPT transformer
        features = self._policy(obs)
        features = features[:, num_prompt_feats:]  # Remove prompt features

        # Extract features at action token positions
        num_feat_per_step = self._num_feat_per_step + 1  # +1 for action token
        features = features[:, num_feat_per_step - 1 :: num_feat_per_step]

        if self._policy_head == "diffusion":
            B, T, D = features.shape
            features = einops.rearrange(features, "B T D -> (B T) D")[:, None]
            if action is not None:
                action = einops.rearrange(
                    action, "B T (Q D) -> B T Q D", Q=self._num_queries
                )
                action = einops.rearrange(action, "B T Q D -> (B T) Q D")

        # action head
        pred_action = self._action_head(
            features,
            stddev,
            **{"action_seq": action},
        )

        if action is None:
            if self._policy_head == "diffusion":
                pred_action = einops.rearrange(pred_action, "(B T) Q D -> B T Q D", B=B)
                pred_action = einops.rearrange(pred_action, "B T Q D -> B T (Q D)")

            return pred_action
        else:
            loss = self._action_head.loss_fn(
                pred_action,
                action,
                reduction="mean",
            )
            if isinstance(loss, tuple):
                loss = loss[0]
            losses = [loss]

            return pred_action, losses


class BCAgent:
    """
    Point-Bridge behavior cloning agent.
    
    This is the main agent class that combines:
        - Point cloud encoders (PointNet) for robot and object observations
        - Proprioception encoder (MLP) for end-effector state
        - Language encoder (MLP) for task conditioning
        - Actor network (GPT + action head) for action prediction
        
    The agent learns manipulation policies via behavior cloning on demonstration
    data, using 3D point representations for cross-domain transfer.
    
    Key features:
        - Supports both point-based and image-based observations
        - Action chunking for temporal consistency
        - Language conditioning for multi-task learning
        - Normalization of observations and actions
        
    Training flow:
        1. Encode observations (points, proprio, language) to feature vectors
        2. Stack features temporally (history_len timesteps)
        3. Process through Actor to predict actions
        4. Compute MSE loss against expert actions
        5. Update encoders and actor via gradient descent
        
    Evaluation flow:
        1. Buffer observations over eval_history_len timesteps
        2. Encode and predict actions
        3. Apply temporal aggregation if action chunking is enabled
        4. Denormalize and execute actions
    """
    
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        stddev_schedule,
        use_tb,
        policy_head,
        use_language,
        pixel_keys,
        proprio_key,
        use_proprio,
        history_len,
        eval_history_len,
        action_chunking,
        num_queries,
        temporal_agg_strategy,
        max_episode_len,
        film,
        obs_type,
        action_mode,
        robot_points_key,
        object_points_key,
        num_points_per_obj,
    ):
        self.device = device
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.stddev_schedule = stddev_schedule
        self.use_tb = use_tb
        self.policy_head = policy_head
        self.use_language = use_language
        self.use_proprio = use_proprio
        self.history_len = history_len
        self.eval_history_len = eval_history_len
        self.film = film
        self.num_robot_points = 8 + 1  # +1 for gripper

        # obs params
        self.obs_type = obs_type
        self.action_mode = action_mode
        self.robot_points_key = robot_points_key
        self.object_points_key = f"{object_points_key}_{num_points_per_obj}"
        self.robot_points_key = f"{self.robot_points_key}_3d"
        self.object_points_key = f"{self.object_points_key}_3d"

        # actor parameters
        self._act_dim = (
            action_shape[0]
            if action_mode == "pose"
            else 3 * self.num_robot_points  # since 3D points
        )

        # language
        self.language_fusion = "none" if not self.use_language else "film"
        self.language_dim = 384
        self.repr_dim = 512  # Keep original value to match the saved model

        # keys
        self.pixel_keys = pixel_keys
        self.proprio_key = proprio_key

        # action chunking params
        self.action_chunking = action_chunking
        self.max_episode_len = max_episode_len
        self.num_queries = num_queries if self.action_chunking else 1
        self.temporal_agg_strategy = temporal_agg_strategy

        # number of inputs per time step
        num_feat_per_step = 0
        if "image" in self.obs_type:
            num_feat_per_step = len(self.pixel_keys)
        if self.use_proprio:
            num_feat_per_step += 1
        if "points" in self.obs_type:
            num_feat_per_step += 1  # for robot
            num_feat_per_step += 1  # for pcd

        # observation params
        if self.use_proprio:
            proprio_shape = obs_shape[self.proprio_key]
        obs_shape = obs_shape[self.pixel_keys[0]]

        # Track model size
        model_size = 0

        # encoder
        if "image" in self.obs_type:
            self.encoder = ResnetEncoder(
                obs_shape,
                self.repr_dim,
                language_dim=self.repr_dim,
                language_fusion=self.language_fusion,
            ).to(device)
            model_size += sum(
                p.numel() for p in self.encoder.parameters() if p.requires_grad
            )
        elif "points" in self.obs_type:
            self.encoder = PointNetEncoderXYZ(
                in_channels=3,  # since 3D points
                out_channels=self.repr_dim,
                use_layernorm=True,
                obj_features_dim=self.language_dim,
            ).to(device)
            self.encoder.apply(utils.weight_init)

            model_size += sum(
                p.numel() for p in self.encoder.parameters() if p.requires_grad
            )

        # projector for proprioceptive features
        if self.use_proprio:
            self.proprio_projector = MLP(
                proprio_shape[0], hidden_channels=[self.repr_dim, self.repr_dim]
            ).to(device)
            self.proprio_projector.apply(utils.weight_init)
            model_size += sum(
                p.numel()
                for p in self.proprio_projector.parameters()
                if p.requires_grad
            )

        # language encoder
        if self.use_language:
            self.language_projector = MLP(
                self.language_dim,
                hidden_channels=[self.repr_dim, self.repr_dim],
            ).to(device)
            self.language_projector.apply(utils.weight_init)
            model_size += sum(
                p.numel()
                for p in self.language_projector.parameters()
                if p.requires_grad
            )

        # actor
        action_dim = (
            self._act_dim * self.num_queries if self.action_chunking else self._act_dim
        )
        self.actor = Actor(
            self.repr_dim,
            action_dim,
            hidden_dim,
            self.policy_head,
            num_feat_per_step,
            self.num_queries,
            self.action_mode,
            self.num_robot_points,
            device,
        ).to(device)
        model_size += sum(p.numel() for p in self.actor.parameters() if p.requires_grad)

        print("Total parameter count: %.2fM" % (model_size / 1e6,))

        # optimizers
        # encoder
        params = list(self.encoder.parameters())
        self.encoder_opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        # proprio
        if self.use_proprio:
            self.proprio_opt = torch.optim.AdamW(
                self.proprio_projector.parameters(), lr=lr, weight_decay=1e-4
            )
        # language
        if self.use_language:
            self.language_opt = torch.optim.AdamW(
                self.language_projector.parameters(), lr=lr, weight_decay=1e-4
            )
        # actor
        actor_params = list(self.actor.parameters())
        self.actor_opt = torch.optim.AdamW(actor_params, lr=lr, weight_decay=1e-4)

        # scaling for images at inference
        self.test_aug = T.Compose([T.ToPILImage(), T.ToTensor()])

        self.train()
        self.buffer_reset()

    def __repr__(self):
        return "bc"

    def train(self, training=True):
        self.training = training
        if training:
            self.encoder.train(training)
            if self.use_proprio:
                self.proprio_projector.train(training)
            if self.use_language:
                self.language_projector.train(training)
            self.actor.train(training)
        else:
            self.encoder.eval()
            if self.use_proprio:
                self.proprio_projector.eval()
            if self.use_language:
                self.language_projector.eval()
            self.actor.eval()

    def buffer_reset(self):
        self.observation_buffer = {}
        for key in self.pixel_keys:
            if "image" in self.obs_type:
                self.observation_buffer[key] = deque(maxlen=self.eval_history_len)
        if "points" in self.obs_type:
            self.observation_buffer[self.robot_points_key] = deque(
                maxlen=self.eval_history_len
            )
            self.observation_buffer[self.object_points_key] = deque(
                maxlen=self.eval_history_len
            )
        if self.use_proprio:
            self.proprio_buffer = deque(maxlen=self.eval_history_len)

        # temporal aggregation
        if self.action_chunking:
            if self.temporal_agg_strategy == "exponential_average":
                self.all_time_actions = torch.zeros(
                    [
                        self.max_episode_len,
                        self.max_episode_len + self.num_queries,
                        self._act_dim,
                    ]
                ).to(self.device)

    def clear_buffers(self):
        del self.observation_buffer
        if self.use_proprio:
            del self.proprio_buffer
        if self.action_chunking:
            if self.temporal_agg_strategy == "exponential_average":
                del self.all_time_actions

    def act(self, obs, norm_stats, step, global_step, **kwargs):
        """
        Predict action from current observation (used during evaluation/deployment).
        
        Args:
            obs (dict): Dictionary of observations containing:
                - robot_points_3d: Robot keypoint positions
                - object_points_N_3d: Object point clouds  
                - proprio: End-effector state (optional)
                - task_emb: Language embedding (optional)
            norm_stats (dict): Normalization statistics from training
            step (int): Current timestep in episode
            global_step (int): Global training step (for stddev schedule)
            
        Returns:
            np.ndarray: Action to execute, shape (act_dim,)
                For pose mode: (10,) = pos(3) + rot_6d(6) + gripper(1)
                
        Process:
            1. Normalize observations using training statistics
            2. Buffer observations over eval_history_len timesteps
            3. Encode observations to features
            4. Predict actions using Actor
            5. Apply temporal aggregation (if action chunking enabled)
            6. Denormalize actions
            
        Note:
            - Maintains observation buffers between calls (reset with buffer_reset())
            - First calls may have incomplete history (padded with zeros)
            - Action chunking uses exponential averaging of predictions from multiple timesteps
        """
        if norm_stats is not None:
            pre_process = {
                self.proprio_key: lambda s_qpos: (
                    s_qpos - norm_stats[self.proprio_key]["min"]
                )
                / (
                    norm_stats[self.proprio_key]["max"]
                    - norm_stats[self.proprio_key]["min"]
                    + 1e-5
                ),
                "depth": lambda d: (d - norm_stats["depth"]["min"])
                / (norm_stats["depth"]["max"] - norm_stats["depth"]["min"] + 1e-5),
                "past_tracks": lambda x: (x - norm_stats["past_tracks"]["min"])
                / (
                    norm_stats["past_tracks"]["max"]
                    - norm_stats["past_tracks"]["min"]
                    + 1e-5
                ),
            }
            post_process = {
                self.proprio_key: lambda s: s
                * (
                    norm_stats[self.proprio_key]["max"]
                    - norm_stats[self.proprio_key]["min"]
                )
                + norm_stats[self.proprio_key]["min"],
            }
            if self.action_mode == "pose":
                post_process["actions"] = (
                    lambda a: a
                    * (norm_stats["actions"]["max"] - norm_stats["actions"]["min"])
                    + norm_stats["actions"]["min"]
                )
            elif self.action_mode == "points":
                max_act, min_act = (
                    norm_stats["future_tracks"]["max"],
                    norm_stats["future_tracks"]["min"],
                )
                max_gripper, min_gripper = (
                    norm_stats["gripper_states"]["max"],
                    norm_stats["gripper_states"]["min"],
                )
                max_gripper = np.array([max_gripper] * 3)
                min_gripper = np.array([min_gripper] * 3)
                max_act = np.concatenate([max_act, max_gripper], axis=-1)
                min_act = np.concatenate([min_act, min_gripper], axis=-1)
                post_process["actions"] = lambda x: x * (max_act - min_act) + min_act

        # lang projection
        if self.use_language:
            if "image" in self.obs_type:
                key = self.pixel_keys[0]
            elif "points" in self.obs_type:
                key = self.object_points_key
            repeat_len = min(
                len(self.observation_buffer[key]) + 1, self.eval_history_len
            )
            lang_features = (
                torch.as_tensor(obs["task_emb"], device=self.device)
                .float()[None]
                .repeat(repeat_len, 1)
            )
            lang_features = self.language_projector(lang_features)
        else:
            lang_features = None

        # add to buffer
        features = []
        if "image" in self.obs_type:
            for key in self.pixel_keys:
                self.observation_buffer[key].append(
                    self.test_aug(obs[key].transpose(1, 2, 0)).numpy()
                )
                pixels = torch.as_tensor(
                    np.array(self.observation_buffer[key]), device=self.device
                ).float()
                # encoder
                lang = lang_features if self.film and self.use_language else None
                pixels = self.encoder(pixels, lang=lang)
                features.append(pixels)

        if "points" in self.obs_type:
            # robot points
            past_robot_tracks = pre_process["past_tracks"](obs[self.robot_points_key])
            self.observation_buffer[self.robot_points_key].append(past_robot_tracks)
            past_robot_tracks = torch.as_tensor(
                np.array(self.observation_buffer[self.robot_points_key]),
                device=self.device,
            ).float()
            past_robot_tracks = self.encoder(past_robot_tracks)  # (t, d)
            features.append(past_robot_tracks)

            # object points
            past_object_tracks = pre_process["past_tracks"](obs[self.object_points_key])
            self.observation_buffer[self.object_points_key].append(past_object_tracks)
            past_object_tracks = torch.as_tensor(
                np.array(self.observation_buffer[self.object_points_key]),
                device=self.device,
            ).float()
            past_object_tracks = einops.rearrange(
                past_object_tracks, "t n p d -> t (n p) d"
            )
            past_object_tracks = self.encoder(past_object_tracks)  # (t, d)
            features.append(past_object_tracks)

        if self.use_proprio:
            obs[self.proprio_key] = pre_process[self.proprio_key](obs[self.proprio_key])
            self.proprio_buffer.append(obs[self.proprio_key])
            proprio = torch.as_tensor(
                np.array(self.proprio_buffer), device=self.device
            ).float()
            proprio = self.proprio_projector(proprio)
            features.append(proprio)

        features = torch.cat(features, dim=-1).view(-1, self.repr_dim)

        if self.use_language:
            features = torch.cat([lang_features[-1:], features], dim=0)
            num_prompt_feats = 1
        else:
            num_prompt_feats = 0

        # For deterministic head, reshape features to (B, T, D) format
        if self.policy_head == "deterministic":
            # Reshape features to (batch_size, num_steps, feature_dim)
            features = features.view(1, -1, self.repr_dim)  # (1, T, D)

        stddev = utils.schedule(self.stddev_schedule, global_step)
        action = self.actor(
            features.unsqueeze(0) if self.policy_head != "deterministic" else features,
            num_prompt_feats,
            stddev,
        )

        if self.policy_head == "deterministic":
            action = action.mean
        elif self.policy_head == "diffusion":
            action = action["action_pred"] if isinstance(action, dict) else action

        # absolute actions
        if self.action_chunking and self.temporal_agg_strategy == "exponential_average":
            action = action.view(-1, self.num_queries, self._act_dim)
            self.all_time_actions[[step], step : step + self.num_queries] = action[-1:]
            actions_for_curr_step = self.all_time_actions[:, step]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.005
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            if norm_stats is not None:
                return post_process["actions"](action.cpu().numpy()[0])
            return action.cpu().numpy()[0]
        else:
            if norm_stats is not None:
                return post_process["actions"](action.cpu().numpy()[0, -1])
            return action.cpu().numpy()[0, -1, :]

    def update(self, expert_replay_iter, step):
        """
        Update agent parameters via behavior cloning on a batch of demonstrations.
        
        Args:
            expert_replay_iter: Iterator over demonstration batches
            step (int): Current training step
            
        Returns:
            dict: Training metrics including "actor_loss"
            
        Process:
            1. Sample a batch from expert demonstrations
            2. Encode observations (points, proprio, language) to features
            3. Forward pass through Actor to predict actions
            4. Compute MSE loss between predicted and expert actions
            5. Backpropagate and update all parameters
            
        Updates:
            - Point cloud encoder (PointNet)
            - Proprioception encoder (MLP)
            - Language encoder (MLP) if using language
            - Actor (GPT + action head)
            
        Note:
            - All observations and actions are pre-normalized in the dataset
            - For diffusion head, also updates EMA model every 10 steps
        """
        metrics = dict()

        batch = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)

        if self.action_mode == "pose":
            action = data["actions"].float()
        elif self.action_mode == "points":
            future_tracks = data["future_tracks"].float()  # (b, t, p, q*d)
            future_gripper_states = data["future_gripper_states"].float()  # (b, t, q)

            future_gripper_states = future_gripper_states[..., None].repeat(1, 1, 1, 3)
            action = torch.cat([future_tracks, future_gripper_states], dim=-1)

        # lang projection
        if self.use_language:
            lang_features = (
                data["task_emb"].float()[:, None].repeat(1, self.history_len, 1)
            )
            lang_features = self.language_projector(lang_features)
            lang_features = einops.rearrange(lang_features, "b t d -> (b t) d")
        else:
            lang_features = None

        # features
        features = []
        for key in self.pixel_keys:
            if "image" in self.obs_type:
                pixel = data[key].float()
                shape = pixel.shape
                # rearrange
                pixel = einops.rearrange(pixel, "b t c h w -> (b t) c h w")
                # encode
                lang = lang_features if self.film and self.use_language else None
                pixel = self.encoder(pixel, lang=lang)
                pixel = einops.rearrange(pixel, "(b t) d -> b t d", t=shape[1])
                features.append(pixel)

        if "points" in self.obs_type:
            past_robot_tracks = data["past_robot_tracks"].float()  # (b, t, p, d)
            past_robot_tracks = einops.rearrange(
                past_robot_tracks, "b t p d -> (b t) p d"
            )
            past_robot_tracks = self.encoder(past_robot_tracks)
            past_robot_tracks = einops.rearrange(
                past_robot_tracks, "(b t) d -> b t d", t=self.history_len
            )
            features.append(past_robot_tracks)

            past_object_tracks = data["past_object_tracks"].float()  # (b, t, n, p, d)
            if len(past_object_tracks.shape) == 5:
                past_object_tracks = einops.rearrange(
                    past_object_tracks, "b t n p d -> (b t) (n p) d"
                )
            elif len(past_object_tracks.shape) == 4:
                # TODO: this doesn't happen so remove
                past_object_tracks = einops.rearrange(
                    past_object_tracks, "b t p d -> (b t) p d"
                )
            past_object_tracks = self.encoder(past_object_tracks)  # (b*t, d)
            past_object_tracks = einops.rearrange(
                past_object_tracks, "(b t) d -> b t d", t=self.history_len
            )
            features.append(past_object_tracks)

        if self.use_proprio:
            proprio = data[self.proprio_key].float()
            proprio = self.proprio_projector(proprio)
            features.append(proprio)

        # concatenate
        features = torch.cat(features, dim=-1).view(
            action.shape[0], -1, self.repr_dim
        )  # (B, T * num_feat_per_step, D)

        if self.use_language:
            lang_features = einops.rearrange(
                lang_features, "(b t) d -> b t d", t=self.history_len
            )
            lang_features = lang_features[:, -1:]
            features = torch.cat([lang_features, features], dim=1)
            num_prompt_feats = 1
        else:
            num_prompt_feats = 0

        # rearrange action
        if self.action_chunking:
            action = einops.rearrange(action, "b t1 t2 d -> b t1 (t2 d)")

        # actor loss
        stddev = utils.schedule(self.stddev_schedule, step)
        _, actor_losses = self.actor(features, num_prompt_feats, stddev, action)
        loss = actor_losses[0]["actor_loss"]
        for loss_item in actor_losses[1:]:
            loss += loss_item["actor_loss"]

        # optimizer step
        self.encoder_opt.zero_grad(set_to_none=True)
        if self.use_proprio:
            self.proprio_opt.zero_grad(set_to_none=True)
        if self.use_language:
            self.language_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.encoder_opt.step()
        if self.use_proprio:
            self.proprio_opt.step()
        if self.use_language:
            self.language_opt.step()
        self.actor_opt.step()

        if self.policy_head == "diffusion" and step % 10 == 0:
            self.actor._action_head.net.ema_step()

        if self.use_tb:
            for key, value in actor_losses[0].items():
                metrics[key] = value.item()

        return metrics

    def save_snapshot(self):
        model_keys = ["actor", "encoder"]
        opt_keys = ["actor_opt", "encoder_opt"]
        if self.use_proprio:
            model_keys += ["proprio_projector"]
            opt_keys += ["proprio_opt"]
        if self.use_language:
            model_keys += ["language_projector"]
            opt_keys += ["language_opt"]

        # models
        payload = {k: self.__dict__[k].state_dict() for k in model_keys}
        # optimizers
        payload.update({k: self.__dict__[k] for k in opt_keys})

        others = [
            "use_proprio",
            "max_episode_len",
        ]
        payload.update({k: self.__dict__[k] for k in others})
        return payload

    def load_snapshot(self, payload, eval=True):
        # models
        model_keys = ["actor", "encoder"]
        if self.use_proprio:
            model_keys += ["proprio_projector"]
        if self.use_language:
            model_keys += ["language_projector"]
        for k in model_keys:
            self.__dict__[k].load_state_dict(payload[k])

        if eval:
            self.train(False)
        return
