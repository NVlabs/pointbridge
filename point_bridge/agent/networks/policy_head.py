# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import einops
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from point_bridge import utils

# from agent.networks.utils.diffusion_policy import DiffusionPolicy


######################################### Diffusion Head #########################################


class DiffusionHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        obs_horizon,  # history len
        pred_horizon,
        hidden_size=1024,
        num_layers=2,
        device="cpu",
        loss_coef=100.0,
        policy_type="transformer",
        noise_scheduler="ddpm",
        num_diffusion_iters=100,
        num_inference_iters=100,
    ):
        super().__init__()

        self.net = DiffusionPolicy(
            obs_dim=input_size,
            act_dim=output_size,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            policy_type=policy_type,
            device=device,
            noise_scheduler=noise_scheduler,
            num_diffusion_iters=num_diffusion_iters,
            num_inference_iters=num_inference_iters,
        )

        self.loss_coef = loss_coef

    def forward(self, x, stddev=None, **kwargs):
        return self.net(x, kwargs.get("action_seq", None))

    def loss_fn(self, out, target, action_mask=None, reduction="mean", **kwargs):
        noise_pred = out["noise_pred"]
        noise = out["noise"]
        if action_mask is not None:
            noise_pred *= action_mask
            noise *= action_mask

        return {
            "actor_loss": F.mse_loss(noise_pred, noise, reduction=reduction)
            * self.loss_coef,
        }


######################################### Deterministic Head #########################################


class DeterministicHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        num_heads=4,
        max_seq_len=100,
        dropout=0.1,
        loss_coef=1.0,
        sep_gripper_loss=False,
        num_queries=1,
    ):
        super().__init__()
        self.loss_coef = loss_coef
        self.sep_gripper_loss = sep_gripper_loss
        self.max_seq_len = max_seq_len
        self.num_queries = num_queries
        self.action_dim = output_size // num_queries

        self.pos_embed = nn.Embedding(self.max_seq_len, hidden_size)

        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # cross attention through transformer decoder layers
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True
            ),
            num_layers=num_layers,
        )

        # output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, self.action_dim),
        )

        self.apply(utils.weight_init)

    def forward(self, x, stddev=None, ret_action_value=False, **kwargs):
        B, T, _ = x.shape

        # apply positional encoding
        pos_emb = self.pos_embed(torch.arange(self.num_queries, device=x.device))
        pos_emb = pos_emb[None, None].repeat(B, T, 1, 1)

        B, T, Q, D = pos_emb.shape
        pos_emb = einops.rearrange(pos_emb, "B T Q D -> (B T) Q D")
        x = einops.rearrange(x, "B T D -> (B T) D")[..., None, :]

        # pass through causal transformer decoder
        # Pass action through a causal transformer decoder with causal mask
        # The transformer expects (tgt, memory), where tgt is the sequence to decode (action), and memory is the context (x)
        # We'll use a causal mask so each position can only attend to previous positions

        # Create causal mask for the action sequence (shape: (Q, Q))
        causal_mask = torch.triu(
            torch.ones(Q, Q, device=pos_emb.device), diagonal=1
        ).bool()
        # Transformer expects mask shape (tgt_len, tgt_len)

        # x is the context (memory), shape: (B*T, 1, D)
        # Pass action as tgt, x as memory
        # Since transformer is batch_first=True, do not transpose
        decoded = self.transformer_decoder(
            tgt=pos_emb, memory=x, tgt_mask=causal_mask  # (B*T, Q, D)  # (B*T, 1, D)
        )
        decoded = einops.rearrange(decoded, "(B T) Q D -> B T Q D", B=B, T=T)

        pred_action = self.output_proj(decoded)
        pred_action = einops.rearrange(pred_action, "B T Q D -> B T (Q D)")

        mu = pred_action
        std = stddev if stddev is not None else 0.1
        std = torch.ones_like(mu) * std
        dist = utils.Normal(mu, std)
        if ret_action_value:
            return dist.mean
        else:
            return dist

    def loss_fn(self, dist, target, mask=None, reduction="mean", **kwargs):
        log_probs = dist.log_prob(target)

        # mask is of shape (B, N) and log_probs is of shape (B, N, D)
        # We need to extract indices for which the N dimension in mask is 1
        # and then extract the corresponding log_probs
        shape = log_probs.shape
        if mask is not None:
            mask = mask.bool()
            mask = mask.unsqueeze(-1).repeat(1, 1, shape[-1])
        log_probs = log_probs[mask].view(shape[0], -1, shape[-1])
        loss = -log_probs

        if self.sep_gripper_loss:
            # TODO: consider gripper pred is always true
            gripper_loss, track_loss = loss[:, 0], loss[:, 1:]
            if reduction == "mean":
                loss = gripper_loss.mean() + track_loss.mean()
            elif reduction == "none":
                loss = gripper_loss + track_loss
            elif reduction == "sum":
                loss = gripper_loss.sum() + track_loss.sum()
            else:
                raise NotImplementedError
            if not hasattr(self, "count"):
                self.count = 0
            self.count += 1
            if self.count % 100 == 0:
                print(
                    f"gripper_loss: {gripper_loss.mean()}, track_loss: {track_loss.mean()}"
                )
        else:
            if reduction == "mean":
                loss = loss.mean() * self.loss_coef
            elif reduction == "none":
                loss = loss * self.loss_coef
            elif reduction == "sum":
                loss = loss.sum() * self.loss_coef
            else:
                raise NotImplementedError

        return {
            "actor_loss": loss,
        }
