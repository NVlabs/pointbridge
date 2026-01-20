# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

#!/usr/bin/env python3

import os
import warnings

warnings.filterwarnings("ignore")

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import numpy as np
from tqdm import tqdm

from point_bridge import utils
from point_bridge.logger import Logger
from point_bridge.replay_buffer import make_expert_replay_loader
from point_bridge.video import VideoRecorder

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    # if cfg.use_proprio:
    obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape

    try:
        cfg.agent.num_object_points = (
            cfg.suite.num_points_per_obj * cfg.suite.task_make_fn.max_num_objects
        )
    except:
        pass

    return hydra.utils.instantiate(cfg.agent)


class Workspace:
    """
    Evaluation workspace for Point-Bridge policies.
    
    This class manages policy evaluation on environments, including:
    - Loading trained checkpoints
    - Creating evaluation environments
    - Running episodes and computing success rates
    - Recording evaluation videos
    - Logging results
    
    Unlike WorkspaceIL (training), this class focuses solely on evaluation
    and loads normalization statistics from the checkpoint rather than
    computing them from data.
    
    Attributes:
        work_dir (Path): Working directory for logs and videos
        cfg: Hydra configuration object
        device (torch.device): Device for evaluation (cuda/cpu)
        env (list): List of dm_env environments for evaluation
        agent (BCAgent): Point-Bridge policy agent
        stats (dict): Normalization statistics loaded from checkpoint
        logger (Logger): Logger for evaluation metrics
        video_recorder (VideoRecorder): Records evaluation videos
    """
    
    def __init__(self, cfg):
        """
        Initialize the evaluation workspace.
        
        Args:
            cfg: Hydra configuration object (from config_eval.yaml)
        """
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # Load data (used to get task information, not for training)
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = 300
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )

        try:
            # for point based methods
            self.num_points_per_obj = self.cfg.suite.num_points_per_obj
            self.cfg.suite.task_make_fn.max_num_objects = (
                self.expert_replay_loader.dataset._max_num_objects
            )
        except:
            pass

        try:
            self.cfg.suite.task_make_fn.task_names = (
                self.expert_replay_loader.dataset.tasks
            )
        except:
            pass

        self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create agent
        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )

        self.envs_till_idx = len(self.env)
        self.expert_replay_loader.dataset.envs_till_idx = self.envs_till_idx
        self.expert_replay_iter = iter(self.expert_replay_loader)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def eval(self):
        """
        Evaluate the loaded policy on all environments.
        
        Runs multiple episodes per environment (num_eval_episodes) and records:
        - Success rate per environment
        - Average episode reward per environment
        - Evaluation videos
        
        Results are logged to TensorBoard and printed to console.
        Videos are saved to work_dir as {global_frame}_env{idx}.mp4
        
        Note:
            - Agent is set to eval mode (train=False)
            - Uses normalization statistics loaded from checkpoint
            - Try-except commented out for debugging (enable in production)
        """
        self.agent.train(False)
        episode_rewards = []
        successes = []
        for env_idx in range(self.envs_till_idx):
            print(f"evaluating env {env_idx}")
            episode, total_reward = 0, 0
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
            success = []

            self.video_recorder.init(self.env[env_idx], enabled=True)
            while eval_until_episode(episode):
                # try:
                time_step = self.env[env_idx].reset()
                # except:
                #     success.append(0)
                #     episode += 1
                #     continue

                self.agent.buffer_reset()
                step = 0

                # Create a tqdm progress bar for this episode
                pbar = tqdm(
                    desc=f"Env {env_idx} Episode {episode}",
                    unit="steps",
                    unit_scale=True,
                    unit_divisor=1000,
                    dynamic_ncols=True,
                )

                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            time_step.observation,
                            self.stats,
                            step,
                            self.global_step,
                        )

                    time_step = self.env[env_idx].step(action)
                    self.video_recorder.record(self.env[env_idx])
                    total_reward += time_step.reward
                    step += 1

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({"reward": f"{total_reward:.2f}", "step": step})

                # Close the progress bar
                pbar.close()

                episode += 1
                success.append(time_step.observation["goal_achieved"])
            self.video_recorder.save(f"{self.global_frame}_env{env_idx}.mp4")
            episode_rewards.append(total_reward / episode)
            successes.append(np.mean(success))

        for _ in range(len(self.env) - self.envs_till_idx):
            episode_rewards.append(0)
            successes.append(0)
        
        # Explicitly close all environments to prevent EGL cleanup errors
        for env in self.env:
            try:
                if hasattr(env, 'close'):
                    env.close()
            except Exception:
                pass

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward_env{env_idx}", reward)
                log(f"success_env{env_idx}", successes[env_idx])
            log("episode_reward", np.mean(episode_rewards[: self.envs_till_idx]))
            log("success", np.mean(successes))
            log("episode_length", step * self.cfg.suite.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

        self.agent.train(True)

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        self.agent.clear_buffers()
        keys_to_save = ["timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        self.agent.buffer_reset()

    def load_snapshot(self, snapshots):
        """
        Load trained checkpoint for evaluation.
        
        Args:
            snapshots (dict): Dictionary with key "bc" pointing to checkpoint path
            
        Note:
            - Loads agent weights and sets to eval mode (eval=True)
            - Loads normalization statistics from checkpoint (critical!)
            - Uses weights_only=False for compatibility with newer PyTorch versions
            
        The loaded stats are used to normalize observations and denormalize actions
        during evaluation, ensuring consistency with training.
        """
        # Load behavior cloning checkpoint
        with snapshots["bc"].open("rb") as f:
            # payload = torch.load(f)
            payload = torch.load(f, weights_only=False)  # for newer PyTorch (5090 compatibility)
        
        # Separate agent parameters from workspace parameters
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        self.agent.load_snapshot(agent_payload, eval=True)

        # Load normalization statistics - CRITICAL for correct evaluation
        # These must match the statistics used during training
        self.stats = payload["stats"]


@hydra.main(config_path="cfgs", config_name="config_eval")
def main(cfg):
    """
    Main entry point for Point-Bridge evaluation.
    
    This script:
    1. Creates evaluation environments
    2. Loads a trained checkpoint (including weights and normalization stats)
    3. Runs evaluation episodes and computes success rates
    4. Records videos of evaluation rollouts
    5. Logs results to TensorBoard
    
    Args:
        cfg: Hydra configuration from cfgs/config_eval.yaml
        
    Example usage:
        python eval.py agent=pb suite=mimiclabs dataloader=mimiclabs \\
            bc_weight=./exp_local/2025.01.15/my_exp/.../snapshot/300000.pt \\
            suite.num_eval_episodes=20
            
    Required parameters:
        - bc_weight: Path to trained checkpoint (.pt file)
        - suite: Environment configuration (mimiclabs or fr3)
        - dataloader: Dataset configuration (for task information)
        
    Note:
        - Uses config_eval.yaml (not config.yaml) which sets eval=true by default
        - Normalization statistics are loaded from checkpoint, not recomputed
        - Can evaluate with either GT points or VLM-extracted points (set suite.use_vlm_points)
    """
    workspace = Workspace(cfg)

    # Load trained checkpoint - REQUIRED for evaluation
    snapshots = {}
    bc_snapshot = Path(cfg.bc_weight)
    if not bc_snapshot.exists():
        raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
    print(f"loading bc weight: {bc_snapshot}")
    snapshots["bc"] = bc_snapshot
    workspace.load_snapshot(snapshots)

    # Run evaluation
    workspace.eval()


if __name__ == "__main__":
    main()
