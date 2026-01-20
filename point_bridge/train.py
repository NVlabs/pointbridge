# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

#!/usr/bin/env python3

import warnings
import os

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
    """
    Create Point-Bridge agent from configuration and environment specs.
    
    This function extracts observation and action shapes from the environment
    specifications and instantiates the agent (BCAgent) with appropriate
    parameters for point-based policy learning.
    
    Args:
        obs_spec: Environment observation specification (from env.observation_spec())
        action_spec: Environment action specification (from env.action_spec())
        cfg: Hydra configuration object containing agent, suite, and training parameters
        
    Returns:
        BCAgent: Instantiated Point-Bridge agent ready for training/evaluation
        
    Note:
        - Automatically computes total object points from num_points_per_obj * max_num_objects
        - Handles both image-based and point-based observation modalities
    """
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape

    try:
        # For point-based methods: compute total number of object points
        # This is used to determine the input size for the PointNet encoder
        cfg.agent.num_object_points = (
            cfg.suite.num_points_per_obj * cfg.suite.task_make_fn.max_num_objects
        )
    except:
        pass

    return hydra.utils.instantiate(cfg.agent)


class WorkspaceIL:
    """
    Main training workspace for Point-Bridge imitation learning.
    
    This class manages the entire training pipeline including:
    - Loading demonstration data and computing normalization statistics
    - Creating environments for evaluation
    - Instantiating the Point-Bridge agent
    - Training loop with periodic evaluation and checkpointing
    - Logging metrics to TensorBoard
    
    The workspace follows the standard RL training paradigm but focuses on
    behavior cloning from expert demonstrations.
    
    Attributes:
        work_dir (Path): Working directory for logs and checkpoints
        cfg: Hydra configuration object
        device (torch.device): Device for training (cuda/cpu)
        expert_replay_loader: DataLoader for demonstration data
        stats (dict): Normalization statistics computed from data
        env (list): List of dm_env environments for evaluation
        agent (BCAgent): Point-Bridge policy agent
        logger (Logger): TensorBoard logger
        video_recorder (VideoRecorder): Records evaluation videos
        _global_step (int): Current training step
        _global_episode (int): Current episode count
    """
    
    def __init__(self, cfg):
        """
        Initialize the training workspace.
        
        Args:
            cfg: Hydra configuration object with training parameters
        """
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # Load demonstration data and create DataLoader
        # The dataset computes normalization statistics (min/max) for all modalities
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)
        self.stats = self.expert_replay_loader.dataset.stats  # Normalization statistics

        # Create logger for TensorBoard
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        
        # Create environments for evaluation
        # Extract dataset statistics to configure environment properly
        self.cfg.suite.task_make_fn.max_episode_len = (
            self.expert_replay_loader.dataset._max_episode_len
        )
        self.max_episode_len = self.expert_replay_loader.dataset._max_episode_len
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )

        try:
            # For point-based methods: pass number of points per object and max objects
            # This ensures environment and agent use consistent point cloud sizes
            self.num_points_per_obj = self.cfg.suite.num_points_per_obj
            self.cfg.suite.task_make_fn.max_num_objects = (
                self.expert_replay_loader.dataset._max_num_objects
            )
        except:
            pass

        try:
            # For MimicLabs: pass task names to create appropriate environments
            self.cfg.suite.task_make_fn.task_names = (
                self.expert_replay_loader.dataset.tasks
            )
        except:
            pass

        # Create environments (returns list of dm_env environments)
        self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create agent
        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )

        self.envs_till_idx = self.expert_replay_loader.dataset.envs_till_idx

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
        Evaluate the current policy on all environments.
        
        Runs multiple episodes per environment and records:
        - Episode rewards
        - Success rates (based on goal_achieved flag)
        - Evaluation videos
        
        Results are logged to TensorBoard and videos are saved to work_dir.
        The agent is set to eval mode during evaluation and restored to train mode after.
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
                try:
                    time_step = self.env[env_idx].reset()
                except:
                    success.append(0)
                    episode += 1
                    continue

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

    def train(self):
        """
        Main training loop for Point-Bridge.
        
        Trains the agent via behavior cloning on demonstration data with:
        - Periodic evaluation (every eval_every_steps)
        - Periodic logging (every log_every_steps) 
        - Periodic checkpointing (every save_every_steps)
        
        The training continues until num_train_steps is reached. Each step:
        1. Samples a batch from demonstration data
        2. Computes policy loss (MSE between predicted and expert actions)
        3. Updates agent parameters via gradient descent
        """
        # Predicates for controlling training/eval/logging frequency
        train_until_step = utils.Until(self.cfg.suite.num_train_steps, 1)
        log_every_step = utils.Every(self.cfg.suite.log_every_steps, 1)
        eval_every_step = utils.Every(self.cfg.suite.eval_every_steps, 1)
        save_every_step = utils.Every(self.cfg.suite.save_every_steps, 1)

        metrics = None
        while train_until_step(self.global_step):
            # try to evaluate
            if (
                self.cfg.eval
                and eval_every_step(self.global_step)
                and self.global_step > 0
            ):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            # update
            metrics = self.agent.update(
                self.expert_replay_iter,
                self.global_step,
            )
            self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # log
            if log_every_step(self.global_step):
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                    log("total_time", total_time)
                    log("actor_loss", metrics["actor_loss"])
                    log("step", self.global_step)

            # save snapshot
            if save_every_step(self.global_step):
                self.save_snapshot()

            self._global_step += 1

    def save_snapshot(self):
        """
        Save training checkpoint to disk.
        
        Saves a checkpoint containing:
        - Agent weights (encoder, policy, optimizers)
        - Training state (global_step, timer)
        - Normalization statistics (for evaluation)
        - Episode length information
        
        Checkpoints are saved to work_dir/snapshot/{global_step}.pt
        These can be loaded for evaluation or to resume training.
        """
        snapshot_dir = self.work_dir / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        snapshot = snapshot_dir / f"{self.global_step}.pt"
        self.agent.clear_buffers()  # Clear observation buffers to reduce checkpoint size
        keys_to_save = [
            "timer",
            "_global_step",
            "_global_episode",
            "stats",  # Critical: normalization statistics for evaluation
            "max_episode_len",
        ]
        if hasattr(self, "num_points_per_obj"):
            keys_to_save.append("num_points_per_obj")
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        self.agent.buffer_reset()

    def load_snapshot(self, snapshots):
        """
        Load checkpoint from disk to resume training.
        
        Args:
            snapshots (dict): Dictionary with key "bc" pointing to checkpoint path
            
        Note:
            - Loads agent weights and training state
            - Sets agent to training mode (eval=False)
            - Used for resuming interrupted training runs
        """
        # Load behavior cloning checkpoint
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f)
        
        # Separate agent parameters from workspace parameters
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        self.agent.load_snapshot(agent_payload, eval=False)


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    """
    Main entry point for Point-Bridge training.
    
    This script:
    1. Loads demonstration data and computes normalization statistics
    2. Creates environments for evaluation
    3. Instantiates the Point-Bridge agent
    4. Trains via behavior cloning with periodic evaluation
    5. Saves checkpoints periodically
    
    Args:
        cfg: Hydra configuration from cfgs/config.yaml
        
    Example usage:
        python train.py agent=pb suite=mimiclabs dataloader=mimiclabs \\
            experiment=my_exp suite.num_train_steps=300000
            
    Configuration files:
        - config.yaml: Main config
        - agent/pb.yaml: Agent architecture
        - suite/mimiclabs.yaml or suite/fr3.yaml: Environment
        - dataloader/mimiclabs.yaml or dataloader/fr3.yaml: Dataset
    """
    workspace = WorkspaceIL(cfg)

    # Optionally load pre-trained weights to resume training
    if cfg.load_bc:
        snapshots = {}
        bc_snapshot = Path(cfg.bc_weight)
        if not bc_snapshot.exists():
            raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
        print(f"loading bc weight: {bc_snapshot}")
        snapshots["bc"] = bc_snapshot
        workspace.load_snapshot(snapshots)

    workspace.train()


if __name__ == "__main__":
    main()
