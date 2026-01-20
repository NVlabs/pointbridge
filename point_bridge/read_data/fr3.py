# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import einops
import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
from scipy.spatial.transform import Rotation as R
from point_bridge.robot_utils.common.utils import matrix_to_rotation_6d


class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        suffix,
        num_demos_per_task,
        history_len,
        action_chunking,
        num_queries,
        img_size,
        num_robot_points,
        num_points_per_obj,
        robot_points_key,
        object_points_key,
        pixel_keys,
        act_subsample,
        obs_subsample,
        obs_type,
        action_mode,
        task_indices=None,
        sampling_probability=None,
        noise_object_points=False,
        noise_std=0.1,
    ):
        path = f"{path}_{suffix}" if suffix is not None else path

        self._history_len = history_len
        self._img_size = np.array(img_size)
        self._pixel_keys = pixel_keys
        self._act_subsample = act_subsample
        self._obs_subsample = obs_subsample
        self._obs_type = obs_type
        self._action_mode = action_mode
        self._noise_object_points = noise_object_points
        self._noise_std = noise_std
        self._sampling_probability = sampling_probability

        # track points
        self._num_robot_points = num_robot_points
        self._num_points_per_obj = num_points_per_obj
        self._robot_points_key = robot_points_key
        self._object_points_key = f"{object_points_key}_{self._num_points_per_obj}"

        # temporal aggregation
        self._action_chunking = action_chunking
        self._num_queries = num_queries if action_chunking else 1

        # list of tasks in path
        self.tasks = [f.stem for f in Path(path).glob("*.pkl")]
        self.tasks.sort()
        if task_indices is not None:
            self.tasks = [self.tasks[i] for i in task_indices]

        if self._sampling_probability is not None:
            assert len(self._sampling_probability) == len(
                self.tasks
            ), "Sampling probability must be the same length as the number of tasks"
            self._sampling_probability = np.array(self._sampling_probability)
            self._sampling_probability = self._sampling_probability / np.sum(
                self._sampling_probability
            )

        # get data paths
        self._paths = []
        for task_name in self.tasks:
            self._paths.extend([Path(path) / f"{task_name}.pkl"])

        paths = {}
        for idx, path in enumerate(self._paths):
            paths[idx] = path
        del self._paths
        self._paths = paths

        # data stats
        self._max_episode_len = 0
        self._max_state_dim = 0
        self._num_samples = 0
        min_track, max_track = None, None
        min_future_track, max_future_track = None, None
        min_eef_state, max_eef_state = None, None
        min_act, max_act = None, None
        max_gripper, min_gripper = None, None

        # read data
        self._episodes = {}
        self._num_demos = {}
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            observations = data["observations"]
            task_emb = data["task_emb"]
            object_points = data["object_points"] if "object_points" in data else None

            # store
            self._episodes[_path_idx] = []
            self._num_demos[_path_idx] = min(num_demos_per_task, len(observations))
            for i in range(min(num_demos_per_task, len(observations))):
                # make gripper states of shape (T,) to keep things consistent across datasets
                if len(observations[i]["gripper_states"].shape) == 2:
                    observations[i]["gripper_states"] = observations[i][
                        "gripper_states"
                    ].flatten()

                # compute actions
                actions = np.concatenate(
                    [
                        observations[i]["eef_states"][self._act_subsample :],
                        observations[i]["gripper_states"][self._act_subsample :, None],
                    ],
                    axis=1,
                )
                actions = np.concatenate(
                    [actions, [actions[-1]] * self._act_subsample], axis=0
                )

                # convert orientation to 6d rotations
                pos, ori, gripper = actions[:, :3], actions[:, 3:7], actions[:, 7:]
                ori = R.from_quat(ori).as_matrix()
                ori = matrix_to_rotation_6d(ori)
                actions = np.concatenate([pos, ori, gripper], axis=1)

                if len(actions) == 0:
                    continue

                # store gripper stats
                if max_gripper is None:
                    max_gripper = np.max(observations[i]["gripper_states"])
                    min_gripper = np.min(observations[i]["gripper_states"])
                else:
                    max_gripper = max(
                        max_gripper, np.max(observations[i]["gripper_states"])
                    )
                    min_gripper = min(
                        min_gripper, np.min(observations[i]["gripper_states"])
                    )

                # store min, max eef state
                eef_states = observations[i]["eef_states"]
                pos, ori = eef_states[:, :3], eef_states[:, 3:7]
                ori = R.from_quat(ori).as_matrix()
                ori = matrix_to_rotation_6d(ori)
                gripper = observations[i]["gripper_states"][:, None]
                eef_states = np.concatenate([pos, ori, gripper], axis=1)
                min_eef_state = (
                    np.minimum(min_eef_state, np.min(eef_states, axis=0))
                    if min_eef_state is not None
                    else np.min(eef_states, axis=0)
                )
                max_eef_state = (
                    np.maximum(max_eef_state, np.max(eef_states, axis=0))
                    if max_eef_state is not None
                    else np.max(eef_states, axis=0)
                )
                observations[i]["eef_states"] = eef_states

                # store
                episode = dict(
                    observation=observations[i],
                    action=actions,
                    task_emb=task_emb,
                )
                if object_points is not None:
                    episode["object_points"] = object_points
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(
                    self._max_episode_len,
                    len(actions),
                )
                self._max_state_dim = self._num_robot_points * 3
                self._num_samples += len(observations[i]["eef_states"])

                if "points" in self._obs_type:
                    # min, max track
                    track_key = f"{self._robot_points_key}_3d"
                    track = observations[i][track_key]
                    track = einops.rearrange(track, "t n d -> (t n) d")
                    min_track = (
                        np.minimum(min_track, np.min(track, axis=0))
                        if min_track is not None
                        else np.min(track, axis=0)
                    )
                    max_track = (
                        np.maximum(max_track, np.max(track, axis=0))
                        if max_track is not None
                        else np.max(track, axis=0)
                    )

                if self._action_mode == "points":
                    # max and min future tracks
                    min_ft = np.concatenate(
                        [min_track for _ in range(self._num_robot_points)],
                        axis=0,
                    )
                    max_ft = np.concatenate(
                        [max_track for _ in range(self._num_robot_points)],
                        axis=0,
                    )
                    min_future_track = (
                        np.minimum(min_future_track, min_ft)
                        if min_future_track is not None
                        else min_ft
                    )
                    max_future_track = (
                        np.maximum(max_future_track, max_ft)
                        if max_future_track is not None
                        else max_ft
                    )

                elif self._action_mode == "pose":
                    # max, min action
                    if min_act is None:
                        min_act = np.min(actions, axis=0)
                        max_act = np.max(actions, axis=0)
                    else:
                        min_act = np.minimum(min_act, np.min(actions, axis=0))
                        max_act = np.maximum(max_act, np.max(actions, axis=0))

        self.stats = {
            "past_tracks": {
                "min": min_track,
                "max": max_track,
            },
            "future_tracks": {
                "min": min_future_track,
                "max": max_future_track,
            },
            "gripper_states": {
                "min": min_gripper,
                "max": max_gripper,
            },
            "proprioceptive": {
                "min": min_eef_state,
                "max": max_eef_state,
            },
            "actions": {
                "min": min_act,
                "max": max_act,
            },
        }

        self.preprocess = {}
        if "points" in self._obs_type:
            self.preprocess["past_tracks"] = lambda x: (
                x - self.stats["past_tracks"]["min"]
            ) / (
                self.stats["past_tracks"]["max"]
                - self.stats["past_tracks"]["min"]
                + 1e-5
            )
        if self._action_mode == "points":
            self.preprocess["future_tracks"] = lambda x: (
                x - self.stats["future_tracks"]["min"]
            ) / (
                self.stats["future_tracks"]["max"]
                - self.stats["future_tracks"]["min"]
                + 1e-5
            )
        self.preprocess["gripper_states"] = lambda x: (
            x - self.stats["gripper_states"]["min"]
        ) / (
            self.stats["gripper_states"]["max"]
            - self.stats["gripper_states"]["min"]
            + 1e-5
        )
        self.preprocess["proprioceptive"] = lambda x: (
            x - self.stats["proprioceptive"]["min"]
        ) / (
            self.stats["proprioceptive"]["max"]
            - self.stats["proprioceptive"]["min"]
            + 1e-5
        )
        if self._action_mode == "pose":
            self.preprocess["actions"] = lambda x: (
                x - self.stats["actions"]["min"]
            ) / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5)

        # augmentations for images
        if "image" in self._obs_type:
            self.aug = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ]
            )

        # Samples from envs
        self.envs_till_idx = len(self._episodes)

    def _sample_episode(self, env_idx=None):
        if env_idx is not None:
            idx = env_idx
        else:
            if self._sampling_probability is not None:
                indices = np.arange(len(self._episodes))
                idx = np.random.choice(indices, p=self._sampling_probability)
            else:
                idx = np.random.choice(list(self._episodes.keys()))

        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode

    def _sample(self):
        episodes, env_idx = self._sample_episode()
        observations = episodes["observation"]
        actions = episodes["action"]
        task_emb = episodes["task_emb"]
        traj_len = len(actions)

        # Sample obs, action
        sample_idx = np.random.randint(0, traj_len)

        # init return dict
        return_dict = {
            "task_emb": task_emb,
        }

        # robot states
        past_robot_states = observations["eef_states"][
            max(
                0,
                sample_idx
                - self._history_len * self._obs_subsample
                + self._obs_subsample,
            ) : sample_idx
            + 1 : self._obs_subsample
        ]
        if len(past_robot_states) < self._history_len:
            prior = np.array(
                [past_robot_states[0]] * (self._history_len - len(past_robot_states))
            )
            past_robot_states = np.concatenate([prior, past_robot_states], axis=0)
        return_dict["proprioceptive"] = self.preprocess["proprioceptive"](
            past_robot_states
        )

        if "image" in self._obs_type:
            for key in self._pixel_keys:
                sampled_pixel = observations[key][
                    max(
                        0,
                        sample_idx
                        - self._history_len * self._obs_subsample
                        + self._obs_subsample,
                    ) : sample_idx
                    + 1 : self._obs_subsample
                ]  # (T, H, W, C) for image, (T, N, 3) for pcd
                if sampled_pixel.shape[0] < self._history_len:
                    prior = np.concatenate(
                        [
                            sampled_pixel[:1]
                            for _ in range(self._history_len - sampled_pixel.shape[0])
                        ],
                        axis=0,
                    )
                    sampled_pixel = np.concatenate([prior, sampled_pixel], axis=0)

                sampled_pixel = torch.stack(
                    [self.aug(sampled_pixel[i]) for i in range(len(sampled_pixel))]
                )
                return_dict[key] = sampled_pixel

        if "depth" in self._obs_type:
            for key in self._pixel_keys:
                sampled_depth = observations[f"depth_{key}"][
                    max(
                        0,
                        sample_idx
                        - self._history_len * self._obs_subsample
                        + self._obs_subsample,
                    ) : sample_idx
                    + 1 : self._obs_subsample
                ]
                if sampled_depth.shape[0] < self._history_len:
                    prior = np.concatenate(
                        [
                            sampled_depth[:1]
                            for _ in range(self._history_len - sampled_depth.shape[0])
                        ],
                        axis=0,
                    )
                    sampled_depth = np.concatenate([prior, sampled_depth], axis=0)
                sampled_depth = torch.stack(
                    [
                        torch.tensor(self.preprocess["depth"](sampled_depth[i]))[None]
                        for i in range(len(sampled_depth))
                    ]
                )
                return_dict[f"depth_{key}"] = sampled_depth

        if "points" in self._obs_type:
            track_key = f"{self._robot_points_key}_3d"
            num_points = self._num_robot_points
            past_robot_points = observations[track_key][
                max(
                    0,
                    sample_idx
                    - self._history_len * self._obs_subsample
                    + self._obs_subsample,
                ) : sample_idx
                + 1 : self._obs_subsample
            ][:, -num_points:]
            if len(past_robot_points) < self._history_len:
                prior = np.array(
                    [past_robot_points[0]]
                    * (self._history_len - len(past_robot_points))
                )
                past_robot_points = np.concatenate(
                    [prior, past_robot_points], axis=0
                )  # (H, N, D)
            return_dict["past_robot_tracks"] = self.preprocess["past_tracks"](
                past_robot_points
            )

            object_key = f"{self._object_points_key}_3d"
            past_object_points = observations[object_key][
                max(
                    0,
                    sample_idx
                    - self._history_len * self._obs_subsample
                    + self._obs_subsample,
                ) : sample_idx
                + 1 : self._obs_subsample
            ]
            if len(past_object_points) < self._history_len:
                prior = np.array(
                    [past_object_points[0]]
                    * (self._history_len - len(past_object_points))
                )
                past_object_points = np.concatenate([prior, past_object_points], axis=0)
            past_object_points = past_object_points[:, :, : self._num_points_per_obj]
            return_dict["past_object_tracks"] = self.preprocess["past_tracks"](
                past_object_points
            )

        # past gripper_states
        past_gripper_states = observations[f"gripper_states"][
            max(
                0,
                sample_idx
                - self._history_len * self._obs_subsample
                + self._obs_subsample,
            ) : sample_idx
            + 1 : self._obs_subsample
        ]
        if len(past_gripper_states) < self._history_len:
            prior = np.array(
                [past_gripper_states[0]]
                * (self._history_len - len(past_gripper_states))
            )
            past_gripper_states = np.concatenate([prior, past_gripper_states], axis=0)
        if len(past_gripper_states.shape) > 1:
            past_gripper_states = past_gripper_states.reshape(-1)
        return_dict["past_gripper_states"] = self.preprocess["gripper_states"](
            past_gripper_states
        )

        future_tracks = []
        num_future_tracks = self._history_len + self._num_queries - 1

        # for action sampling
        start_idx = min(sample_idx + self._act_subsample, traj_len - 1)
        end_idx = min(start_idx + num_future_tracks * self._act_subsample, traj_len)

        if self._action_mode == "points":
            # robot track
            track_key = f"{self._robot_points_key}_3d"
            num_points = self._num_robot_points
            ft = observations[track_key][start_idx : end_idx : self._act_subsample][
                :, -num_points:
            ]
            if len(ft) < num_future_tracks:
                post = np.array([ft[-1]] * (num_future_tracks - len(ft)))
                ft = np.concatenate([ft, post], axis=0)  # (T, N, D)
            ft = ft.transpose(
                1, 0, 2
            )  # (N, T, D) where T=history_len+num_queries-1=H+Q-1
            ft = np.lib.stride_tricks.sliding_window_view(
                ft, self._num_queries, 1
            )  # (N, H, D, Q)
            ft = ft.transpose(1, 0, 3, 2)  # (H, N, Q, D)

            ft = einops.rearrange(ft, "h n q d -> h q n d")
            ft = einops.rearrange(ft, "h q n d -> h q (n d)")

            future_tracks.append(ft)

            future_tracks = np.concatenate(future_tracks, axis=1)
            return_dict["future_tracks"] = self.preprocess["future_tracks"](
                future_tracks
            )
            # future gripper_states
            future_gripper_states = observations[f"gripper_states"][
                start_idx : end_idx : self._act_subsample
            ]
            if len(future_gripper_states) < num_future_tracks:
                post = np.array(
                    [future_gripper_states[-1]]
                    * (num_future_tracks - len(future_gripper_states))
                )
                future_gripper_states = np.concatenate(
                    [future_gripper_states, post], axis=0
                )
            future_gripper_states = future_gripper_states.reshape(
                future_gripper_states.shape[0]
            )
            future_gripper_states = np.lib.stride_tricks.sliding_window_view(
                future_gripper_states, self._num_queries
            )
            return_dict["future_gripper_states"] = self.preprocess["gripper_states"](
                future_gripper_states
            )

        elif self._action_mode == "pose":
            if self._action_chunking:
                # arrange sampled action to be of shape (history_len, num_queries, action_dim)
                num_actions = (
                    self._history_len + self._num_queries - 1
                )  # -1 since its num_queries including the last action of the history
                start_idx = sample_idx
                end_idx = min(
                    start_idx + num_actions * self._act_subsample, len(actions)
                )
                act = actions[start_idx : end_idx : self._act_subsample]
                if len(act) < num_actions:
                    post = np.array(
                        [actions[-1]] * (num_actions - len(act)), dtype=np.float32
                    )
                    act = np.concatenate([act, post], axis=0)
                sampled_action = np.lib.stride_tricks.sliding_window_view(
                    act, (self._num_queries, actions.shape[-1])
                )  # (H, Q, D)
                sampled_action = sampled_action[:, 0]
            else:
                sampled_action = actions[
                    sample_idx : sample_idx
                    + self._history_len * self._act_subsample : self._act_subsample
                ]
            return_dict["actions"] = self.preprocess["actions"](sampled_action)

        return return_dict

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples
