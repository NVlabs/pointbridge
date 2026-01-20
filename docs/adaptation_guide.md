# Adaptation Guide: Using Point-Bridge for New Environments

This guide explains how to adapt Point-Bridge to:
1. A new simulation environment
2. A new real-world robot setup
3. Custom tasks and objects

## Table of Contents

- [Core Components Overview](#core-components-overview)
- [Adapting to New Simulation](#adapting-to-new-simulation)
- [Adapting to Real-World Setup](#adapting-to-real-world-setup)

---

## Core Components Overview

To adapt Point-Bridge, you need to understand these key components:

### 1. Environment Wrapper (`suite/*.py`)

The environment wrapper provides an interface between your robot/simulation environment and the Point-Bridge policy.

**Key responsibilities**:
- Convert raw observations to point clouds
- Handle camera calibration and transformations
- Extract robot and object keypoints
- Manage action execution

**Reference**: `point_bridge/suite/mimiclabs.py` and `point_bridge/suite/fr3.py`

### 2. Data Reader (`read_data/*.py`)

Loads demonstration data and computes statistics for normalization.

**Key responsibilities**:
- Load trajectories from disk
- Compute min/max statistics for normalization
- Handle temporal batching and action chunking

**Reference**: `point_bridge/read_data/mimiclabs.py` and `point_bridge/read_data/fr3.py`

### 3. Configuration (`cfgs/`)

Hydra configs that specify all hyperparameters.

**Key files**:
- `suite/*.yaml`: Environment-specific settings
- `dataloader/*.yaml`: Dataset paths and parameters
- `agent/pb.yaml`: Policy architecture
- `config.yaml`: Main composition

### 4. Data Processing Scripts (`robot_utils/`)

Scripts to convert raw data to the pickle format expected by Point-Bridge.

**Key responsibilities**:
- Extract 3D points from observations
- Compute camera transformations
- Sample object points

---

## Adapting to New Simulation

Follow these steps to use Point-Bridge with a new simulation environment (e.g., IsaacGym, PyBullet, Gazebo).

### Step 1: Create Environment Wrapper

Create a new file `point_bridge/suite/my_env.py`:

```python
import dm_env
import numpy as np
from dm_env import specs, StepType

class MyEnvWrapper(dm_env.Environment):
    """
    Wrapper for your simulation environment.
    """
    
    def __init__(
        self,
        task_name,
        width=256,
        height=256,
        num_robot_points=8,
        num_points_per_obj=128,
        robot_points_key="robot_tracks",
        object_points_key="object_tracks",
        obs_type=["points"],
        action_mode="pose",
        **kwargs
    ):
        self.task_name = task_name
        self.width = width
        self.height = height
        self.num_robot_points = num_robot_points
        self.num_points_per_obj = num_points_per_obj
        self.robot_points_key = f"{robot_points_key}_3d"
        self.object_points_key = f"{object_points_key}_{num_points_per_obj}_3d"
        self.obs_type = obs_type
        self.action_mode = action_mode
        
        # Initialize your simulation environment
        self.env = self._create_env()
        
    def _create_env(self):
        """Create and return your simulation environment."""
        # TODO: Initialize your simulation
        # Example: return MySimulator(task_name=self.task_name)
        pass
    
    def reset(self):
        """Reset environment and return initial observation."""
        # Reset your simulation
        obs = self.env.reset()
        
        # Extract 3D points from simulation
        robot_points_3d = self._get_robot_points()  # Shape: (num_robot_points, 3)
        object_points_3d = self._get_object_points()  # Shape: (num_objects, num_points_per_obj, 3)
        
        # Build observation dictionary
        observation = {
            self.robot_points_key: robot_points_3d,
            self.object_points_key: object_points_3d,
            "eef_pose": self._get_eef_pose(),  # Shape: (7,) - pos (3) + quat (4)
            "gripper_state": self._get_gripper_state(),  # Shape: (1,)
            "goal_achieved": False,
        }
        
        return dm_env.TimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=observation
        )
    
    def step(self, action):
        """Execute action and return next observation."""
        # Execute action in your simulation
        # action is shape (10,) for pose mode: pos (3) + 6d_rot (6) + gripper (1)
        pos = action[:3]
        rot_6d = action[3:9]
        gripper = action[9]
        
        # Convert 6D rotation to your representation
        from point_bridge.robot_utils.common.utils import rotation_6d_to_matrix
        rot_mat = rotation_6d_to_matrix(rot_6d)
        
        # Apply action to your robot
        obs, reward, done, info = self.env.step(pos, rot_mat, gripper)
        
        # Extract points for next observation
        robot_points_3d = self._get_robot_points()
        object_points_3d = self._get_object_points()
        
        observation = {
            self.robot_points_key: robot_points_3d,
            self.object_points_key: object_points_3d,
            "eef_pose": self._get_eef_pose(),
            "gripper_state": self._get_gripper_state(),
            "goal_achieved": info.get("success", False),
        }
        
        step_type = StepType.LAST if done else StepType.MID
        
        return dm_env.TimeStep(
            step_type=step_type,
            reward=reward,
            discount=1.0 if not done else 0.0,
            observation=observation
        )
    
    def _get_robot_points(self):
        """
        Extract 3D keypoints on the robot.
        
        Returns:
            np.ndarray: Shape (num_robot_points, 3)
        """
        # TODO: Get positions of robot keypoints in robot base frame
        # Example keypoints: wrist, finger tips, gripper center, etc.
        # See point_bridge/robot_utils/common/franka_gripper_points.py for Franka example
        
        # Example:
        joint_positions = self.env.get_joint_positions()
        keypoint_positions = self.env.forward_kinematics(joint_positions, keypoint_indices=[...])
        
        return keypoint_positions  # Shape: (num_robot_points, 3)
    
    def _get_object_points(self):
        """
        Extract 3D points on task-relevant objects.
        
        Returns:
            np.ndarray: Shape (num_objects, num_points_per_obj, 3)
        """
        # TODO: Sample points on object meshes or bounding boxes
        # See point_bridge/robot_utils/mimiclabs/sample_object_points.py for reference
        
        object_meshes = self.env.get_object_meshes()
        object_points = []
        
        for mesh in object_meshes:
            # Sample points uniformly on mesh surface
            points = self._sample_mesh_points(mesh, self.num_points_per_obj)
            object_points.append(points)
        
        # Pad if fewer than expected objects
        while len(object_points) < self.expected_num_objects:
            object_points.append(np.zeros((self.num_points_per_obj, 3)))
        
        return np.array(object_points)  # Shape: (num_objects, num_points_per_obj, 3)
    
    def _get_eef_pose(self):
        """Get end-effector pose as (x, y, z, qx, qy, qz, qw)."""
        # TODO: Get current end-effector pose from your simulation
        return self.env.get_eef_pose()
    
    def _get_gripper_state(self):
        """Get gripper state (0 = closed, 1 = open)."""
        # TODO: Get gripper openness from your simulation
        return np.array([self.env.get_gripper_state()])
    
    def observation_spec(self):
        """Define observation space."""
        return {
            self.robot_points_key: specs.Array(
                shape=(self.num_robot_points, 3),
                dtype=np.float32,
                name=self.robot_points_key
            ),
            self.object_points_key: specs.Array(
                shape=(self.expected_num_objects, self.num_points_per_obj, 3),
                dtype=np.float32,
                name=self.object_points_key
            ),
            "eef_pose": specs.Array(shape=(7,), dtype=np.float32, name="eef_pose"),
            "gripper_state": specs.Array(shape=(1,), dtype=np.float32, name="gripper_state"),
            "goal_achieved": specs.Array(shape=(), dtype=bool, name="goal_achieved"),
        }
    
    def action_spec(self):
        """Define action space."""
        # For pose mode: (x, y, z, 6d_rot, gripper) = 10 dims
        return specs.BoundedArray(
            shape=(10,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name="action"
        )
```

### Step 2: Create Data Processor

Create `point_bridge/robot_utils/my_env/generate_pkl.py`:

```python
"""
Convert your demonstration data to Point-Bridge pickle format.
"""

import numpy as np
import pickle as pkl
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from point_bridge.robot_utils.common.utils import matrix_to_rotation_6d


def process_demonstration(demo_data):
    """
    Convert a single demonstration to Point-Bridge format.
    
    Args:
        demo_data: Your raw demonstration data
        
    Returns:
        dict: Processed observation dictionary
    """
    observations = {
        "eef_states": [],  # (T, 10) - pos(3) + 6d_rot(6) + gripper(1)
        "gripper_states": [],  # (T,)
        "robot_tracks_3d": [],  # (T, num_robot_points, 3)
        "object_tracks_128_3d": [],  # (T, num_objects, 128, 3)
    }
    
    for timestep in demo_data:
        # Extract end-effector pose
        pos = timestep["eef_position"]  # (3,)
        quat = timestep["eef_orientation"]  # (4,) as (x, y, z, w)
        
        # Convert quaternion to 6D rotation
        rot_mat = R.from_quat(quat).as_matrix()
        rot_6d = matrix_to_rotation_6d(rot_mat)
        
        gripper = timestep["gripper_state"]  # scalar
        eef_state = np.concatenate([pos, rot_6d, [gripper]])
        
        observations["eef_states"].append(eef_state)
        observations["gripper_states"].append(gripper)
        
        # Extract robot keypoints
        robot_points = timestep["robot_keypoints_3d"]  # (num_robot_points, 3)
        observations["robot_tracks_3d"].append(robot_points)
        
        # Extract object points
        object_points = timestep["object_points_3d"]  # (num_objects, 128, 3)
        observations["object_tracks_128_3d"].append(object_points)
    
    # Convert to numpy arrays
    for key in observations:
        observations[key] = np.array(observations[key])
    
    return observations


def generate_dataset(raw_data_dir, output_dir, task_name, num_demos=50):
    """
    Generate Point-Bridge dataset from raw demonstrations.
    
    Args:
        raw_data_dir: Directory containing raw demonstration data
        output_dir: Where to save processed pickle files
        task_name: Name of the task
        num_demos: Number of demonstrations to process
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw demonstrations
    demo_files = sorted(Path(raw_data_dir).glob("*.npy"))[:num_demos]
    
    all_observations = []
    all_actions = []
    
    for demo_file in demo_files:
        print(f"Processing {demo_file}")
        demo_data = np.load(demo_file, allow_pickle=True).item()
        
        # Process demonstration
        observations = process_demonstration(demo_data)
        
        # Compute actions (future end-effector states)
        actions = observations["eef_states"][1:]
        actions = np.concatenate([actions, [actions[-1]]], axis=0)
        
        all_observations.append(observations)
        all_actions.append(actions)
    
    # Get task embedding (for language-conditioned policies)
    from sentence_transformers import SentenceTransformer
    lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    task_description = f"task: {task_name}"
    task_emb = lang_model.encode([task_description])[0]
    
    # Determine max number of objects
    max_num_objects = max(
        obs["object_tracks_128_3d"].shape[1] for obs in all_observations
    )
    
    # Save to pickle
    output_data = {
        "observations": all_observations,
        "actions": all_actions,
        "task_emb": task_emb,
        "object_points": None,  # Set if you have object point metadata
        "object_features": np.zeros((max_num_objects, 384)),  # Placeholder
    }
    
    output_file = output_dir / f"{task_name}.pkl"
    with open(output_file, "wb") as f:
        pkl.dump(output_data, f)
    
    print(f"Saved dataset to {output_file}")
    print(f"Number of demonstrations: {len(all_observations)}")
    print(f"Max trajectory length: {max(len(obs['eef_states']) for obs in all_observations)}")
    print(f"Number of objects: {max_num_objects}")


if __name__ == "__main__":
    generate_dataset(
        raw_data_dir="/path/to/raw/demos",
        output_dir="/path/to/output",
        task_name="my_task",
        num_demos=50
    )
```

### Step 3: Create Configuration Files

Create `point_bridge/cfgs/suite/my_env.yaml`:

```yaml
# @package _global_

# Environment settings
task_make_fn:
  _target_: point_bridge.suite.my_env.make
  task_names: ["my_task"]
  action_repeat: 1
  height: 256
  width: 256
  seed: ${seed}
  max_episode_len: 300  # Will be set from dataset
  max_state_dim: 100  # Will be set from dataset
  eval: ${eval}
  pixel_keys: ${pixel_keys}
  num_robot_points: ${num_robot_points}
  num_points_per_obj: ${suite.num_points_per_obj}
  robot_points_key: ${robot_points_key}
  object_points_key: ${object_points_key}
  obs_type: ${suite.obs_type}
  action_mode: ${suite.action_mode}
  use_vlm_points: false
  vlm_mode: "segment_depth"

# Suite parameters
hidden_dim: 512
action_repeat: 1
num_eval_episodes: 10
eval_every_steps: 50000
log_every_steps: 100
save_every_steps: 50000
num_train_steps: 300010

# Observation settings
history_len: 1
eval_history_len: 1
pixel_keys: ["pixels"]
proprio_key: eef_pose
feature_key: robot_tracks_3d
robot_points_key: robot_tracks
object_points_key: object_tracks
obs_type: [points]
action_mode: pose
num_robot_points: 8
num_points_per_obj: 128
```

Create `point_bridge/cfgs/dataloader/my_env.yaml`:

```yaml
# @package _global_

bc_dataset:
  _target_: point_bridge.read_data.my_env.BCDataset
  path: ${data_dir}/my_env
  suffix: null
  num_demos_per_task: ${num_demos_per_task}
  history_len: ${suite.history_len}
  action_chunking: ${action_chunking}
  num_queries: ${num_queries}
  img_size: [${suite.height}, ${suite.width}]
  num_robot_points: ${num_robot_points}
  num_points_per_obj: ${suite.num_points_per_obj}
  robot_points_key: ${robot_points_key}
  object_points_key: ${object_points_key}
  pixel_keys: ${pixel_keys}
  act_subsample: 1
  obs_subsample: 1
  obs_type: ${suite.obs_type}
  action_mode: ${suite.action_mode}
  task_indices: [0]  # Override from command line
```

### Step 4: Create Data Reader

Create `point_bridge/read_data/my_env.py` by copying and adapting `point_bridge/read_data/mimiclabs.py`. The key is to ensure your data loader returns batches with the correct keys and shapes.

### Step 5: Train and Evaluate

```bash
# Generate data
cd point_bridge/robot_utils/my_env
python generate_pkl.py

# Train
cd point_bridge
python train.py \
    agent=pb \
    suite=my_env \
    dataloader=my_env \
    eval=false \
    dataloader.bc_dataset.suffix=null \
    dataloader.bc_dataset.task_indices=[0] \
    experiment=my_env_experiment

# Evaluate
python eval.py \
    agent=pb \
    suite=my_env \
    dataloader=my_env \
    eval=true \
    bc_weight=./exp_local/.../snapshot/300000.pt
```

---

## Adapting to Real-World Setup

Adapting Point-Bridge to a new real-world robot involves several additional steps beyond simulation.

### Overview

Real-world deployment requires:
1. Camera calibration (intrinsics and extrinsics)
2. Vision foundation model setup (for point extraction)
3. Robot control interface
4. Data collection pipeline

### Step 1: Camera Calibration

#### Intrinsic Calibration

We use instrinsic matrices directly from the camera. You can also use standard calibration tools to get camera intrinsic matrices:

```python
# Example: Using OpenCV
import cv2
import numpy as np

# Calibrate using checkerboard
# ... standard OpenCV calibration ...

# Save intrinsics
camera_intrinsics = {
    "camera_left": {
        "fx": 700.0,
        "fy": 700.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480,
        "matrix": [[700, 0, 320], [0, 700, 240], [0, 0, 1]]
    },
    # ... more cameras
}

import json
with open("camera_intrinsics.json", "w") as f:
    json.dump(camera_intrinsics, f, indent=2)
```

#### Extrinsic Calibration

Compute transformations from camera frame to robot base frame. Reference: `point_bridge/robot_utils/fr3/calibration/generate_extrinsic.py`


### Step 2: Setup Vision Foundation Models

Point-Bridge uses several VLMs for point extraction. This information has also been provided in the installation instructions in [install.md](install.md).

#### Foundation Stereo (for depth estimation)

```bash
# Pull Docker image
docker pull siddhanthaldar/foundationstereo

# Run container
cd third_party/FoundationStereo/docker
bash run_container.sh

# Inside container, start server
cd point_bridge/model_servers/foundation_stereo
python server.py
```

#### SAM 2 (for segmentation)

```bash
cd third_party/segment-anything-2-real-time
pip install -e .

# Download checkpoints
cd checkpoints
./download_ckpts.sh
```

#### Molmo (for object detection)

```bash
cd point_bridge/model_servers/molmo
python server.py  # Starts server on port 45000
```

### Step 3: Create Robot Control Interface

Create a wrapper that converts Point-Bridge actions to robot commands:

```python
class MyRobotController:
    """Interface between Point-Bridge and your robot."""
    
    def __init__(self):
        # Initialize connection to your robot
        self.robot = self._connect_to_robot()
        
    def execute_action(self, action):
        """
        Execute a Point-Bridge action on the robot.
        
        Args:
            action: (10,) array with pos(3) + 6d_rot(6) + gripper(1)
        """
        pos = action[:3]
        rot_6d = action[3:9]
        gripper = action[9]
        
        # Convert 6D rotation to your robot's representation
        from point_bridge.robot_utils.common.utils import rotation_6d_to_matrix
        from scipy.spatial.transform import Rotation as R
        
        rot_mat = rotation_6d_to_matrix(rot_6d)
        quat = R.from_matrix(rot_mat).as_quat()  # (x, y, z, w)
        
        # Send command to robot
        self.robot.move_to_pose(
            position=pos,
            orientation=quat,
            gripper_state=gripper
        )
    
    def get_current_state(self):
        """Get current robot state."""
        return {
            "eef_position": self.robot.get_eef_position(),
            "eef_orientation": self.robot.get_eef_orientation(),
            "gripper_state": self.robot.get_gripper_state(),
            "joint_positions": self.robot.get_joint_positions(),
        }
```

### Step 4: Implement Point Extraction

Create environment wrapper that uses VLMs for point extraction:

```python
class RealWorldEnvWrapper(dm_env.Environment):
    """Wrapper for real-world robot with VLM-based point extraction."""
    
    def __init__(self, use_vlm_points=True, **kwargs):
        self.robot = MyRobotController()
        self.use_vlm_points = use_vlm_points
        
        if use_vlm_points:
            # Initialize VLM clients
            from point_bridge.detection_utils.utils import init_models_for_vlm_detection
            (
                self.sam_predictor,
                self.gemini_client,
                self.molmo_client,
                self.lang_model,
                self.mast3r_model,
                self.dust3r_inference,
            ) = init_models_for_vlm_detection()
            
            # Initialize depth model
            from point_bridge.model_servers.foundation_stereo.client import FoundationStereoZMQClient
            self.depth_model = FoundationStereoZMQClient(server_address="localhost:60000")
        
        # Load camera calibration
        self.camera_intrinsics = self._load_intrinsics()
        self.camera_extrinsics = self._load_extrinsics()
    
    def _extract_points_vlm(self, images, task_description):
        """
        Extract 3D points using VLMs.
        
        Returns:
            robot_points: (num_robot_points, 3)
            object_points: (num_objects, num_points_per_obj, 3)
        """
        from point_bridge.robot_utils.common.vlm_detection import (
            get_vlm_points_using_segments_depth
        )
        
        # Get stereo depth
        left_img = images["left"]
        right_img = images["right"]
        depth = self.depth_model.predict(left_img, right_img, self.camera_intrinsics["left"])
        
        # Extract object points using VLM
        is_first_step = not hasattr(self, "vlm_tracker")
        
        if is_first_step:
            self.vlm_tracker = None
            self.vlm_objects = []
        
        robot_points, object_points, self.vlm_tracker, self.vlm_objects, _, _ = \
            get_vlm_points_using_segments_depth(
                ref_image=left_img,
                ref_depth=depth,
                task_description=task_description,
                vlm_tracker=self.vlm_tracker,
                vlm_objects=self.vlm_objects,
                is_first_step=is_first_step,
                num_points_per_obj=self.num_points_per_obj,
                current_pose=self.robot.get_current_state()["eef_position"],
                intrinsic_matrix=self.camera_intrinsics["left"]["matrix"],
                extrinsic_matrix=self.camera_extrinsics["left"]["matrix"],
                sam_predictor=self.sam_predictor,
                gemini_client=self.gemini_client,
                molmo_client=self.molmo_client,
                task_name=self.task_name,
                height_limits=(0, 480),
                width_limits=(0, 640),
            )
        
        return robot_points, object_points
```

### Step 5: Data Collection

Collect demonstrations using teleoperation:

```bash
# Start teleoperation interface
python collect_demos.py --task my_task --num_demos 50

# Process collected data
cd point_bridge/robot_utils/my_robot
python process_data.py --data_dir /path/to/raw/data
python generate_pkl.py
```

### Step 6: Training and Deployment

Train the policy on your collected data, then deploy:

```python
# deploy_policy.py
import hydra
from point_bridge.agent.pb import BCAgent

class PolicyDeployer:
    def __init__(self, checkpoint_path):
        # Load trained policy
        self.agent = self._load_agent(checkpoint_path)
        self.env = RealWorldEnvWrapper(use_vlm_points=True)
        
    def run_episode(self):
        time_step = self.env.reset()
        self.agent.buffer_reset()
        
        step = 0
        while not time_step.last():
            # Get action from policy
            action = self.agent.act(
                time_step.observation,
                self.stats,
                step,
                global_step=0
            )
            
            # Execute action
            time_step = self.env.step(action)
            step += 1
        
        return time_step.observation["goal_achieved"]

if __name__ == "__main__":
    deployer = PolicyDeployer(checkpoint_path="path/to/checkpoint.pt")
    success = deployer.run_episode()
    print(f"Episode success: {success}")
```

