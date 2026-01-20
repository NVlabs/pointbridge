# Getting Started with Point-Bridge

This guide will help you get started with Point-Bridge quickly, whether you're interested in simulation experiments or real-world deployment.

## Quick Start Overview

Point-Bridge is a framework for learning robot manipulation policies using 3D point representations. The key advantage is that point-based representations enable transfer from simulation to real-world or across different camera viewpoints.

### What You Can Do

1. **Simulation Experiments**: Train and evaluate policies in MimicLabs simulation environment
2. **Real-World Deployment**: Deploy trained policies on a Franka FR3 robot
3. **Custom Environments**: Adapt the pipeline to your own simulation or real-world setup

## Prerequisites

- Ubuntu 24.04
- NVIDIA GPU with CUDA support
- Python 3.11
- Conda package manager

## Installation

Follow the detailed installation instructions in [install.md](install.md). The basic steps are:

1. Clone the repository with submodules
2. Create and activate the conda environment
3. Install Point-Bridge and dependencies
4. Install simulation dependencies (for MimicLabs)
5. Setup vision foundation models (for real-world experiments)

## Your First Experiment

### 1. Download Pre-trained Dataset

```bash
cd scripts
python download_dataset.py
```

The data has been provided as a [Hugging Face Dataset](https://huggingface.co/datasets/siddhanthaldar/Point-Bridge).

### 2. Generate Point Data

```bash
cd point_bridge/robot_utils/mimiclabs
python generate_pkl.py
```

**Note**: Edit `generate_pkl.py` to set the `TASK_NAME` to one of the directory names in `data/mimicgen_data`. In this case, `TASK_NAME` must be one of `bowl_on_plate`, `mug_on_plate`, or `stack_bowls`.

### 3 Set local path in config file for training and evaluation
Set the `root_dir` in `point_bridge/cfgs/local.yaml` to your repository root.

### 4. Train a Policy

```bash
cd point_bridge
python train.py \
    agent=pb \
    suite=mimiclabs \
    dataloader=mimiclabs \
    eval=false \
    suite.save_every_steps=100000 \
    suite.num_train_steps=300010 \
    use_language=false \
    use_proprio=true \
    num_queries=40 \
    suite.history_len=1 \
    suite.obs_type=[points] \
    dataloader.bc_dataset.suffix=no_images \
    dataloader.bc_dataset.task_indices=[0,1,2,3] \
    experiment=my_first_experiment \
    suite.action_mode=pose \
    suite.num_points_per_obj=128
```

NOTE: See [simulation_experiments.md](simulation_experiments.md) for more details on how to configure this command. Output of the training script is logged in the `exp_local/<date>/<experiment_name>` directory.

### 5. Evaluate the Policy

```bash
python eval.py \
    agent=pb \
    suite=mimiclabs \
    dataloader=mimiclabs \
    eval=true \
    suite.num_eval_episodes=10 \
    experiment=my_first_experiment \
    use_language=false \
    use_proprio=true \
    num_queries=40 \
    suite.history_len=1 \
    suite.eval_history_len=1 \
    suite.obs_type=[points] \
    dataloader.bc_dataset.task_indices=[0,1,2,3] \
    dataloader.bc_dataset.suffix=no_images \
    suite.pixel_keys=["pixels_right","pixels_left"] \
    suite.action_mode=pose \
    suite.num_points_per_obj=128 \
    bc_weight=/path/to/checkpoint.pt
```

Replace the `bc_weight` path with your actual checkpoint path.

NOTE: See [simulation_experiments.md](simulation_experiments.md) for more details on how to configure this command. At the end of the evaluation, all videos and logs are saved in the `exp_local/eval/<date>/<experiment_name>/` directory. The success rate is also printed as `SR` in the terminal.

## Understanding Key Concepts

### Point Representations

Point-Bridge uses 3D point clouds as the primary observation modality:

- **Robot Points**: Fixed keypoints on the robot (e.g., gripper, wrist)
- **Object Points**: Sampled points on task-relevant objects
- **Point Extraction**:
  - **Simulation**: Ground-truth from physics engine
  - **Real-world**: Extracted using VLMs (Molmo for detection, SAM 2 for segmentation, Foundation Stereo for depth)

### Configuration System

Point-Bridge uses Hydra for configuration management:

- **agent/**: Policy architecture configs (pb.yaml)
- **suite/**: Environment configs (mimiclabs.yaml, fr3.yaml)
- **dataloader/**: Dataset configs (mimiclabs.yaml, fr3.yaml)
- **config.yaml**: Main config that composes the above

You can override any config parameter from the command line:

```bash
python train.py suite.num_train_steps=500000 batch_size=32
```

### Action Chunking

The policy predicts multiple future actions at once (`num_queries`), then uses exponential averaging for smooth execution:

- `action_chunking=true`: Enable action chunking
- `num_queries=40`: Number of future actions to predict
- `temporal_agg_strategy="exponential_average"`: Aggregation method

## Next Steps

- **Simulation Experiments**: See [simulation_experiments.md](simulation_experiments.md)
- **Real-World Deployment**: See [real_evaluation.md](real_evaluation.md)
- **Adapting to New Environments**: See [adaptation_guide.md](adaptation_guide.md)
- **Understanding the Code**: See [Architecture & Codebase Overview](architecture.md)
