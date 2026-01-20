# Architecture & Codebase Overview

This document provides a detailed overview of the Point-Bridge architecture and codebase organization.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Observations                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Robot Points │  │Object Points │  │ Proprioception│  │
│  │   (9, 3)     │  │(N, 128, 3)   │  │    (10,)      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬────────┘  │
└─────────┼──────────────────┼──────────────────┼──────────┘
          │                  │                  │
          ▼                  ▼                  ▼
    ┌─────────┐        ┌─────────┐       ┌─────────┐
    │ PointNet│        │ PointNet│       │   MLP   │
    │ Encoder │        │ Encoder │       │         │
    └────┬────┘        └────┬────┘       └────┬────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ GPT Transformer  │
                  │  (8 layers)      │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │   Action Head     │
                  │  (Deterministic/ │
                  │    Diffusion)     │
                  └────────┬─────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Actions   │
                    │(40 queries) │
                    └─────────────┘
```

## Key Components

### PointNet Encoder

Processes point clouds into feature vectors:
- **Input**: Point cloud of shape `(num_points, 3)`
- **Output**: Feature vector of shape `(repr_dim,)`
- **Architecture**: Per-point MLP followed by max pooling aggregation
- **Purpose**: Extract geometric features from 3D points

See [Core Concepts](concepts.md#policy-architecture) for more details.

### GPT Transformer

Aggregates temporal information and processes feature sequences:
- **Layers**: 8 transformer layers
- **Attention Heads**: 4
- **Hidden Dimension**: 512
- **Purpose**: Model temporal dependencies and predict future actions

### Action Head

Predicts multiple future actions (action chunking):
- **Deterministic**: Direct MLP prediction
- **Diffusion** (optional): Multimodal action prediction via denoising
- **Output**: `num_queries` future actions (default: 40)

See [Core Concepts](concepts.md#temporal-action-chunking) for details on action chunking.

## Data Flow

1. **Observation Processing**:
   - Robot points, object points, and proprioception are encoded separately
   - Each modality uses its own encoder (PointNet for points, MLP for proprioception)

2. **Feature Aggregation**:
   - Features from multiple timesteps (history) are concatenated
   - Optional language conditioning can be prepended

3. **Action Prediction**:
   - Transformer processes the feature sequence
   - Action head predicts multiple future actions
   - Temporal aggregation smooths execution

## Repository Structure

```
point_bridge/
├── agent/              # Policy implementation (PointNet + GPT)
├── cfgs/               # Hydra configuration files
├── detection_utils/    # VLM integration (SAM 2, point tracking)
├── model_servers/      # Servers for foundation models
├── read_data/          # Dataset loaders
├── robot_utils/        # Robot-specific utilities and data processing
│   ├── common/         # Shared utilities (transforms, camera utils)
│   ├── mimiclabs/      # MimicLabs-specific processing
│   └── fr3/            # Franka FR3-specific processing
├── suite/              # Environment wrappers (MimicLabs, FR3)
├── train.py            # Training script
├── eval.py             # Evaluation script
└── utils.py            # General utilities

docs/                   # Documentation
third_party/            # External dependencies
```

## Codebase Components

### Configuration (`point_bridge/cfgs/`)

Hydra configuration files for:
- **agent/**: Policy architecture configs (pb.yaml)
- **suite/**: Environment configs (mimiclabs.yaml, fr3.yaml)
- **dataloader/**: Dataset configs (mimiclabs.yaml, fr3.yaml)
- **config.yaml**: Main config that composes the above

### Agent (`point_bridge/agent/`)

Implementation of the Point-Bridge policy:
- **networks/**: Neural network components (PointNet, GPT, action heads)
  - **dp3_encoder.py**: PointNet encoder for point clouds
  - **gpt.py**: GPT transformer for temporal modeling
  - **policy_head.py**: Action prediction head (deterministic/diffusion)
- **pb.py**: Main agent class that orchestrates the policy

### Data Loading (`point_bridge/read_data/`)

Data readers for:
- MimicLabs datasets
- FR3 robot datasets
- Handles point cloud processing and normalization
- Computes statistics for data normalization

### Environments (`point_bridge/suite/`)

Environment wrappers that interface with:
- MimicLabs simulation
- FR3 real robot
- Provides unified observation/action interface
- Handles point extraction (ground-truth in sim, VLM-based in real)

### Robot Utilities (`point_bridge/robot_utils/`)

Robot-specific processing:
- **common/**: Shared utilities (transforms, camera calibration, point extraction, VLM detection)
- **mimiclabs/**: MimicLabs-specific data processing (`generate_pkl.py`)
- **fr3/**: Franka FR3-specific utilities and data processing

### Detection Utils (`point_bridge/detection_utils/`)

Vision foundation model integration:
- **point_tracker.py**: Tracks 3D points across frames
- **segment_tracker.py**: Tracks object segments (SAM 2 masks)
- **utils.py**: Detection utility functions and VLM integration

### Model Servers (`point_bridge/model_servers/`)

Servers for vision foundation models:
- Molmo server (object detection)
- Foundation Stereo integration (depth estimation)
- SAM 2 integration (segmentation)

### Scripts

- **train.py**: Main training script
- **eval.py**: Evaluation script
- **utils.py**: General utility functions
- **video.py**: Video recording utilities
- **logger.py**: Logging utilities
