## Installation Instructions
- Clone the repository: `git clone https://github.com/NVlabs/pointbridge`
- Move to the repository: `cd pointbridge`
- Initialize submodules: `git submodule update --init --recursive`
- Create conda environment: `conda env create -f conda_env.yaml`
- Activate the environment and install repository: `conda activate pointbridge` and `pip install -e .`
- Install MimicLabs (for simulation experiments)
```
# install LIBERO (base environment for MimicLabs)
cd third_party/LIBERO
pip install -e .
cd ../../

# install MimicGen (for synthetic data generation)
cd third_party/mimicgen
pip install -e .
cd ../../

# install RoboCasa (for additional assets)
cd third_party/robocasa
pip install -e .
# download robocasa assets
python robocasa/scripts/download_kitchen_assets.py
python robocasa/scripts/setup_macros.py
cd ../../

# install mimiclabs
cd third_party/mimiclabs
pip install -e .
pip install -r requirements.txt
cd ../../
```
- Install torch and other dependencies. This is done separately since installing third party dependencies can sometime mess up the torch versions.
```
pip install torch==2.9.1 torchvision==0.24.1
pip install google-genai
```
- Setup for vision foundation models (for real world experiments)
    - Foundation Stereo (for stereo depth estimation)
    ```
    # Setup
    # first docker pull siddhanthaldar/foundationstereo
    cd third_party/FoundationStereo/docker
    bash run_container.sh

    # instructions for running the server
    cd point_bridge/model_servers/foundation_stereo/
    python server.py
    ```
    - SAM 2 (for object segmentation and tracking)
    ```
    # Setup
    cd third_party/segment-anything-2-real-time
    pip install -e .

    # Download checkpoint
    cd checkpoints
    ./download_ckpts.sh
    ```
    - MAST3R (for cross-view point correspondence -- required only when running point tracking baselines)
    ```
    cd third_party/mast3r
    git submodule update --init --recursive
    ```
    - Co-Tracker (for 2D point tracking -- required only when running point tracking baselines)
    ```
    # Setup

    cd third_party/co-tracker
    pip install -e .
    mkdir -p checkpoints && cd checkpoints
    wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
    ```
