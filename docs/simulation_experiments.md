# Simulation Experiments Instructions

## Dataset setup

- Download the dataset from [Hugging Face](https://huggingface.co/datasets/siddhanthaldar/Point-Bridge) and place it in the `/path/to/pointbridge/data` directory.
- To generate point data for training Point Bridge, run the following command:
```
cd point_bridge/robot_utils/mimiclabs

# For generating data filtered data from the real camera
python generate_pkl.py

# For generating data from randomly placed cameras around the robot
python generate_pkl_manycams.py

# For generating data data containing full point cloud
python generate_pkl_fullpcd.py
```
- Set `HOME_DIR` in the generate data scripts to the root directory of the pointbridge repository.
- Set TASK_NAME in the generate data scripts to the name of the task you want to generate data for.
- To generate data with images, set `save_pixels = True` in the generate data scripts.
- All the scripts add the camera view from the real world setup in sim and generate 3D points using 
  depth from the real camera. To use default sim cameras, set `add_real_cam = False` in the generate data scripts.
- The generated pickle files are saved in the `expert_demos` directory.

## Train Instructions

- For training single task Point Bridge, run the following command:
    ```
    cd point_bridge
    python train.py agent=pb suite=mimiclabs dataloader=mimiclabs eval=false suite.save_every_steps=100000 suite.num_train_steps=300010 use_language=false use_proprio=true num_queries=40 suite.history_len=1 suite.obs_type=[points] dataloader.bc_dataset.suffix=<dataset_suffix> dataloader.bc_dataset.task_indices=[0,1,2,3] experiment=<exp_name> suite.action_mode=pose suite.num_points_per_obj=128
    ```
    - Set `dataloader.bc_dataset.suffix` to the suffix of the dataset you want to train on. For example, for mimiclabs, the generated pickle dataset will be save in `expert_demos/mimiclabs_<dataset_suffix>`. Set `dataloader.bc_dataset.suffix=<dataset_suffix>` in this case.
    - Set `dataloader.bc_dataset.task_indices` to the task indices you want to train on. For example, for the bowl on plate task, the task indices should be [0,1,2,3]. Set `dataloader.bc_dataset.task_indices=[0,1,2,3]` in this case. These indices are the ordered indices of the `.pkl` files in the dataset folder.
    - Set `experiment=<exp_name>` to the name of the experiment you want to run. For example, for the bowl on plate task, the experiment name could be set to `mimiclabs_point_bridge_bowl_on_plate`.
    - For multitask training, set `use_language=true` and modify the `dataloader.bc_dataset.task_indices` to all the task indices you want to train on.
- Training for 300k steps takes around 1.5 hours on a NVIDIA RTX 5090 GPU.
- For each experiment, the logged files and checkpoints are stored in the `exp_local/<date>/<experiment_name>` directory.

## Evaluation Instructions

- For evaluating single task Point Bridge on ground truth points in sim, run the following command:
    ```
    cd point_bridge
    python eval.py agent=pb suite=mimiclabs dataloader=mimiclabs eval=true suite.num_eval_episodes=10 experiment=<exp_name> use_language=false use_proprio=true num_queries=40 suite.history_len=1 suite.eval_history_len=1 suite.obs_type=[points] dataloader.bc_dataset.task_indices=[0,1,2,3] dataloader.bc_dataset.suffix=<dataset_suffix> suite.pixel_keys=["pixels_right","pixels_left"] suite.action_mode=pose suite.num_points_per_obj=128 bc_weight=...
    ```
    - Set `dataloader.bc_dataset.suffix` to the suffix of the dataset you want to evaluate on. For example, for mimiclabs, the generated pickle dataset will be save in `expert_demos/mimiclabs_<dataset_suffix>`. Set `dataloader.bc_dataset.suffix=<dataset_suffix>` in this case.
    - Set `dataloader.bc_dataset.task_indices` to the task indices you want to evaluate on. For example, for the bowl on plate task, the task indices should be [0,1,2,3]. Set `dataloader.bc_dataset.task_indices=[0,1,2,3]` in this case. These indices are the ordered indices of the `.pkl` files in the dataset folder.
    - Set `experiment=<exp_name>` to the name of the experiment you want to evaluate. For example, for the bowl on plate task, the experiment name could be set to `mimiclabs_point_bridge_bowl_on_plate`.
    - Set `bc_weight=...` to the path of the checkpoint you want to evaluate.
    - For multitask evaluation, set `use_language=true` and modify the `dataloader.bc_dataset.task_indices` to all the task indices you want to evaluate on.
- For evaluating using VLM generated points, appending `suite.use_vlm_points=true suite.vlm_mode=segment_depth` to the previous command.
    - You will also need to launch Molmo and Foundation Stereo servers. Refer to the [VLM Launch Instructions](#vlm-launch-instructions) section for more details.
- Evaluation in simulation with ground truth points works at 15Hz and that in the real world (with VLM points) works at 5Hz.
- At the end of the evaluation, all videos and logs are saved in the `exp_local/eval/<date>/<experiment_name>/` directory. The success rate is also printed as `SR` in the terminal.

## VLM Launch Instructions
- Launch Molmo server:
    ```
    cd point_bridge/model_servers/molmo
    python server.py
    ```
    - If launching the Molmo server on a different machine, forward the port 45000 from the machine running the server to the machine running the evaluation script.
    ```
    ssh -L 45000:localhost:45000 <username>@<machine_ip>
    ```
- Launch Foundation Stereo server:
    ```
    cd point_bridge/third_party/FoundationStereo/docker
    bash run_container.sh

    # inside the container
    cd point_bridge/model_servers/foundation_stereo
    python server.py
    ```
    - This must be launched on the same machine as the machine running the evaluation script.

**NOTE:** We observed that VLM evaluations (mostly whole object segmentation using SAM 2) are not very reliable in simulation. So VLM evaluations are used for real world experiments only, and we use the ground truth points for simulation evaluations.

## Instructions for multitask training
- For language-conditioned training, set `use_language=true` in the training and evaluation commands.
- For multitask training, set `dataloader.bc_dataset.task_indices` to all the task indices you want to train on.

