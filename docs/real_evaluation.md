# Real World Evaluation Instructions

## Data Collection

- We use [Franka Teach](third_party/FrankaTeach) to collect data in the real world.
- The code must also work with other teleoperation systems with necessary modifications for data processing and robot control.

## Point Extraction in the Real World

- As opposed to simulation, which provides us with the ground truth points, we need to extract points from the real world images using the vision foundation models.
- We use the following methods to extract points from the real world images:
  - [Molmo](https://github.com/facebookresearch/Molmo) for object detection
  - [Foundation Stereo](https://github.com/facebookresearch/FoundationStereo) for stereo depth estimation
  - [SAM 2](https://github.com/facebookresearch/segment-anything) for whole object segmentation
- Instructions for launching the model servers are provided in the [VLM Launch Instructions](#vlm-launch-instructions) section.
- After launching the model servers, we can extract points from the real world data using the data preprocessing scripts in `point_bridge/robot_utils/fr3` to save the data for each task in a pickle file.

## Data Generation Instructions

- Go to the `point_bridge/robot_utils/fr3` directory.
- To generate point data for training Point Bridge, first process the raw data from Franka Teach:
```
# Process the raw data from Franka Teach. This will be different for different teleoperation systems.
python process_data.py --data_dir /path/to/raw_data --task_names <task_name1> <task_name2> <task_name3> --num_demos <num_demos> --lang <language_instruction_with_underscores>
```
- Next, for data collected for camera calibration, run the following command:
```
# Convert the camera calibration data to a pickle file
python generate_calib_pkl.py --data_dir /path/to/raw_data --calib_path /path/to/calibration_data --task_names <task_name1> <task_name2> <task_name3> --num_demos <num_demos>
```
- For task data, run the following command:
```
# Convert the processedtask data to a pickle file (set DATASET_PATH and task_names in generate_pkl.py)
python generate_pkl.py
```

## Camera Calibration Instructions
- Go to the `point_bridge/robot_utils/fr3/calibration` directory.
- Camera intrinsics are obtained directly from the camera. In this case, the intrinsic matrices for the left and right cameras in the ZED stereo camera are stored in `constants.py`.
- To generate the extrinsic matrices, run the following command:
```
# Generate the extrinsic matrices. Set the PATH_DATA_PKL to the path of the pkl file generated for the calibration data.
python generate_extrinsic.py 
```

## Training and Evaluation Instructions

- Use the same training and evaluation instructions as provided in the [Simulation Experiments Instructions](simulation_experiments.md). Just set `suite` and `dataloader` to `fr3` in the launch command.