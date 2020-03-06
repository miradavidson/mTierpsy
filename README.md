# Mouse Tracker

Analysis of mouse trajectory data using pose estimation and feature extraction. This document describes how to run the pipeline from beginning to end using DeepLabCut and modules for feature computation.

## Installation

Because of conflicting dependencies, installation is not always straight-forward. Python version should be 3.6, and packages should be installed according to versions defined in the requirements. If you are not using a GPU, install `tensorflow==1.12` instead of `tensorflow-gpu`. If you encounter any other problems, check the [DeepLabCut documentation](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md "DeepLabCut Installation").

## Preparing videos

### Convert and crop videos

You can convert all videos to mp4 by running

```sh
find . -type f -name "*.mpg" -exec sh -c 'ffmpeg -i "$1" "${1%}.mp4"' sh {} \;
```

And cropping with

```sh
find . -type f -name "*.mpg" -exec sh -c 'ffmpeg -i "$1" -vf "crop=w:h:x:y" "${1%.*}_quadrant.mp4"' sh {} \;
```

For example:

```sh
find . -type f -name "*.mpg" \
-exec sh -c 'ffmpeg -i "$1" -vf "crop=240:240:145:105" "${1%.*}_UL.mp4"' sh {} \; \
-exec sh -c 'ffmpeg -i "$1" -vf "crop=240:240:385:105" "${1%.*}_UR.mp4"' sh {} \; \
-exec sh -c 'ffmpeg -i "$1" -vf "crop=240:240:145:350" "${1%.*}_LL.mp4"' sh {} \; \
-exec sh -c 'ffmpeg -i "$1" -vf "crop=240:240:385:350" "${1%.*}_LR.mp4"' sh {} \;
```

Check the resulting videos and tune the cropping parameters accordingly, as different videos may have differently positioned boxes. Set the `video_path` in the metadata accordingly.

### Metadata
It is very important to have a complete metadata file. Each metadata file should contain the following columns:
- `file_id` File identifier. This is used to link metadata to features and should be unique for each video.
- `experiment` Name of the experiment. It is used for calibration across all entries within the same experiment. If each experiment was taken with a different set-up, it can be set equal to `original_video_path`.
- `video_path` Path to the mouse-specific video.
- `original_video_path` (optional) This is only necessary if you are going to use the stabilisation module. IMPORTANT: It should be in .mp4 format or the stabilisation will fail.
- `box_labels_path` Path to the labelled boxes. This will automatically be added when running the analysis.
- `labels_path` Path to the output of DeepLabCut. This will automatically be added when running the analysis.


### Creating a new project
To create a new project, run 

```py
from mousetracker import MouseTracker
project = MouseTracker(path/to/project)
project.create_new_project(path/to/metadata)
```

A new directory with your project name will be created with a `box` and `videos` directory. It is up to you if you want to store your data in the `videos` directory or if you want to refer to the original data. If you do store them in the videos directory, you will have to update your metadata accordingly.

If you want to resume your analysis at any point, simply run

`project = MouseTracker(path/to/project)`


## Add box labels
Corners of the box have to be labelled for measures of the mouse relative to the box. Run

`project.extract_images()`

This will produce the first frame from every video in the `box` folder (ordered as in the metadata file). You will have to label the corners using ImageJ. Open every image in ImageJ, select Multipoint and Analyze>Measure (Ctrl-M). Save to a CSV file with the same numbering (i.e. `0.png` becomes `0.csv`) and save to the `box` folder. Then, run

`project.add_box_to_metadata()`

This orders the box corners and automatically adds the path to the metadata file.

## Calibration
You will need to generate a calibration file. To do so, run:

`project.calibrate(box_size)`

Alternatively, you can make your own CSV file formatted as:

|      | experiment |
|------|------------|
| fps  | ...        |
| pxmm | ...        |

## Pose estimation using DeepLabCut
To extract posture from the videos, you will have to specify the config file of the DeepLabCut model. On `behavgenom$`, the model lives in `mousetracker_dlcmodel`.

`project.label_videos(config)`

You can specify the GPU with the `gputout` parameter.

## Video stabilisation (optional)
If you wish to stabilise your videos, you can run the video stabilisation module. This is an interactive process - that means that you will have to check with each video whether the stabilisation has been successful. We don't want to introduce fake motion! It is expected that some video have drifting frames - however, if you spot a very big peak, there may have been a tracking error and you will want to correct for this (e.g. by changing your mask).

You can load the module by running

`project.stabilise_videos(mask)`

As you see, you will have to define a mask in order to not track the mice or edges. A mask should be a numpy array with the same shape as a frame. You can use the `make_mask` function in the video stabilisation module to generate this. You can then view your mask using the same function by passing the `mask` parameter. An example case:

```py
from mousetracker import make_mask
video = ... # extract example video from your metadata
frame, mask = make_mask(video)
mask[20:-20,20:-20] = 255  # don't include edges of video
mask[100:600,150:630] = 0  # mask box
make_mask(video, mask)
```

Right now, the program will apply the mask on all videos. However, different experiments may have different distances from the box and therefore one mask will not be appropriate for all videos. This is currently not supported. 

For each video, you will be shown a) the points selected for stabilisation and b) the detected motion (dx, dy, da). You will be asked if you want to watch a 'Before and After'. You can quit the video anytime by pressing 'q'. If the detected motion seems reasonable and you are happy with the stabilisation, you can choose to save it to a transformation file.

You can then transform the associated labels (and update the metadata) by running

```project.transform_labels()```


## Feature extraction
You can run the feature extraction with:

`project.extract_features()`

Options are:
- `calibration` Specify the calibration file. The default is `calibration.csv`, which is the output of the calibrate function.
- `output` Specify the output file. The default is `features.csv`.
- `start`, `end` Specifiy the start and end frame, respectively.

## Analysis

The `Analysis` class contains some standard stats functions for the analysis of features. Running `mousetracker.Analysis(metadata, features)` will return an object that comes with in-built functions. It's up to you what you want to explore!
