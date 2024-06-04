# Error Detection on Egocentric Procedural Task Videos

- [Preparation](#Preparation)
- [Preprocessing](#Preprocessing)
- [Training](#Training)
- [Inference](#Inference)

This is the official implementation of [Error Detection on Egocentric Procedural Task Videos]()

Please cite our CVPR 204 paper if our paper/implementation is helpful for your research:

```
@InProceedings{,
    author    = {},
    title     = {},
    booktitle = {},
    month     = {},
    year      = {},
    pages     = {}
}
```

## Preparation

Setup the conda environment.

```
conda env create -f environment.yml
```

Run setup.py to generate the directories needed

Dowload the dataset and annotations from the following link
[EgoPER dataset]()

- Annotations
    - annotation.json: the annotation file of 5 tasks, containing time stamps, step names, step decriptions, and action types.
    - active_object.json: the annotation file of 5 tasks, containing frame-wise object and active object bounding boxes, categories, and if objects are active.
- Dataset
    - {task_name}_videos.zip: it contains trimmed RGB videos.
    - {task_name}_other_modalities.zip it contains other modalities such as depth, audio, gaze, hand tracking, etc.
    - training.txt, validation.txt, test.txt: the splits for training, validation, and test.
    - trim_start_end.txt: the start and end time that we trimmed from the original videos.

## Preprocessing

Create a dataset folder for the task you want

```
mkdir data
mkdir data/EgoPER
mkdir data/EgoPER/pinwheels
```

Create a video and frame folder. Extract pinwheels_videos.zip into the video folder and extract frames from the videos.

```
mkdir data/EgoPER/pinwheels/frames_10fps
mkdir data/EgoPER/pinwheels/trim_videos
cd preprocessing
python extract_frames.py
```

Generate I3D features based on the video frames with the pre-trained weight kinetics400-rgb-i3d-resnet-50-f32-s2-precise_bn-warmupcosine-bs1024-e196.pth.tar



## Training

- Modify root_dir in libs/datasets/egoper.py to the correct directory.

### Model details
- The action segmentation backbone is ActionFormer
- The number of protoypes of each step is 2

```
./run_EgoPER_train.sh
```


## Inference

```
./run_EgoPER_eval.sh
```

