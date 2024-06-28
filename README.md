# Error Detection on Egocentric Procedural Task Videos

- [Preparation](#Preparation)
- [Preprocessing](#Preprocessing)
- [Training](#Training)
- [Inference](#Inference)

This is the official implementation of [Error Detection on Egocentric Procedural Task Videos](https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_Error_Detection_in_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf)

Please cite our CVPR 204 paper if our paper/implementation is helpful for your research:

```
@InProceedings{Lee_2024_CVPR,
    author    = {Lee, Shih-Po and Lu, Zijia and Zhang, Zekun and Hoai, Minh and Elhamifar, Ehsan},
    title     = {Error Detection in Egocentric Procedural Task Videos},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {18655-18666}
}
```

## Preparation

Setup the conda environment.

```
conda env create -f environment.yml
```

Run setup.py to generate the directories needed

Visit our [project page](https://www.khoury.northeastern.edu/home/eelhami/egoper.htm) to see more details of our dataset.

Please send an email with the following information to lee.shih@northeastern.edu for downloading our datasets and annotations. The shared link will be expired in two weeks.
- Your Full Name
- Institution/Organization
- Advisor/Supervisor Name
- Current Position/Title
- Emaill Address (with institutional domain name)
- Purpose

Here are files information in the dataset.
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

Download annotation.json, active_object.json, mean.npy, and std.npy and put them under data/EgoPER

Create a video and frame folder. Extract pinwheels_videos.zip into the video folder and extract frames from the videos.

```
mkdir data/EgoPER/pinwheels/frames_10fps
mkdir data/EgoPER/pinwheels/trim_videos
cd preprocessing
python extract_frames.py
```

Generate I3D features based on the video frames with the [pre-trained weight](https://drive.google.com/file/d/1SF4NduQ7w08wP00IgftZjnRqRYRdppd6/view?usp=sharing)

Move the weight under I3D_extractor/src/feature_extractor/pretrained_models.

Change root_dir in features_{task_name}.sh to correct path, e.g., data/EgoPER/pinwheels and run

```
mkdir data/EgoPER/pinwheels/features_10fps
cd I3D_extractor
./features_pin.sh
```

## Training

- Modify root_dir in libs/datasets/egoper.py to the correct directory. 
- The action segmentation backbone is ActionFormer
- The number of protoypes of each step is 2

```
./run_EgoPER_train.sh
```


## Inference
- The code will evaluation the performance of action segmentation and error detection.

```
./run_EgoPER_eval.sh
```
