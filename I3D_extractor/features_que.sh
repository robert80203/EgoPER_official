export PYTHONPATH=`pwd` 
root_dir=/mnt/raptor/datasets/EgoPER/quesadilla
# Step 1: extract frame images from the videos
#python extract_frames.py $root_dir

video_dir=$root_dir/frames_10fps
feature_dir=$root_dir/features_10fps

# Step 2: extract pretrained video features: the video frames are saved in video_dir, and the extracted features will be saved in feature_dir
for video in `ls $video_dir`
do
	echo $video
	CUDA_VISIBLE_DEVICES=6 python -m src.feature_extract --feature_model src/feature_extractor/pretrained_models/kinetics400-rgb-i3d-resnet-50-f32-s2-precise_bn-warmupcosine-bs1024-e196.pth.tar --frames $video_dir/$video --savedir $feature_dir --mp
done


