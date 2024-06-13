import argparse
import torch
import time
import numpy as np
import os
from PIL import Image
from glob import glob
from tqdm import tqdm
import pickle

from .feature_extractor.opts import arg_parser as arg_parser_feature
from .feature_extractor.extract_feature import build_model
from yacs.config import CfgNode
import json
from .feature_extractor.utils.video_transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize
import torchvision.transforms as transforms

def hiedict2cfg(cfg_dict:dict) -> CfgNode:
    cfg = CfgNode()
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            v = hiedict2cfg(v)
        cfg[k] = v
    return cfg

def load_cfg_json(cfg_file):

    with open(cfg_file, 'r') as fp:
        cfg_dict = json.load(fp)
    cfg = hiedict2cfg(cfg_dict)

    return cfg

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    else:
        return x

def create_feature_model(args, num_classes, parallel=False):

    args.num_classes = num_classes

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    model, arch_name = build_model(args, test_mode=True)
    # mean = model.mean(args.modality)
    # std = model.std(args.modality)

    # # overwrite mean and std if they are presented in command
    # if args.mean is not None:
    #     if args.modality == 'rgb':
    #         if len(args.mean) != 3:
    #             raise ValueError("When training with rgb, dim of mean must be three.")
    #     elif args.modality == 'flow':
    #         if len(args.mean) != 1:
    #             raise ValueError("When training with flow, dim of mean must be three.")
    #     mean = args.mean

    # if args.std is not None:
    #     if args.modality == 'rgb':
    #         if len(args.std) != 3:
    #             raise ValueError("When training with rgb, dim of std must be three.")
    #     elif args.modality == 'flow':
    #         if len(args.std) != 1:
    #             raise ValueError("When training with flow, dim of std must be three.")
    #     std = args.std

    model = model.cuda()
    model.eval()

    if args.pretrained is not None:
        print("=> using pre-trained feature model '{}'".format(arch_name))
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> creating model '{}'".format(arch_name))

    if parallel:
        model = torch.nn.DataParallel(model).cuda()
    # args.mean, args.std = mean, std
    return model

class VideoDataSet(torch.utils.data.Dataset):

    def __init__(self, image_list, transform=None):
        """
        Args:
            transform: the transformer for preprocessing
        """

        self.transform = transform
        self.image_list = image_list

    def _load_image(self, image_path_file):

        def _safe_load_image(img_path):
            img_tmp = Image.open(img_path)
            img = img_tmp.copy()
            img_tmp.close()
            return img

        num_try = 0
        # image_path_file = self._image_path(directory, idx)
        img = None
        while num_try < 10:
            try:
                img = [_safe_load_image(image_path_file)]
                break
            except Exception as e:
                print('[Will try load again] error loading image: {}, error: {}'.format(image_path_file, str(e)))
                num_try += 1

        if img is None:
            raise ValueError('[Fail 10 times] error loading image: {}'.format(image_path_file))

        return img

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        """
        Returns:
            torch.FloatTensor: TxCxHxW dimension 
        """

        images = self._load_image(self.image_list[index]) # PIL image

        images = self.transform(images) # torch tensor
        TC, H, W = images.shape
        images = images.view(-1, 3, H, W) # T, C, H, W, T==1

        return images

def create_frame_loader(feature_args, feature_model, image_list):
    mean = feature_model.mean(feature_args.modality)
    std = feature_model.std(feature_args.modality)
    # overwrite mean and std if they are presented in command
    if feature_args.mean is not None:
        if feature_args.modality == 'rgb':
            if len(feature_args.mean) != 3:
                raise ValueError("When training with rgb, dim of mean must be three.")
        elif feature_args.modality == 'flow':
            if len(feature_args.mean) != 1:
                raise ValueError("When training with flow, dim of mean must be three.")
        mean = feature_args.mean

    if feature_args.std is not None:
        if feature_args.modality == 'rgb':
            if len(feature_args.std) != 3:
                raise ValueError("When training with rgb, dim of std must be three.")
        elif feature_args.modality == 'flow':
            if len(feature_args.std) != 1:
                raise ValueError("When training with flow, dim of std must be three.")
        std = feature_args.std


    # augmentor
    if feature_args.disable_scaleup:
        scale_size = feature_args.input_size
    else:
        scale_size = int(feature_args.input_size / 0.875 + 0.5)

    augments = [
        GroupScale(scale_size),
        GroupCenterCrop(feature_args.input_size),
        Stack(threed_data=feature_args.threed_data),
        ToTorchFormatTensor(num_clips_crops=feature_args.num_clips * feature_args.num_crops), # second option is useless
        GroupNormalize(mean=mean, std=std, threed_data=feature_args.threed_data)
    ]

    augmentor = transforms.Compose(augments)

    loader = VideoDataSet(image_list, augmentor)
    return loader

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help="gpu id to use")
parser.add_argument('--frames', type=str, required=True, help="Folder storing all the frames from a video")
parser.add_argument('--feature_model', type=str, required=True, help="path to feature model checkpoint")
parser.add_argument('--mp', action='store_true')
parser.add_argument('--savedir', type=str, required=True)

args = parser.parse_args()

FEATURE_DIM = 2048


print(args.frames)
vname = os.path.basename(args.frames)
savedir = args.savedir
os.makedirs(savedir, exist_ok=True)
savefname = f'{args.savedir}/{vname}.npy'
print(savefname)
if os.path.exists(savefname) or (not os.path.exists(args.frames)):
    exit()

if args.mp:
    print('WARNING - Enable Mixed Precision')

feature_parser = argparse.ArgumentParser('feature model')
arg_parser_feature(feature_parser)
feature_args = feature_parser.parse_args( args=['--groups', '32', '-e', 
            '--frames_per_group', '2', '--without_t_stride', 
            '--logdir', 'logs/', '--dataset', "kinetics400", 
            '--backbone_net', "i3d_resnet", '--num_crops', '1', '--input_size', '256', 
            '--disable_scaleup', '-b', '1', '-j', '24', '--dense_sampling', '--gpu', '0', 
            '--datadir', "", '-d', '50', '--pretrained', args.feature_model ])
feature_model = create_feature_model(feature_args, 400, parallel=False)
feature_model.eval()
feature_model.cuda()

print(args.frames)

images = glob(os.path.join(args.frames, '*.jpg'))
if len(images) == 0:
    images = glob(os.path.join(args.frames, '*.png'))
images.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
loader = create_frame_loader(feature_args, feature_model, images)


### start inference
frame_feature = None

frame_buffer = feature_args.groups * feature_args.frames_per_group
frame_buffer = torch.zeros([frame_buffer, 3, feature_args.input_size, feature_args.input_size]) # window_size, C, H, W
frame_idx = torch.arange(feature_args.groups) * feature_args.frames_per_group 
frame_idx = frame_idx.cuda()
frame_buffer = frame_buffer.cuda()

with torch.no_grad():            
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.mp):
        for i, frame in enumerate(tqdm(loader, disable=True)):
            frame = frame.cuda()
            frame_buffer = frame_buffer.roll(-1, 0)
            frame_buffer[-1] = frame[0]

            frame_clip = torch.transpose(frame_buffer[frame_idx], 1, 0).unsqueeze(0) # T, C, H, W -> 1, C, T, H, W

            feature = feature_model(frame_clip) # 1, 2048

            if frame_feature is None:
                frame_feature = feature.detach().cpu().numpy()
            else:
                frame_feature = np.concatenate([frame_feature, feature.detach().cpu().numpy()], axis=0)
            if frame_feature.shape[0] % 100 == 0:
                print(frame_feature.shape)

np.save(savefname, to_numpy(frame_feature))
