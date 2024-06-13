import os
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm

from .models import build_model
from .utils.utils import build_dataflow, AverageMeter, accuracy
from .utils.video_transforms import *
from .utils.video_dataset import VideoDataSet
from .utils.dataset_config import get_dataset_config
#from opts import arg_parser


def next_batch(data, batch_size):
    n_batch = int(np.ceil(data.shape[0] / batch_size))
    for i in range(n_batch):
        yield data[i * batch_size: min((i+1) * batch_size, data.shape[0])]

def get_features(data, model, args, cfg):
    if cfg.feature == 'i3d':
        return eval_a_batch(data, model, 
                            n_frames=args.groups,
                            batch_size=args.groups_per_batch,
                            keep_spatial=False)
    if cfg.feature == 'i3ds' or cfg.feature == 'i3df':
        return eval_a_batch(data, model, 
                            batch_size=args.groups_per_batch,
                            n_frames=args.groups,
                            keep_spatial=True)


def eval_a_batch(data, model, batch_size=1, n_frames=16, keep_spatial=False):
    with torch.no_grad():
        #print('data shape:', data.shape)
        tmp = []
        T = data.shape[2]
        data = torch.cat([torch.zeros_like(data)[:, :, :n_frames-1, :, :], data], dim=2)
        #print('cat data:', data.shape)
        for i in range(T):
            tmp.append(data[0, :, i:i+n_frames, :, :])
        data = torch.stack(tmp)
        data_loader = next_batch(data, batch_size=batch_size)
        feature_list = []
        for i, data_batch in enumerate(data_loader):
            feature = model(data_batch, keep_spatial=keep_spatial)
            feature_list.append(feature)
        feature = torch.cat(feature_list, dim=0)

    return feature

def create_feature_model(args, parallel=True):
    cudnn.benchmark = True

    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(args.dataset)

    data_list_name = args.data_list_name

    args.num_classes = num_classes
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    model, arch_name = build_model(args, test_mode=True)
    mean = model.mean(args.modality)
    std = model.std(args.modality)

    # overwrite mean and std if they are presented in command
    if args.mean is not None:
        if args.modality == 'rgb':
            if len(args.mean) != 3:
                raise ValueError("When training with rgb, dim of mean must be three.")
        elif args.modality == 'flow':
            if len(args.mean) != 1:
                raise ValueError("When training with flow, dim of mean must be three.")
        mean = args.mean

    if args.std is not None:
        if args.modality == 'rgb':
            if len(args.std) != 3:
                raise ValueError("When training with rgb, dim of std must be three.")
        elif args.modality == 'flow':
            if len(args.std) != 1:
                raise ValueError("When training with flow, dim of std must be three.")
        std = args.std

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
    args.mean, args.std = mean, std
    return model

def get_dataloader(args, cfg):
    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(args.dataset)
    # augmentor
    if args.disable_scaleup:
        scale_size = args.input_size
    else:
        scale_size = int(args.input_size / 0.875 + 0.5)
    mean, std = args.mean, args.std

    augments = []
    if cfg.feature == 'i3d' or cfg.feature == 'i3ds':
        augments += [
            GroupScale(scale_size),
            GroupCenterCrop(args.input_size)
        ]
    elif cfg.feature == 'i3df':
        augments += [
            GroupScale(args.input_size),
        ]

    #if args.num_crops == 1:
    #    print('scale_size:', scale_size, args.input_size)
    #    augments += [
    #        GroupScale(scale_size),
    #        GroupCenterCrop(args.input_size)
    #    ]
    #else:
    #    flip = True if args.num_crops == 10 else False
    #    augments += [
    #        GroupOverSample(args.input_size, scale_size, num_crops=args.num_crops, flip=flip),
    #    ]
    augments += [
        Stack(threed_data=args.threed_data),
        ToTorchFormatTensor(num_clips_crops=args.num_clips * args.num_crops),
        GroupNormalize(mean=mean, std=std, threed_data=args.threed_data)
    ]

    augmentor = transforms.Compose(augments)

    # Data loading code
    data_list = os.path.join(args.datadir, args.data_list_name)
    sample_offsets = list(range(-args.num_clips // 2 + 1, args.num_clips // 2 + 1))

    val_dataset = VideoDataSet(args.datadir, data_list, args.groups, args.frames_per_group,
                                 num_clips=args.num_clips, modality=args.modality,
                                 image_tmpl=image_tmpl, dense_sampling=args.dense_sampling,
                                 fixed_offset=not args.random_sampling,
                                 transform=augmentor, is_train=False, test_mode=not args.evaluate,
                                 seperator=filename_seperator, filter_video=filter_video)

    data_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                 workers=args.workers)

    return data_loader


