# python imports
import os
import glob
import time
import pickle
import argparse
from pprint import pprint

# torch imports
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

# our code
from libs.core import load_config
from libs.modeling import make_meta_arch
from libs.datasets import make_dataset, make_data_loader, to_frame_wise, to_segments
from libs.utils import valid_one_epoch, fix_random_seed

################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['test_split']) > 0, "Test set must be specified!"
    assert len(cfg['val_split']) > 0, "Validation set must be specified!"
    assert len(cfg['train_split']) > 0, "Train set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['train_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    train_loader = make_data_loader(
        train_dataset, False, None, 1, cfg['loader']['num_workers']
    )
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    test_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['test_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    test_loader = make_data_loader(
        test_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    
    # model.init_prototypes()

    # not ideal for multi GPU training
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    output_name = 'pred_seg_results_'

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    print_freq = args.print_freq
    """Test the model on the validation set"""
    model.eval()
    
    # loop over training set
    for iter_idx, video_list in enumerate(train_loader, 0):
        with torch.no_grad():
            output = model(video_list)
            # initialize the step prototypes
            if iter_idx == 0:
                model(video_list, mode='clustering_init')
            # update the step prototypes
            model(video_list, mode='clustering')
            # use all videos in the training set, or set a limit
            if iter_idx == len(train_loader) - 1 or iter_idx >= 500:
                model(video_list, mode='clustering_flush')
                break
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            torch.cuda.synchronize()
            print('Train: [{0:05d}/{1:05d}]\t'.format(iter_idx, len(train_loader)))

    print('Training set done!')
    print('Size of validation set:', len(val_loader))

    for iter_idx, video_list in enumerate(val_loader, 0):
        with torch.no_grad():
            output = model(video_list)
            num_vids = len(output)
            for vid_idx in range(num_vids):
                # generate frame-wise prediction
                preds = to_frame_wise(output[vid_idx]['segments'], output[vid_idx]['labels'],
                                    output[vid_idx]['scores'], video_list[vid_idx]['feats'].size(1), 
                                    fps=video_list[vid_idx]['fps'])
                # use the frame-wise prediction to generate action labels and time stamps
                action_labels, time_stamp_labels = to_segments(preds)
                video_list[vid_idx]['segments'] = torch.tensor(time_stamp_labels)
                video_list[vid_idx]['labels'] = torch.tensor(action_labels).long()
            model(video_list, mode='get_thresholds')

    print('Validation set done!')

    for ratio in range(-20, 21):
        results = {
            'video-id': [],
            't-start' : [],
            't-end': [],
            'label': [],
            'score': []
        }
        threshold = ratio / 10
        output_file = os.path.join(args.ckpt, output_name+'%.2f.pkl'%(threshold))

        # loop over test set
        for iter_idx, video_list in enumerate(test_loader, 0):
            # forward the model (wo. grad)
            with torch.no_grad():
                
                # predict segments (boundaries and action steps)
                output = model(video_list)
                num_vids = len(output)
                for vid_idx in range(num_vids):
                    # generate frame-wise results and re-generate segments
                    preds = to_frame_wise(output[vid_idx]['segments'], output[vid_idx]['labels'],
                                        output[vid_idx]['scores'], video_list[vid_idx]['feats'].size(1), 
                                        fps=video_list[vid_idx]['fps'])
                    action_labels, time_stamp_labels = to_segments(preds)
                    video_id = output[vid_idx]['video_id']
                    video_list[vid_idx]['segments'] = torch.tensor(time_stamp_labels)
                    video_list[vid_idx]['labels'] = torch.tensor(action_labels).long()

                # perform error detection
                output = model(video_list, mode=args.mode, threshold=threshold)

                num_vids = len(output)
                for vid_idx in range(num_vids):
                    if output[vid_idx]['segments'].shape[0] > 0:
                        video_id = output[vid_idx]['video_id']
                        if video_id not in results:
                            results[video_id] = {}
                        results[video_id]['segments'] = output[vid_idx]['segments'].numpy()
                        results[video_id]['label'] = output[vid_idx]['labels'].numpy()
                        results[video_id]['score'] = output[vid_idx]['scores'].numpy()

            # printing
            if (iter_idx != 0) and iter_idx % (print_freq) == 0:
                torch.cuda.synchronize()
                print('Threshold:%.3f, Test: [%05d/%05d]\t'%(threshold, iter_idx, len(test_loader)))

        with open(output_file, "wb") as f:
            pickle.dump(results, f)
    print('Test set done!')

    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        help='print frequency (default: 20 iterations)')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--mode', default='similarity', type=str)  
    parser.add_argument('--score', action='store_true')       
    args = parser.parse_args()
    main(args)
