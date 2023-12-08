import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from .datasets import register_dataset
from .data_utils import truncate_feats, generate_node_connected

@register_dataset("HoloAssist")
class HoloAssistdataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        default_fps,     # default fps
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        height,
        width,
        num_classes,
        background_ratio, # ratio of sampled background
        num_node,       # num of nodes in a graph
        use_gcn,
        task,
    ):

        root_dir = '/mnt/raptor/shihpo'
        self.split = split
        self.is_training = is_training
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.crop_ratio = crop_ratio
        self.background_ratio = background_ratio
        self.use_gcn = use_gcn
        self.bg_idx = 0
        self.annotations = {}

        self.feat_path = os.path.join('/mnt/raptor/datasets/HoloAssist_shihpo/feature_10fps_mp')

        
        with open(os.path.join(root_dir, 'holoassist', self.split+'.txt'), 'r') as fp:
            lines = fp.readlines()
            self.data_list = [line.strip('\n') for line in lines]
        with open(os.path.join(root_dir, 'EgoPER/preprocess/holoassist_annotation.json'), 'r') as fp:
            all_annot = json.load(fp)
        
        annot = all_annot[task]
        for i in range(len(annot['segments'])):
            video_id = annot['segments'][i]['video_id']
            if video_id in self.data_list:
                actions = [int(action) for action in annot['segments'][i]['labels']['action']]
                action_types = [int(action_type) for action_type in annot['segments'][i]['labels']['action_type']]
                self.annotations[video_id] = [np.array(annot['segments'][i]['labels']['time_stamp']) * self.default_fps, 
                                              np.array(actions),
                                              np.array(action_types),
                                              annot['segments'][i]['labels']['error_description']]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_id = self.data_list[idx]
        annots = self.annotations[video_id]
        time_stamps, action_labels, action_labels_error, error_description = annots
        error_description = annots[1]
        
        feats = np.load(os.path.join(self.feat_path, video_id+'.npy'))

        # ignore some background segments
        if self.is_training:
            delete_idx = []
            for i in range(len(action_labels)):
                if action_labels[i] == self.bg_idx and random.random() > self.background_ratio:
                    delete_idx.append(i)
            if len(delete_idx) != 0:
                time_stamps = np.delete(time_stamps, delete_idx, 0)
                action_labels = np.delete(action_labels, delete_idx, 0)
                action_labels_error = np.delete(action_labels_error, delete_idx, 0)

        data_dict = {
            'feats': torch.from_numpy(feats).permute(1, 0).float(),
            'segments': torch.from_numpy(time_stamps).float(),
            'labels': torch.from_numpy(action_labels).long(),
            'labels_error': torch.from_numpy(action_labels_error).long(),
            'video_id': str(video_id),
            'fps': self.default_fps,
            'duration': len(feats) / self.default_fps,
        }

        # truncate the features during training
        if self.is_training:
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio)

        return data_dict


