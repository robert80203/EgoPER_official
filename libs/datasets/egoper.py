import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from .datasets import register_dataset
from .data_utils import truncate_feats, generate_node_connected

@register_dataset("EgoPER")
class EgoPERdataset(Dataset):
    def __init__(
        self,
        is_training,        # if in training mode
        split,              # split, a tuple/list allowing concat of subsets
        default_fps,        # default fps
        max_seq_len,        # maximum sequence length during training
        trunc_thresh,       # threshold for truncate an action segment
        crop_ratio,         # a tuple (e.g., (0.9, 1.0)) for random cropping
        height,             # height of the frame (default: 720)
        width,              # width of the frame (default: 1280)
        num_classes,        # num of action classes
        background_ratio,   # ratio of sampled background
        num_node,           # num of nodes in a graph
        use_gcn,            # if using AOD
        task,               # task name
        ###########
        #-- Changes here
        feat_dirname: str = 'feature_10fps',
        data_root_dir: str = './data'
        ###########
    ):
        ##############
        # changes here
        # root_dir = '/mnt/raptor/datasets/EgoPER'
        root_dir = data_root_dir
        ##############
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
        
        ##############
        # changes here
        print(f'Using feature directory: {feat_dirname}')
        # self.feat_path = os.path.join(root_dir, task, 'features_10fps')
        self.feat_path = os.path.join(root_dir, task, feat_dirname)
        ##############
        
        with open(os.path.join(root_dir, task, self.split+'.txt'), 'r') as fp:
            lines = fp.readlines()
            self.data_list = [line.strip('\n') for line in lines]
        with open(os.path.join(root_dir, 'annotation.json'), 'r') as fp:
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


        # graph input
        if self.use_gcn:
            with open(os.path.join(root_dir, 'active_object.json'), 'r') as fp:
                all_active_obj = json.load(fp)
            
            active_obj = all_active_obj[task]
            self.bboxes = {}
            self.bbox_classes = {}
            self.edge_maps = {}
            for i in range(len(active_obj)):
                video_id = active_obj[i]['video_id']
                if video_id in self.data_list:
                    object_info = active_obj[i]['active_obj']
                    bbox_class, bbox, edge_map = generate_node_connected(object_info, num_node, height, width)
                    self.bboxes[video_id] = bbox
                    self.bbox_classes[video_id] = bbox_class
                    self.edge_maps[video_id] = edge_map

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

        
        if self.use_gcn:
            bbox_class = self.bbox_classes[video_id]
            bbox = self.bboxes[video_id]
            edge_map = self.edge_maps[video_id]
            data_dict['bbox_class'] = torch.tensor(bbox_class).long()
            data_dict['bbox'] = torch.tensor(bbox).float()
            data_dict['edge_map'] = torch.tensor(edge_map).float()


        # truncate the features during training
        if self.is_training:
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio)

        return data_dict


