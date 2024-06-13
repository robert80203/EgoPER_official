#!/usr/bin/python3

import numpy as np
import os
import torch
from ..home import get_project_base
# from dataclasses import dataclass
from yacs.config import CfgNode
from . import hand_detection as hd
from .utils import shrink_frame_label, parse_label

BASE = get_project_base()

def load_feature(feature_dir, video, feature_type, transpose):
    file_name = os.path.join(feature_dir, video+feature_type)
    if feature_type == '.npy':
        feature = np.load(file_name)
    elif feature_type == '.npz':
        feature = np.load(file_name)
        feature = feature['data']

    if transpose:
        feature = feature.T
    if feature.dtype != np.float32:
        feature = feature.astype(np.float32)
    
    return feature #[::sample_rate]

# def load_action_mapping(map_fname):
#     label2index = dict()
#     index2label = dict()
#     with open(map_fname, 'r') as f:
#         content = f.read().split('\n')[0:-1]
#         for line in content:
#             id_, l = line.split('|')
#             index2label[int(id_)] = l
#             label2index[l] = int(id_)

#     return label2index, index2label

class Dataset(object):
    """
    self.features[video]: the feature array of the given video (frames x dimension)
    self.input_dimension: dimension of video features
    self.n_classes: number of classes
    """

    def __init__(self, video_list, nclasses, load_video_func):
        """
        """

        self.video_list = video_list
        self.load_video = load_video_func
        self.nclasses = nclasses

        self.data = {}
        self.data[video_list[0]] = load_video_func(video_list[0])
        self.input_dimension = self.data[video_list[0]][0].shape[1] 
    
    def __str__(self):
        string = "< Dataset %d videos, %d feat-size, %d classes >"
        string = string % (len(self.video_list), self.input_dimension, self.nclasses)
        return string
    
    def __repr__(self):
        return str(self)

    def get_vnames(self):
        return self.video_list

    def __getitem__(self, video):
        if video not in self.data:
            self.data[video] = self.load_video(video)

        return self.data[video]

    def __len__(self):
        return len(self.video_list)


# class DataLoader():

#     def __init__(self, dataset, batch_size, shuffle=False):

#         self.num_video = len(dataset)
#         self.dataset = dataset
#         self.videos = list(dataset.get_vnames())
#         self.shuffle = shuffle
#         self.batch_size = batch_size

#         self.num_batch = int(np.ceil(self.num_video/self.batch_size))

#         self.selector = list(range(self.num_video))
#         self.index = 0
#         if self.shuffle:
#             np.random.shuffle(self.selector)
#             # self.selector = self.selector.tolist()

#     def __len__(self):
#         return self.num_batch

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.index >= self.num_video:
#             if self.shuffle:
#                 np.random.shuffle(self.selector)
#                 # self.selector = self.selector.tolist()
#             self.index = 0
#             raise StopIteration

#         else:
#             video_idx = self.selector[self.index : self.index+self.batch_size]
#             if len(video_idx) < self.batch_size:
#                 video_idx = video_idx + self.selector[:self.batch_size-len(video_idx)]
#             videos = [self.videos[i] for i in video_idx]
#             self.index += self.batch_size

#             batch_sequence = []
#             seq_len = []
#             batch_train_label = []
#             batch_eval_label = []
#             for vfname in videos:
#                 sequence, train_label, eval_label = self.dataset[vfname]
#                 seq_len.append(sequence.shape[0])
#                 batch_sequence.append(torch.from_numpy(sequence))
#                 batch_train_label.append(torch.LongTensor(train_label))
#                 batch_eval_label.append(eval_label)


#             seq_len = torch.LongTensor(seq_len)
#             return videos, batch_sequence, seq_len, batch_train_label, batch_eval_label

# class DataLoader():

#     def __init__(self, dataset: Dataset, batch_size, shuffle=False, nmodify=0, sample_rate=1, hide_bg=False):

#         self.num_video = len(dataset)
#         self.dataset = dataset
#         self.videos = list(dataset.get_vnames())
#         self.shuffle = shuffle
#         self.batch_size = batch_size
#         self.sample_rate = sample_rate
#         self.hide_bg = hide_bg
#         self.nmodify = nmodify

#         self.num_batch = int(np.ceil(self.num_video/self.batch_size))
#         self.nclasses = dataset.nclasses

#         self.selector = list(range(self.num_video))
#         self.index = 0
#         if self.shuffle:
#             np.random.shuffle(self.selector)
#             # self.selector = self.selector.tolist()

#     def __len__(self):
#         return self.num_batch

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.index >= self.num_video:
#             if self.shuffle:
#                 np.random.shuffle(self.selector)
#             self.index = 0
#             raise StopIteration

#         else:
#             video_idx = self.selector[self.index : self.index+self.batch_size]
#             if len(video_idx) < self.batch_size:
#                 video_idx = video_idx + self.selector[:self.batch_size-len(video_idx)]
#             videos = [self.videos[i] for i in video_idx]
#             self.index += self.batch_size

#             batch_sequence = []
#             seq_len = []
#             batch_transcript = []
#             batch_train_label = []
#             batch_lapse_label = []
#             batch_slip_label = []
#             batch_eval_label = []
#             batch_eval_lapse_label = []
#             for vfname in videos:
#                 sequence, train_label, train_lapse_label, eval_label, eval_lapse_label = self.dataset[vfname]
#                 segs = parse_label(train_label)
#                 if self.hide_bg:
#                     segs = [ s for s in segs if s.action != 0 ]
#                 N = len(segs)
#                 transcript = [s.action for s in segs] 
#                 T = sequence.shape[0]

#                 slip_label = torch.LongTensor([1]*T)
#                 modify_loc = np.random.choice(len(transcript), self.nmodify, replace=False)
#                 modify_loc = np.sort(modify_loc)
#                 for n in modify_loc:
#                     slip_label[segs[n].start:segs[n].end+1] = 0
#                 for i in range(self.nmodify):
#                     n = modify_loc[i]
#                     if np.random.rand() > 0.5: # delete
#                         transcript.pop(n)
#                         modify_loc = modify_loc - 1
#                     else: # swap
#                         c = transcript[n]
#                         while c == transcript[n]:
#                             c = np.random.choice(self.nclasses-1)+1 # do not choose bg
#                         transcript[n] = c

#                 batch_transcript.append(torch.LongTensor(transcript))
#                 batch_slip_label.append(slip_label)

#                 batch_sequence.append(torch.from_numpy(sequence))
#                 batch_train_label.append(torch.LongTensor(train_label))
#                 batch_lapse_label.append(torch.LongTensor(train_lapse_label))

#                 batch_eval_label.append(eval_label)
#                 batch_eval_lapse_label.append(eval_lapse_label)


#             return videos, batch_sequence, batch_transcript, \
#                  batch_train_label, batch_slip_label, batch_lapse_label, \
#                  batch_eval_label, batch_eval_lapse_label

#------------------------------------------------------------------
#------------------------------------------------------------------

# def create_dataset(cfg: CfgNode):
#     if cfg.dataset == "PTG-coffee":
#         map_fname = BASE + 'dataset/PTG_coffee/label_v1/mapping.txt'
#         dataset_path = BASE + 'dataset/PTG_coffee/label_v1'
#         train_split_fname = BASE + 'dataset/PTG_coffee/splits/'+cfg.split+'.train'
#         test_split_fname = BASE + 'dataset/PTG_coffee/splits/'+cfg.split+'.test'
#         feature_transpose = True
#         feature_type = '.npy'

#         if cfg.feature == 'r3d':
#             feature_path = BASE + 'dataset/PTG_coffee/frames_fps15_i3d_rgb_resnet50/'
#             feature_transpose = False
#         elif cfg.feature == 'r3d2':
#             feature_path = BASE + 'dataset/PTG_coffee/feature2/'
#             feature_transpose = False

#         bg_class = [ 0 ]

#     ################################################
#     ################################################
#     print("Loading from", feature_path)

#     label2index, index2label = load_action_mapping(map_fname)
#     nclasses = len(label2index)

#     """
#     load video interface:
#         Input: video name
#         Output:
#             feature, label_for_training, label_for_evaluation
#     """
#     def load_video(vname):
#         feature = load_feature(feature_path, vname, feature_type, feature_transpose) # should be T x D or T x D x H x W

#         with open(dataset_path + '/groundTruth/' + vname + '.txt') as f:
#             lines = f.read().split('\n')[:-1]
#             lines = [ l.split('|') for l in lines ]
#             # assert feature.shape[0] == len(lines), (vname, feature.shape, len(lines))

#         if feature.shape[0] != len(lines):
#             l = min(feature.shape[0], len(lines))
#             feature = feature[:l]
#             lines = lines[:l]

#         idx = [ i for i, l in enumerate(lines) if l[0] != 'HIDDEN' ] 
#         feature = feature[idx]
#         gt_label = [ label2index[lines[i][0]] for i in idx ]
#         error_label = [ 0 if lines[i][1] == 'Error' else 1 for i in idx ]


#         # downsample if necessary
#         if cfg.sr > 1:
#             feature = feature[::cfg.sr]
#             gt_label_sampled = shrink_frame_label(gt_label, cfg.sr)
#             error_label_sampled = shrink_frame_label(error_label, cfg.sr)
#         else:
#             gt_label_sampled = gt_label
#             error_label_sampled = error_label

#         # segs = parse_label(gt_label_sampled)
#         # transcript = [ s.action for s in segs ]

#         return feature, gt_label_sampled, error_label_sampled, gt_label, error_label
    
#     ################################################
#     ################################################

#     with open(train_split_fname, 'r') as f:
#         video_list = f.read().split('\n')[0:-1]
#     dataset = Dataset(video_list, nclasses, load_video)
    
#     with open(test_split_fname, 'r') as f:
#         test_video_list = f.read().split('\n')[0:-1]
#     test_dataset = Dataset(test_video_list, nclasses, load_video)
        

#     test_dataset.index2label = index2label
#     test_dataset.label2index = label2index

#     return dataset, test_dataset
