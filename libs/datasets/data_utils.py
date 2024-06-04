import os
import copy
import random
import numpy as np
import random
import torch

# generate time stamps from frame-wise labels
# label: (num of frames,)
def to_segments(labels):
    pre_label = None
    pre_idx = 0
    segments = []
    output_labels = []
    for i in range(len(labels)):
        if pre_label is None:
            pre_label = labels[i].item()
        
        if pre_label != labels[i]:
            output_labels.append(pre_label)
            segments.append([pre_idx, i - 1])
            pre_label = labels[i].item()
            pre_idx = i
    
    output_labels.append(pre_label)
    segments.append([pre_idx, i])
    return output_labels, segments

# generate frame-wise labels time stamps
# segments: (num of segments, 2), labels: (num of segments,), scores: (num of segments,)
def to_frame_wise(segments, labels, scores, length, default_cls=0, fps=1):
    preds = torch.zeros((length)) + default_cls
    if scores is None:
        for j in range(len(segments)):
            if segments[j, 1] != segments[j, 0]:
                preds[int(segments[j, 0]*fps):int(segments[j, 1]*fps)] = labels[j]
    else:
        asce_scores, indices = torch.sort(scores, 0)
        error_indices = []
        for j in indices:
            if segments[j, 1] != segments[j, 0]: #and scores[j] > 0.3:
                preds[int(segments[j, 0]*fps):int(segments[j, 1]*fps)] = labels[j]

    return preds.long()

# generate fixed-size object classes, object bboxes, and edge maps
def generate_node_connected(object_info, num_node, height, width, remove_unknown=False):
    # generate input connected map
    for i in range(len(object_info)):
        connected_input_map = np.zeros((num_node, num_node))
        num_object = num_node if len(object_info[i]) > num_node else len(object_info[i])
        for j in range(num_object):
            if remove_unknown and object_info[i][j]['category_id'] == 42:
                continue
            
            is_active_j = object_info[i][j]['active']

            # connect the hands to all the active objects
            if object_info[i][j]['category_id'] == 12:
                is_active_j = True
            elif object_info[i][j]['category_id'] == 13:
                is_active_j = True
            
            for k in range(num_object):
                if remove_unknown and object_info[i][j]['category_id'] == 42:
                    continue
                
                is_active_k = object_info[i][k]['active']

                # connect the hands to all the active objects
                if object_info[i][k]['category_id'] == 12:
                    is_active_k = True
                elif object_info[i][k]['category_id'] == 13:
                    is_active_k = True
                
                if j == k:
                    connected_input_map[j, k] = 1
                else:
                    if is_active_j and is_active_k:
                        connected_input_map[j, k] = 1

        if i == 0:
            connected_input_map_all = np.expand_dims(connected_input_map, axis=0)
        else:
            connected_input_map_all = np.concatenate((connected_input_map_all, np.expand_dims(connected_input_map, axis=0)), axis=0)

    # generate object class and BBOX
    for i in range(len(object_info)):
        object_cate = np.zeros((num_node))
        object_bbox = np.zeros((num_node, 4))
        num_object = num_node if len(object_info[i]) > num_node else len(object_info[i])
        for j in range(num_object):
            if remove_unknown and object_info[i][j]['category_id'] == 42:
                continue
            object_cate[j] = object_info[i][j]['category_id']
            x_min, y_min, width, height = object_info[i][j]['bbox']
            object_bbox[j, 0] = x_min / width
            object_bbox[j, 1] = y_min / height
            object_bbox[j, 2] = (x_min + width) / width
            object_bbox[j, 3] = (y_min + height) / height
        if i == 0:
            object_cate_all = np.expand_dims(object_cate, axis=0)
            object_bbox_all = np.expand_dims(object_bbox, axis=0)
        else:
            object_cate_all = np.concatenate((object_cate_all,np.expand_dims(object_cate, axis=0)), axis=0)
            object_bbox_all = np.concatenate((object_bbox_all,np.expand_dims(object_bbox, axis=0)), axis=0)

    return object_cate_all, object_bbox_all, connected_input_map_all 


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch


def truncate_feats(
    data_dict,
    max_seq_len,
    trunc_thresh,
    crop_ratio=None,
    max_num_trials=200,
    has_action=True,
    no_trunc=False
):
    """
    Truncate feats and time stamps in a dict item

    data_dict = {'video_id'        : str
                 'feats'           : Tensor C x T
                 'segments'        : Tensor N x 2 (in feature grid)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : int
                 'feat_num_frames' : in

    """
    # get the meta info
    feat_len = data_dict['feats'].shape[1]
    num_segs = data_dict['segments'].shape[0]

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio == None:
            return data_dict
        # randomly crop the seq by setting max_seq_len to a value in [l, r]
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                #print('corner')
                return data_dict

    # otherwise, deep copy the dict
    data_dict = copy.deepcopy(data_dict)

    # try a few times till a valid truncation with at least one action
    for _ in range(max_num_trials):

        # sample a random truncation of the video feats
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32)

        # compute the intersection between the sampled window and all segments
        window = window[None].repeat(num_segs, 1)
        left = torch.maximum(window[:, 0], data_dict['segments'][:, 0])
        right = torch.minimum(window[:, 1], data_dict['segments'][:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(
            data_dict['segments'][:, 1] - data_dict['segments'][:, 0])
        inter_ratio = inter / area_segs

        # only select those segments over the thresh
        seg_idx = (inter_ratio >= trunc_thresh)

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = torch.logical_and(
                (inter_ratio > 0.0), (inter_ratio < 1.0)
            )
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if seg_idx.sum().item() > 0:
                break
        else:
            # without any constraints
            break
    # print('start, end, feat len', st, ed, feat_len)
    # feats: C x T
    data_dict['feats'] = data_dict['feats'][:, st:ed].clone()
    # segments: N x 2 in feature grids
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    # shift the time stamps due to truncation
    data_dict['segments'] = data_dict['segments'] - st
    # labels: N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()
    # bbox class: T x num_node
    if 'bbox_class' in data_dict:
        data_dict['bbox_class'] = data_dict['bbox_class'][st:ed]
    # bbox class: T x num_node x 4
    if 'bbox' in data_dict:
        data_dict['bbox'] = data_dict['bbox'][st:ed, :]
    # edge map: T x num_node x num_node
    if 'edge_map' in data_dict:
        data_dict['edge_map'] = data_dict['edge_map'][st:ed, :, :]

    data_dict['start'] = st
    data_dict['end'] = ed

    return data_dict
