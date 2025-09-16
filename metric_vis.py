import os
import torch
import pickle
import numpy as np
import json
import argparse
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import auc
from libs.datasets.data_utils import to_frame_wise, to_segments
from eval_utils import Video, Checkpoint, eval_omission_error

# strict version
def error_acc(pred, gt, gt_error):
    
    num_correct = 0
    num_total = 0
    pre_gt = None
    pre_gt_error = None
    is_pred_error = False
    idx = 0
    for i in range(len(gt)):
        if pre_gt is None:
            pre_gt = gt[i]
            is_pre_gt_error = True if gt_error[i] == -1 else False
        
        if pred[i] == -1:
            is_pred_error = True
        

        if pre_gt != gt[i]:
            if is_pred_error and is_pre_gt_error:
                num_correct += 1
            elif not is_pred_error and not is_pre_gt_error:
                num_correct += 1
            is_pred_error = False
            pre_gt = gt[i]
            is_pre_gt_error = True if gt_error[i] == -1 else False
            num_total += 1

    if is_pred_error and is_pre_gt_error:
        num_correct += 1
    elif not is_pred_error and not is_pre_gt_error:
        num_correct += 1
    num_total += 1

    return num_correct, num_total

# loose version
# def error_acc(pred, gt, gt_error):
#     num_correct = 0
#     num_total = 0
#     num_error = 0
#     num_nonerror = 0 
#     pre_gt = None
#     pre_gt_error = None
#     is_pred_error = False
#     for i in range(len(gt)):
#         if pre_gt is None:
#             pre_gt = gt[i]
#             is_pre_gt_error = True if gt_error[i] == -1 else False

#         if pred[i] == -1:
#             is_pred_error = True
#             num_error += 1
#         else:
#             num_nonerror += 1

#         if pre_gt != gt[i]:
#             if is_pred_error and is_pre_gt_error and num_error > num_nonerror:
#                 num_correct += 1
#             elif (not is_pred_error and not is_pre_gt_error) or num_error < num_nonerror:
#                 num_correct += 1
#             is_pred_error = False
#             pre_gt = gt[i]
#             is_pre_gt_error = True if gt_error[i] == -1 else False
#             num_error = 0
#             num_nonerror = 0
#             num_total += 1

#     if is_pred_error and is_pre_gt_error and num_error > num_nonerror:
#         num_correct += 1
#     elif (not is_pred_error and not is_pre_gt_error) or num_error < num_nonerror:
#         num_correct += 1
#     num_total += 1

#     return num_correct, num_total

def acc_tpr_fpr(all_preds, all_gts):
    # fpr = fp / (fp + tn)
    all_gt_normal = all_preds[all_gts == 1] # get predicted non-error items
    fp_tn = len(all_gt_normal) # number of total non-error items in the ground truth
    fp = len(all_gt_normal[all_gt_normal == -1]) # get FP
    
    # tpr = tp / (tp + fn)
    all_gt_error = all_preds[all_gts == -1] # get predicted error items
    tp_fn = len(all_gt_error) # number of total error items in the ground truth
    tp = len(all_gt_error[all_gt_error == -1]) # get TP

    # acc
    acc = torch.eq(torch.LongTensor(all_gts), torch.LongTensor(all_preds)).sum() / len(all_gts)

    if tp_fn == 0:
        if tp == 0:
            tpr = 1
        else:
            tpr = 0
    else:
        tpr = tp / tp_fn

    if fp_tn == 0:
        if fp == 0:
            fpr = 1
        else:
            fpr = 0
    else:
        fpr = fp / fp_tn
    
    return acc, tpr, fpr

def acc_precision_recall_f1(all_preds, all_gts, set_labels, each_class = True):
    if each_class:
        method = None
        acc = None
        for j in set_labels:
            each_acc = torch.eq(torch.LongTensor(all_gts[all_gts == j]), torch.LongTensor(all_preds[all_gts == j])).sum() / len(all_gts[all_gts == j])
            if acc is None:
                acc = each_acc.unsqueeze(0)
            else:
                acc = torch.cat((acc, each_acc.unsqueeze(0)), dim=0)
    else:
        method = 'macro'
        acc = torch.eq(torch.LongTensor(all_gts), torch.LongTensor(all_preds)).sum() / len(all_gts)
    p = precision_score(all_gts, all_preds, labels=set_labels, average=method,zero_division=0)
    r = recall_score(all_gts, all_preds, labels=set_labels, average=method,zero_division=0)

    f1 = 2 * p * r / (p + r)
    
    return acc, p, r, f1

def generate_partitions(inputs):
    cur_class = None
    start = 0
    step_partitions = []
    for i in range(len(inputs)):
        if inputs[i] != cur_class and cur_class is not None:
            step_partitions.append((cur_class, i - start + 1))
            start = i + 1
        cur_class = inputs[i]
    step_partitions.append((inputs[len(inputs) - 1], len(inputs) - start + 1))
    return step_partitions

def pred_vis(all_gts, all_preds, mapping, vname, category_colors=None):
    clean_version = True
    gts = all_gts
    preds = all_preds

    if category_colors is None:
        mycmap = plt.matplotlib.cm.get_cmap('rainbow', len(mapping))
        category_colors = [matplotlib.colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]

    gt_partitions = generate_partitions(gts)
          
    plt.figure(figsize=(25, 4))
    plt.subplot(211)

    data_cum = 0
    for i, (l, w) in enumerate(gt_partitions):
        if clean_version:
            rects = plt.barh('gt_segmentation', w, left=data_cum, height=0.5, color=category_colors[l.item()])
        else:
            rects = plt.barh('gt_segmentation', w, left=data_cum, height=0.5,
                            label=mapping[str(l.item())], color=category_colors[l.item()])
            text_color = 'black'
            plt.bar_label(rects, labels = [l.item()], label_type='center', color=text_color)
        data_cum += w

    if not clean_version:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), ncol=1, bbox_to_anchor=(1, -1), loc='lower left', fontsize='small')

    plt.subplot(212)
    pred_partitions = generate_partitions(preds)
    ata_cum = 0
    for i, (l, w) in enumerate(pred_partitions):
        if clean_version:
            rects = plt.barh('pred_segmentation', w, left=data_cum, height=0.5, color=category_colors[l.item()])
        else:
            rects = plt.barh('pred_segmentation', w, left=data_cum, height=0.5,
                        label=mapping[str(l.item())], color=category_colors[l.item()])
            text_color = 'black'
            plt.bar_label(rects, labels = [l.item()], label_type='center', color=text_color)
        
        data_cum += w
        
    plt.savefig(f'./{vname}.jpg')


def draw_pred(outputs, name, mapping, save_path, category_colors=None):
    clean_version = True
    
    if category_colors is None:
        mycmap = plt.matplotlib.cm.get_cmap('rainbow', len(mapping))
        category_colors = [matplotlib.colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
    
    gt_partitions = generate_partitions(outputs)
          
    plt.figure(figsize=(13, 2))

    data_cum = 0
    for i, (l, w) in enumerate(gt_partitions):
        if clean_version:
            # print(l, end=' ')
            rects = plt.barh(name, w, left=data_cum, height=0.3, color=category_colors[l.item()])
        else:
            rects = plt.barh(name, w, left=data_cum, height=0.3,
                            label=mapping[str(l.item())], color=category_colors[l.item()])
            text_color = 'black'
            plt.bar_label(rects, labels = [l.item()], label_type='center', color=text_color)
        data_cum += w
    
    # print()
    if not clean_version:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), ncol=1, bbox_to_anchor=(1, -1), loc='lower left', fontsize='small')
        
    plt.savefig(save_path+'.png')
    plt.clf()
    plt.close()

class ActionSegmentationErrorDetectionEvaluator:
    def __init__(self, args):
        self.args = args
        self.annotations = {}
        self.step_annotations = {}
        task = args.task
        
        # if args.dataset == 'EgoPER':
        #     root_dir = '/mnt/raptor/datasets/EgoPER'
        #     with open(os.path.join(root_dir, args.task, 'test.txt'), 'r') as fp:
        #     # with open(os.path.join(root_dir, args.task, 'training.txt'), 'r') as fp:
        #         lines = fp.readlines()
        #         self.data_list = [line.strip('\n') for line in lines]
        #     with open(os.path.join(root_dir, 'annotation.json'), 'r') as fp:
        #         all_annot = json.load(fp)
        #     # with open(os.path.join(root_dir, 'action_step.json'), 'r') as fp:
        #     #     # all_step_annot = json.load(fp)
        #     #     self.step_annotations = json.load(fp)[task]
        #     # step_annot = all_step_annot[task]
        #     # for i in range(len(step_annot)):
        #     #     video_id = step_annot[i]['video_id']
        #     #     if video_id in self.data_list:
        #     #         self.step_annotations[video_id] = step_annot[i]['steps']
        # elif args.dataset == "HoloAssist":
        #     root_dir = '/mnt/raptor/datasets/HoloAssist'
        #     with open(os.path.join(root_dir, args.task, 'test.txt'), 'r') as fp:
        #         lines = fp.readlines()
        #         self.data_list = [line.strip('\n') for line in lines]
        #     with open(os.path.join(root_dir, 'annotation.json'), 'r') as fp:
        #         all_annot = json.load(fp)

        root_dir = '/mnt/raptor/datasets/%s'%(args.dataset)
        with open(os.path.join(root_dir, args.task, 'test.txt'), 'r') as fp:
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
                self.annotations[video_id] = [np.array(annot['segments'][i]['labels']['time_stamp']) * args.fps, 
                                              np.array(actions),
                                              np.array(action_types),
                                              annot['segments'][i]['labels']['error_description']]
        
        
        action2idx = annot['action2idx']
        actiontype2idx = annot['actiontype2idx']
        self.idx2action = {}
        for key, value in action2idx.items():
            self.idx2action[int(value)] = key
        
        # new updated
        self.idx2action[len(self.idx2action)] = "addition"

        self.idx2actiontype = {}
        for key, value in actiontype2idx.items():
            self.idx2actiontype[int(value)] = key
        # print(self.idx2actiontype)
        self.set_labels = [i for i in range(len(action2idx))]

    # EDA
    def macro_segment_error_detection(self, output_list=None, threshold=None, is_visualize=False):

        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, "pred_seg_results_%.2f.pkl"%(threshold)), "rb") as f:
            results = pickle.load(f)

        preds = None
        gts = None
        total_correct = 0
        total = 0
        for video_id in self.data_list:
            
            gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
            
            length = int(gt_segments[-1, 1])

            gt = to_frame_wise(gt_segments, gt_labels, None, length)
            gt_error = to_frame_wise(gt_segments, gt_label_types, None, length)

            # convert all error types to -1
            gt_error[gt_error > 0] = -1
            gt_error[gt_error == 0] = 1

            segments = torch.tensor(results[video_id]['segments'])
            labels = torch.tensor(results[video_id]['label'])
            scores = torch.tensor(results[video_id]['score'])

            pred = to_frame_wise(segments, labels, None, length)

            # convert all normal classes to 1
            pred[pred >= 0] = 1

            num_correct, num_total = error_acc(pred, gt, gt_error)
            
            total_correct += num_correct
            total += num_total
        
        if output_list is not None:
            output_list.append(total_correct / total)
            return output_list
        else:
            print("|Error detection accuracy|%.3f|"%(total_correct / total))

    # Micro AUC
    def micro_framewise_error_detection(self, output_list=None, threshold=None, is_visualize=False):
        
        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, "pred_seg_results_%.2f.pkl"%(threshold)), "rb") as f:
            results = pickle.load(f)

        preds = None
        gts = None

        video_pair_list = []
        video_idx = 0

        for video_id in self.data_list:
            
            gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
            
            length = int(gt_segments[-1, 1])

            gt_error = to_frame_wise(gt_segments, gt_label_types, None, length)
            
            # convert all error types to -1
            gt_error[gt_error > 0] = -1
            gt_error[gt_error == 0] = 1

            segments = torch.tensor(results[video_id]['segments'])
            labels = torch.tensor(results[video_id]['label'])
            scores = torch.tensor(results[video_id]['score'])

            # print(video_id, length, segments[-1], len(labels), len(segments))
            pred = to_frame_wise(segments, labels, None, length)

            # convert all normal classes to 1
            pred[pred >= 0] = 1

            if preds is None:
                preds = pred
                gts = gt_error
            else:
                preds = torch.cat((preds, pred), dim=0)
                gts = torch.cat((gts, gt_error), dim=0)
            
            new_pred = pred.clone()
            new_gt_error = gt_error.clone()
            new_pred = 1 - new_pred
            new_gt_error[new_gt_error == 1] = 0
            new_gt_error[new_gt_error == -1] = 1
            video_pair_list.append(Video(video_idx, new_pred.tolist(), new_gt_error.tolist()))
            # video_pair_list.append(Video(video_idx, pred.tolist(), gt_error.tolist()))
            video_idx += 1

            if is_visualize:
                if not os.path.exists(os.path.join('./visualization/', self.args.dataset, self.args.dirname)):
                    os.mkdir(os.path.join('./visualization/', self.args.dataset, self.args.dirname))
                cp_gt = np.copy(gt_error)
                cp_pred = np.copy(pred)
                # category_colors = {
                #     1: 'g',
                #     -1: 'r'
                # }
                if threshold >= -1.0 and threshold <= 1.0:
                    if not os.path.exists(os.path.join('./visualization/', self.args.dataset, self.args.dirname, 'threshold%.1f'%(threshold))):
                        os.mkdir(os.path.join('./visualization/', self.args.dataset, self.args.dirname, 'threshold%.1f'%(threshold)))
                    # pred_vis(cp_gt, cp_pred, self.idx2actiontype, os.path.join('./visualization/', self.args.dataset, self.args.dirname, 'threshold%.1f'%(threshold), 'ed_'+ video_id))#, category_colors=category_colors)
                    cp_pred[cp_pred == 1] = 0
                    cp_pred[cp_pred == -1] = 1
                    draw_pred(cp_pred, "EgoPED", self.idx2actiontype, os.path.join('./visualization/', self.args.dataset, self.args.dirname, 'threshold%.1f'%(threshold), 'ed_'+ video_id))
        final_acc, final_tpr, final_fpr = acc_tpr_fpr(preds, gts)

        if output_list is not None:
            output_list['acc'].append(final_acc)
            output_list['tpr'].append(final_tpr)
            output_list['fpr'].append(final_fpr)
            return output_list
        else:
            ckpt = Checkpoint(bg_class=[-100])
            ckpt.add_videos(video_pair_list)
            out = ckpt.compute_metrics()
            print("|IoU:%.1f|edit:%.1f|F1@0.5:%.1f|Acc:%.1f|"%(out['IoU']*100, out['edit']*100, out['F1@0.50']*100, out['acc']*100))
            print("|Error detection (thres=%.3f, micro)|acc=%.3f|fpr=%.3f|tpr=%.3f|"%(threshold, final_acc, final_fpr, final_tpr))
        
    # Macro AUC
    def macro_framewise_error_detection(self, output_list=None, threshold=None, is_visualize=False):
        
        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, "pred_seg_results_%.2f.pkl"%(threshold)), "rb") as f:
            results = pickle.load(f)

        preds = None
        gts = None
        acc_list = []
        tpr_list = []
        fpr_list = []
        for video_id in self.data_list:
            
            gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
            
            length = int(gt_segments[-1, 1])

            gt_error = to_frame_wise(gt_segments, gt_label_types, None, length)
            
            # convert all error types to -1
            gt_error[gt_error > 0] = -1
            gt_error[gt_error == 0] = 1

            segments = torch.tensor(results[video_id]['segments'])
            labels = torch.tensor(results[video_id]['label'])
            scores = torch.tensor(results[video_id]['score'])

            pred = to_frame_wise(segments, labels, None, length)
            
            # convert all normal classes to 1
            pred[pred >= 0] = 1

            acc, tpr, fpr = acc_tpr_fpr(pred, gt_error)

            acc_list.append(acc)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        final_acc = np.array(acc_list).mean()
        final_tpr = np.array(tpr_list).mean()
        final_fpr = np.array(fpr_list).mean()

        if output_list is not None:
            output_list['acc'].append(final_acc)
            output_list['tpr'].append(final_tpr)
            output_list['fpr'].append(final_fpr)
            return output_list
        else:
            print("|Error detection (thres=%.3f, macro)|acc=%.3f|fpr=%.3f|tpr=%.3f|"%(threshold, final_acc, final_fpr, final_tpr))

    # Accuracy, Precision, Recall F1 of Action Segmentation
    def micro_framewise_action_segmentation(self, eval_each_class=True, is_visualize=False):
        
        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, 'eval_results.pkl'), "rb") as f:
            results = pickle.load(f)
        
        preds = None
        gts = None
        for video_id in self.data_list:
            
            gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
            
            length = int(gt_segments[-1, 1])
            # print(gt_segments)
            gt = to_frame_wise(gt_segments, gt_labels, None, length)

            segments = torch.tensor(results[video_id]['segments'])
            labels = torch.tensor(results[video_id]['label'])
            scores = torch.tensor(results[video_id]['score'])
            pred = to_frame_wise(segments, labels, None, length)

            if preds is None:
                preds = pred
                gts = gt
            else:
                preds = torch.cat((preds, pred), dim=0)
                gts = torch.cat((gts, gt), dim=0)

            if is_visualize:
                if not os.path.exists(os.path.join('./visualization/', self.args.dataset, self.args.dirname)):
                    os.mkdir(os.path.join('./visualization/', self.args.dataset, self.args.dirname))
                cp_gt = np.copy(gt)
                cp_pred = np.copy(pred)
                # new updated
                cp_gt[cp_gt == -1] = len(self.idx2action) - 1
                
                draw_pred(cp_gt, "EgoPED", self.idx2action, os.path.join('./visualization/', self.args.dataset, self.args.dirname, 'gt_asch_'+ video_id), category_colors=None)
                draw_pred(cp_pred, "EgoPED" , self.idx2action, os.path.join('./visualization/', self.args.dataset, self.args.dirname, 'asch_'+ video_id), category_colors=None)
                # print("zz")
        
        acc, p, r, f1 = acc_precision_recall_f1(preds, gts, self.set_labels, eval_each_class)
        if eval_each_class:
            for j in range(len(self.set_labels)):
                print("|Action segmentation (cls head, class %d)|%.3f|%.3f|%.3f|%.3f|"%(self.set_labels[j], p[j], r[j], f1[j], acc[j]))
        else:
            print("|Action segmentation (cls head, macro)|%.3f|%.3f|%.3f|%.3f|"%(p, r, f1, acc))
        
    # IoU, Edit distance, F1@0.5, Accuracy of Action Segmentation
    def standard_action_segmentation(self):
        
        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, 'eval_results.pkl'), "rb") as f:
            results = pickle.load(f)
        
        preds = None
        gts = None
        input_video_list = []

        for video_id in self.data_list:
            
            gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
            
            length = int(gt_segments[-1, 1])

            gt = to_frame_wise(gt_segments, gt_labels, None, length)

            segments = torch.tensor(results[video_id]['segments'])
            labels = torch.tensor(results[video_id]['label'])
            scores = torch.tensor(results[video_id]['score'])

            pred = to_frame_wise(segments, labels, None, length)

            input_video = Video(video_id, pred.tolist(), gt.tolist())
            input_video_list.append(input_video)

        ckpt = Checkpoint(bg_class=[-1])
        ckpt.add_videos(input_video_list)
        out = ckpt.compute_metrics()
        
        print("|Action segmentation|IoU:%.1f|edit:%.1f|F1@0.5:%.1f|Acc:%.1f|"%(out['IoU']*100, out['edit']*100, out['F1@0.50']*100, out['acc']*100))

    def omission_detection(self):
        if self.args.task == 'tea':
            edges = [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (0, 3), (3, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
            N = 12
            last = 11
        elif self.args.task == 'quesadilla':
            edges = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6), (6, 7), (7, 8), (8, 9)]
            N = 10
            last = 9
        elif self.args.task == 'oatmeal':
            edges = [(0, 1), (0, 2), (1, 4), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 13), (13, 14), (14, 15), (15, 11), (11, 12), (15, 10), (12, 16), (10, 16)]
            N = 17
            last = 16
        elif self.args.task == 'pinwheels':
            edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 12), (12, 8), (8, 9), (9, 10), (10, 11), (11, 13), (13, 14)]
            N = 15
            last = 14
        elif self.args.task == 'coffee':
            edges = [(0, 1), (1, 2), (2, 13), (0, 5), (5, 13), (0, 6), (6, 7), (7, 8), (8, 12), (0, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 3), (3, 4), (4, 16)]
            N = 17
            last = 16
        
        nodes = [i for i in range(1, N-1)]

        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, 'eval_results.pkl'), "rb") as f:
            results = pickle.load(f)

        all_pred_action_labels = []
        all_gt_action_labels = []
        for video_id in self.data_list:
            gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
            labels = torch.tensor(results[video_id]['label'])
            all_pred_action_labels.append(labels.numpy())
            gt_omitted = list(set(nodes) - set(gt_labels.tolist()))
            all_gt_action_labels.append(np.array(gt_omitted))
            # print(labels.numpy())
            # print(gt_labels)
            # all_gt_action_labels.append(self.step_annotations[video_id])

        eval_omission_error(self.args.task, all_pred_action_labels, all_gt_action_labels, [edges, N, last])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', type=str)
    parser.add_argument('--dataset', type=str, default='EgoPER')
    parser.add_argument('--task', type=str, default='pinwheels')
    parser.add_argument('--fps', default=10, type=int)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('-as', '--action-segmentation', action='store_true', help='Evaluate action segmentation using cls head')
    parser.add_argument('-ed', '--error-detection', action='store_true')
    parser.add_argument('-od', '--omission-detection', action='store_true', help='always with flag --error')
    parser.add_argument('--threshold', default=-100.0, type=float, help='If set to 0.0, plot the curve and fine the best threshold')
    parser.add_argument('-vis', '--visualize', action='store_true')
    
    args = parser.parse_args()

    evaluator = ActionSegmentationErrorDetectionEvaluator(args)

    if args.action_segmentation:
        evaluator.micro_framewise_action_segmentation(eval_each_class=True)
        evaluator.micro_framewise_action_segmentation(eval_each_class=False, is_visualize=args.visualize)
        evaluator.standard_action_segmentation()

    if args.error_detection:
        if args.threshold != -100.0:
            evaluator.micro_framewise_error_detection(threshold=args.threshold, is_visualize=args.visualize)
            evaluator.macro_framewise_error_detection(threshold=args.threshold, is_visualize=False)
            evaluator.macro_segment_error_detection(threshold=args.threshold, is_visualize=False)
        else:
            error_micro_list = {
                'acc': [],
                'tpr': [],
                'fpr': []
            }
            error_macro_list = {
                'acc': [],
                'tpr': [],
                'fpr': []
            }
            eda_list = []
            thresholds = []
            fprs = []
            tprs = []
            for i in range(-20, 21):
                threshold = i / 10
                thresholds.append(threshold)
                error_micro_list = evaluator.micro_framewise_error_detection(output_list=error_micro_list, threshold=threshold, is_visualize=args.visualize)
                error_macro_list = evaluator.macro_framewise_error_detection(output_list=error_macro_list, threshold=threshold)
                eda_list = evaluator.macro_segment_error_detection(output_list=eda_list, threshold=threshold)

            micro_fprs = np.array(error_micro_list['fpr'])
            micro_tprs = np.array(error_micro_list['tpr'])
            macro_fprs = np.array(error_macro_list['fpr'])
            macro_tprs = np.array(error_macro_list['tpr'])
            micro_fprs_tprs = [micro_fprs, micro_tprs]
            macro_fprs_tprs = [macro_fprs, macro_tprs]

            np.save(os.path.join('./ckpt', args.dataset, args.dirname, 'micro_fpr_tpr.npy'), np.array(micro_fprs_tprs))
            np.save(os.path.join('./ckpt', args.dataset, args.dirname, 'macro_fpr_tpr.npy'), np.array(macro_fprs_tprs))
            np.save(os.path.join('./ckpt', args.dataset, args.dirname, 'eda.npy'), np.array(eda_list))

            micro_fprs = np.sort(micro_fprs)
            micro_tprs = np.sort(micro_tprs)
            macro_fprs = np.sort(macro_fprs)
            macro_tprs = np.sort(macro_tprs)

            micro_fprs = np.concatenate((micro_fprs, np.array([1.0])), axis=0)
            micro_tprs = np.concatenate((micro_tprs, np.array([1.0])), axis=0)
            macro_fprs = np.concatenate((macro_fprs, np.array([1.0])), axis=0)
            macro_tprs = np.concatenate((macro_tprs, np.array([1.0])), axis=0)
            # print(micro_fprs, micro_tprs)
            # print(macro_fprs, macro_tprs)
            # print(eda_list)
            micro_auc_value = auc(micro_fprs, micro_tprs)
            macro_auc_value = auc(macro_fprs, macro_tprs)

            print('|%s|EDA: %.1f|Micro AUC: %.1f|Macro AUC: %.1f|'%(args.dirname, np.array(eda_list).mean() * 100, micro_auc_value * 100, macro_auc_value * 100))

    if args.omission_detection:
        evaluator.omission_detection()
