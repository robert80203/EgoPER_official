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
from libs.datasets.data_utils import generate_time_stamp_labels, to_frame_wise, to_segments
# from action_segmentation_eval import Video, Checkpoint
# from eval_omission_error import eval_omission_error_ordering
from eval_utils import Video, Checkpoint, eval_omission_error

def error_acc(pred, gt, gt_error, bg_to_normal = None):
    if bg_to_normal is not None:
        pred[pred == bg_to_normal[0]] = bg_to_normal[1]
        gt_error[gt_error == bg_to_normal[0]] = bg_to_normal[1]
    
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

def acc_tpr_fpr(all_preds, all_gts, set_labels, bg_to_normal = None):
    if bg_to_normal is not None:
        all_preds[all_preds == bg_to_normal[0]] = bg_to_normal[1]
        all_gts[all_gts == bg_to_normal[0]] = bg_to_normal[1]

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

def acc_precision_recall_f1(all_preds, all_gts, set_labels, each_class = True, bg_to_normal = None):
    if bg_to_normal is not None:
        all_preds[all_preds == bg_to_normal[0]] = bg_to_normal[1]
        all_gts[all_gts == bg_to_normal[0]] = bg_to_normal[1]

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
    # last partition
    #if inputs[len(inputs) - 1] != background_idx:
    step_partitions.append((inputs[len(inputs) - 1], len(inputs) - start + 1))
    return step_partitions

def pred_vis(all_gts, all_preds, mapping, vname, category_colors=None):
    #print(all_gts)
    #print(all_preds)
    ## convert frames to partitions
    #pred_error_partitions = convert_frame2partition(pred_error_frames)
    #gt_error_partitions = convert_frame2partition(gt_error_frames)
    clean_version = True
    gts = all_gts
    preds = all_preds
    
    #print(preds.size())
    #category_colors = plt.matplotlib.cm.get_cmap('tab10')(np.arange(len(mapping))) #plt.matplotlib.cm.get_cmap('PiYG')(np.linspace(0.15, 0.85, len(mapping)))
    

    if category_colors is None:
        mycmap = plt.matplotlib.cm.get_cmap('rainbow', len(mapping))
        category_colors = [matplotlib.colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
    

    # print(category_colors)

    gt_partitions = generate_partitions(gts)
          
    plt.figure(figsize=(25, 4))
    plt.subplot(211)

    data_cum = 0
    for i, (l, w) in enumerate(gt_partitions):
        if clean_version:
            # print(l, end=' ')
            rects = plt.barh('gt_segmentation', w, left=data_cum, height=0.5, color=category_colors[l.item()])
        else:
            rects = plt.barh('gt_segmentation', w, left=data_cum, height=0.5,
                            label=mapping[str(l.item())], color=category_colors[l.item()])
            text_color = 'black'#'white' if r * g * b < 0.5 else 'darkgrey'
            plt.bar_label(rects, labels = [l.item()], label_type='center', color=text_color)
        data_cum += w
    
    # print()
    if not clean_version:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), ncol=1, bbox_to_anchor=(1, -1), loc='lower left', fontsize='small')
    
    
    #plt.legend(ncol=2, bbox_to_anchor=(1, -2), loc='lower left', fontsize='small')

    plt.subplot(212)
    pred_partitions = generate_partitions(preds)
    ata_cum = 0
    for i, (l, w) in enumerate(pred_partitions):
        if clean_version:
            rects = plt.barh('pred_segmentation', w, left=data_cum, height=0.5, color=category_colors[l.item()])
        else:
            rects = plt.barh('pred_segmentation', w, left=data_cum, height=0.5,
                        label=mapping[str(l.item())], color=category_colors[l.item()])
            text_color = 'black'#'white' if r * g * b < 0.5 else 'darkgrey'
            plt.bar_label(rects, labels = [l.item()], label_type='center', color=text_color)
        
        data_cum += w
        
    plt.savefig(f'./{vname}.jpg')


class ActionSegmentationErrorDetectionEvaluator:
    def __init__(self, args):
        self.args = args
        self.annotations = {}
        self.step_annotations = {}
        task = args.task
        root_dir = '/mnt/raptor/shihpo'
        if args.dataset == 'EgoPER':
            with open(os.path.join(root_dir, args.task, 'test.txt'), 'r') as fp:
                lines = fp.readlines()
                self.data_list = [line.strip('\n') for line in lines]
            with open(os.path.join(root_dir, 'EgoPER/preprocess/annotation.json'), 'r') as fp:
                all_annot = json.load(fp)
            with open(os.path.join(root_dir, 'EgoPER/preprocess/action_step.json'), 'r') as fp:
                all_step_annot = json.load(fp)
            step_annot = all_step_annot[task]
            for i in range(len(step_annot)):
                video_id = step_annot[i]['video_id']
                if video_id in self.data_list:
                    self.step_annotations[video_id] = step_annot[i]['steps']
        elif args.dataset == 'HoloAssist':
            with open(os.path.join(root_dir, 'holoassist', 'test.txt'), 'r') as fp:
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
                self.annotations[video_id] = [np.array(annot['segments'][i]['labels']['time_stamp']) * args.fps, 
                                              np.array(actions),
                                              np.array(action_types),
                                              annot['segments'][i]['labels']['error_description']]
        
        
        action2idx = annot['action2idx']
        self.idx2action = {}
        for key, value in action2idx.items():
            self.idx2action[int(value)] = key
        self.set_labels = [i for i in range(len(action2idx))]

    # updated
    def macro_segment_error_detection(self, output_list=None, threshold=None, combine_bg=False, is_visualize=False):
        
        # with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, 'eval_results.pkl'), "rb") as f:
        #     results = pickle.load(f)
        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, "pred_seg_results_%.2f.pkl"%(threshold)), "rb") as f:
            results = pickle.load(f)

        # if convert bg into normal class
        if combine_bg:
            bg_to_normal = [0, 1]
            set_labels = [1, -1]
        else:
            bg_to_normal = None
            set_labels = [0, 1, -1]

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
            gt_error[gt_error > 1] = -1

            segments = torch.tensor(results[video_id]['segments'])
            labels = torch.tensor(results[video_id]['label'])
            scores = torch.tensor(results[video_id]['score'])

            pred = to_frame_wise(segments, labels, None, length)


            # convert all normal classes to 1
            pred[pred > 1] = 1

            num_correct, num_total = error_acc(pred, gt, gt_error, bg_to_normal)
            
            total_correct += num_correct
            total += num_total
        
        if output_list is not None:
            output_list.append(total_correct / total)
            return output_list
        else:
            print("|Error detection accuracy|%.3f|"%(total_correct / total))

    # updated
    def micro_framewise_error_detection(self, output_list=None, threshold=None, eval_each_class=False, combine_bg=False, is_visualize=False):
        
        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, "pred_seg_results_%.2f.pkl"%(threshold)), "rb") as f:
            results = pickle.load(f)

        # if convert bg into normal class
        if combine_bg:
            bg_to_normal = [0, 1]
            set_labels = [1, -1]
        else:
            bg_to_normal = None
            set_labels = [0, 1, -1]

        preds = None
        gts = None
        for video_id in self.data_list:
            
            gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
            
            length = int(gt_segments[-1, 1])

            gt = to_frame_wise(gt_segments, gt_label_types, None, length)
            
            # convert all error types to -1
            gt[gt > 1] = -1

            segments = torch.tensor(results[video_id]['segments'])
            labels = torch.tensor(results[video_id]['label'])
            scores = torch.tensor(results[video_id]['score'])

            # print(video_id, length, segments[-1], len(labels), len(segments))
            pred = to_frame_wise(segments, labels, None, length)

            # convert all normal classes to 1
            pred[pred > 1] = 1

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
                if bg_to_normal is not None:
                    cp_gt[cp_gt == bg_to_normal[0]] = bg_to_normal[1]
                    cp_pred[cp_pred == bg_to_normal[0]] = bg_to_normal[1]
                    category_colors = {
                        -1: 'r',
                        1: 'g'
                    }
                else:
                    category_colors = {
                        -1: 'r',
                        0: 'purple',
                        1: 'g'
                    }
                if threshold >= -1.0 and threshold <= 1.0:
                    if not os.path.exists(os.path.join('./visualization/', self.args.dataset, self.args.dirname, 'threshold%.1f'%(threshold))):
                        os.mkdir(os.path.join('./visualization/', self.args.dataset, self.args.dirname, 'threshold%.1f'%(threshold)))
                    pred_vis(cp_gt, cp_pred, self.idx2action, os.path.join('./visualization/', self.args.dataset, self.args.dirname, 'threshold%.1f'%(threshold), 'ed_'+ video_id), category_colors=category_colors)
        
        final_acc, final_tpr, final_fpr = acc_tpr_fpr(preds, gts, set_labels, bg_to_normal)

        if output_list is not None:
            output_list['acc'].append(final_acc)
            output_list['tpr'].append(final_tpr)
            output_list['fpr'].append(final_fpr)
            return output_list
        else:
            print("|Error detection (thres=%.3f, micro)|acc=%.3f|fpr=%.3f|tpr=%.3f|"%(threshold, final_acc, final_fpr, final_tpr))
        
    # updated
    def macro_framewise_error_detection(self, output_list=None, threshold=None, eval_each_class=False, combine_bg=False, is_visualize=False):
        
        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, "pred_seg_results_%.2f.pkl"%(threshold)), "rb") as f:
            results = pickle.load(f)

        # if convert bg into normal class
        if combine_bg:
            bg_to_normal = [0, 1]
            set_labels = [1, -1]
        else:
            bg_to_normal = None
            set_labels = [0, 1, -1]

        preds = None
        gts = None
        acc_list = []
        tpr_list = []
        fpr_list = []
        for video_id in self.data_list:
            
            gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
            
            length = int(gt_segments[-1, 1])

            gt = to_frame_wise(gt_segments, gt_label_types, None, length)
            
            # convert all error types to -1
            gt[gt > 1] = -1

            segments = torch.tensor(results[video_id]['segments'])
            labels = torch.tensor(results[video_id]['label'])
            scores = torch.tensor(results[video_id]['score'])

            pred = to_frame_wise(segments, labels, None, length)
            
            # convert all normal classes to 1
            pred[pred > 1] = 1

            acc, tpr, fpr = acc_tpr_fpr(pred, gt, set_labels, bg_to_normal)

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

    # updated
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
            # if len(scores) != len(segments):
            #     pred = to_frame_wise(segments, labels, None, length)
            # else:
            #     pred = to_frame_wise(segments, labels, scores, length)

            # pred_steps, _ = to_segments(pred)
            # gt_steps, _ = to_segments(gt)
            # print(video_id)
            # print('pred', pred_steps)
            # print('gt from annot', gt_steps)
            # print('gt from seq', self.step_annotations[video_id])
            # print('=======================')
            
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
                pred_vis(cp_gt, cp_pred, self.idx2action, os.path.join('./visualization/', self.args.dataset, self.args.dirname, 'asch_'+ video_id), category_colors=None)

        
        acc, p, r, f1 = acc_precision_recall_f1(preds, gts, self.set_labels, eval_each_class, bg_to_normal=None)
        if eval_each_class:
            for j in range(len(self.set_labels)):
                print("|Action segmentation (cls head, class %d)|%.3f|%.3f|%.3f|%.3f|"%(self.set_labels[j], p[j], r[j], f1[j], acc[j]))
        else:
            print("|Action segmentation (cls head, macro)|%.3f|%.3f|%.3f|%.3f|"%(p, r, f1, acc))
        
    # updated
    def standard_action_segmentation(self):
        
        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, 'eval_results.pkl'), "rb") as f:
            results = pickle.load(f)
        
        preds = None
        gts = None
        input_video_list = []

        for video_id in self.data_list:
            
            gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
            
            length = int(gt_segments[-1, 1])
            # print(gt_segments)
            gt = to_frame_wise(gt_segments, gt_labels, None, length)

            segments = torch.tensor(results[video_id]['segments'])
            labels = torch.tensor(results[video_id]['label'])
            scores = torch.tensor(results[video_id]['score'])

            pred = to_frame_wise(segments, labels, None, length)
            # if len(scores) != len(segments):
            #     pred = to_frame_wise(segments, labels, None, length)
            # else:
            #     pred = to_frame_wise(segments, labels, scores, length)

            input_video = Video(video_id, pred.tolist(), gt.tolist())
            input_video_list.append(input_video)

        ckpt = Checkpoint(bg_class=[-1])
        ckpt.add_videos(input_video_list)
        out = ckpt.compute_metrics()
        
        print("|Action segmentation|IoU:%.1f|edit:%.1f|F1@0.5:%.1f|Acc:%.1f|"%(out['IoU']*100, out['edit']*100, out['F1@0.50']*100, out['acc']*100))

    # updated
    def omission_detection(self):
        
        with open(os.path.join('ckpt', self.args.dataset, self.args.dirname, 'eval_results.pkl'), "rb") as f:
            results = pickle.load(f)

        # all_gt_action_labels = get_gt_from_sqeuence_list(self.args.dataset, self.action2idx)

        all_pred_action_labels = []
        all_gt_action_labels = []
        for video_id in self.data_list:
            gt_segments, gt_labels, gt_label_types, gt_des = self.annotations[video_id]
            labels = torch.tensor(results[video_id]['label'])
            # print('pred', labels.numpy())
            # print('gt', self.step_annotations[video_id])
            all_pred_action_labels.append(labels.numpy())
            all_gt_action_labels.append(self.step_annotations[video_id])

        eval_omission_error(self.args.task, all_pred_action_labels, all_gt_action_labels)

    def evaluate_random_error_detection(self, eval_each_class=False, combine_bg=True, is_visualize=False):
        simplified_error_label_dict = {
            "-1": "Error",
            "0": "BG",
            "1": "Normal",
        }

        gt_labels, gt_error_labels, video_lengths_id, \
            error_descriptions, id2num_dict, num2id_dict = get_gt(self.args.dataset, self.action2idx, self.args.split)
        # convert all the errors to the same class (-1)
        gt_error_labels[gt_error_labels > 1] = -1

        if combine_bg:
            bg_to_normal = [0, 1]
            set_labels = [1, -1]
        else:
            bg_to_normal = None
            set_labels = [0, 1, -1]

        

        # sample 100 times
        num_samples = 100
        p_all, r_all, f1_all, acc_all = None, None, None, None
        for i in range(num_samples):
            labels = np.array([-1, 0, 1])
            all_preds = np.random.choice(labels, size=len(gt_error_labels))
            cp_gt_error_labels = np.copy(gt_error_labels)
            acc, p, r, f1 = acc_precision_recall_f1(all_preds, cp_gt_error_labels, set_labels, eval_each_class, bg_to_normal)
            if p_all is None:
                p_all, r_all, f1_all, acc_all = p, r, f1, acc
            else:
                p_all += p
                r_all += r
                f1_all += f1
                acc_all += acc
        

        p = p_all / num_samples
        r = r_all / num_samples
        f1 = f1_all / num_samples
        acc = acc_all / num_samples

        if eval_each_class:
            for j in range(len(set_labels)):
                print("|Error detection (Random, class %d)|%.3f|%.3f|%.3f|%.3f|"%(set_labels[j], p[j], r[j], f1[j], acc[j]))
        else:
            print("|Error detection (Random, macro)|%.3f|%.3f|%.3f|%.3f|"%(p, r, f1, acc))
        
        if is_visualize:
            start = 0
            end = 0 
            for i in range(len(video_lengths_id)):
                length, video_id = video_lengths_id[i]
                end += length
                pred_vis(gt_error_labels[start:end], all_preds[start:end], simplified_error_label_dict, './error_visualization/rrandom/ed_'+video_id, is_error=True)
                start += length



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', type=str)
    parser.add_argument('--dataset', type=str, default='EgoPER')
    parser.add_argument('--task', type=str, default='pinwheels')
    parser.add_argument('--fps', default=10, type=int)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('-er', '--eval-random', action='store_true')
    parser.add_argument('-as', '--action-segmentation', action='store_true', help='Evaluate action segmentation using cls head')
    parser.add_argument('-ed', '--error-detection', action='store_true')
    parser.add_argument('-od', '--omission-detection', action='store_true', help='always with flag --error')
    parser.add_argument('--threshold', default=-100.0, type=float, help='If set to 0.0, plot the curve and fine the best threshold')
    parser.add_argument('-vis', '--visualize', action='store_true')
    
    args = parser.parse_args()

    evaluator = ActionSegmentationErrorDetectionEvaluator(args)

    if args.eval_random:
        evaluator.evaluate_random_error_detection(eval_each_class=True, combine_bg=True)
        evaluator.evaluate_random_error_detection(eval_each_class=False, combine_bg=True, is_visualize=args.visualize)
    
    if args.action_segmentation:
        evaluator.micro_framewise_action_segmentation(eval_each_class=True)
        evaluator.micro_framewise_action_segmentation(eval_each_class=False, is_visualize=args.visualize)
        evaluator.standard_action_segmentation()

    if args.error_detection:
        if args.threshold != -100.0:
            evaluator.micro_framewise_error_detection(threshold=args.threshold, eval_each_class=False, combine_bg=True, is_visualize=args.visualize)
            evaluator.macro_framewise_error_detection(threshold=args.threshold, eval_each_class=False, combine_bg=True, is_visualize=False)
            evaluator.macro_segment_error_detection(threshold=args.threshold, combine_bg=True, is_visualize=False)
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
            # for i in range(-20, 31):
                threshold = i / 10
                thresholds.append(threshold)
                error_micro_list = evaluator.micro_framewise_error_detection(output_list=error_micro_list, threshold=threshold, eval_each_class=False, combine_bg=True, is_visualize=args.visualize)
                error_macro_list = evaluator.macro_framewise_error_detection(output_list=error_macro_list, threshold=threshold, eval_each_class=False, combine_bg=True)
                eda_list = evaluator.macro_segment_error_detection(output_list=eda_list, threshold=threshold, combine_bg=True)

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

            micro_auc_value = auc(micro_fprs, micro_tprs)
            macro_auc_value = auc(macro_fprs, macro_tprs)

            print('|%s|EDA: %.1f|Micro AUC: %.1f|Macro AUC: %.1f|'%(args.dirname, np.array(eda_list).mean() * 100, micro_auc_value * 100, macro_auc_value * 100))

    if args.omission_detection:
        evaluator.omission_detection()
