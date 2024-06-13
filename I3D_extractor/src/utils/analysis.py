# from ast import parse
import numpy as np
from collections import defaultdict, Counter, OrderedDict
import os
import pickle
import gzip
from .utils import expand_frame_label, parse_label 
from . import metrics
import logging
import json
from sklearn.metrics import precision_recall_fscore_support
# from glob import glob
# from ..configs.utils import generate_expname, hiedict2cfg, load_cfg_json



class Video():
    
    def __init__(self, vname=''):
        self.vname = vname
    
    def __str__(self):
        return "< Video %s >" % self.vname

    def __repr__(self):
        return "< Video %s >" % self.vname

class Checkpoint():
    """
    for a checkpoint, 
    firstly load out construct a Video object for each video,
    then compute statistics, such as performance,
    """

    __VERSION__ = 1.0
    __DATE__    = "5-13"

    def __init__(self, iteration, bg_class=[ 0 ]):

        self.iteration = iteration
        self.videos = {}

        self.__version__ = Checkpoint.__VERSION__
        self.__date__ = Checkpoint.__DATE__

        self.bg_class = bg_class

    def add_videos(self, videos: list):
        for v in videos:
            self.videos[v.vname] = v

    def drop_videos(self):
        self.videos = {}

    def set_bg_class(self, classes):
        assert isinstance(classes, list)
        self.bg_class = classes

    @staticmethod
    def load(fname):
        with gzip.open(fname, 'rb') as fp:
            ckpt = pickle.load(fp)
            if not ckpt.__version__ == Checkpoint.__VERSION__:
                logging.warning("old version checkpoint found %s" % ckpt.__version__)
                logging.warning(fname)
        return ckpt
    
    def save(self, fname):
        self.fname = fname
        with gzip.open(fname, 'wb') as fp:
            pickle.dump(self, fp)

    def __str__(self):
        return "< Checkpoint[%d] %d videos >" % (self.iteration, len(self.videos))

    def __repr__(self):
        return str(self)

    def _random_video(self):
        vnames = list(self.videos.keys())
        vname = np.random.choice(vnames, 1).item()
        return vname, self.videos[vname]

    def average_losses(self):
        losses = [v.loss for v in self.videos.values()]
        self.loss = easy_reduce(losses, mode='mean')

    def _localization_metrics(self, gt_label, pred_label, slip_label, slip_pred, lapse_label, lapse_pred):

        M = OrderedDict()

        pred_segs = parse_label(pred_label)
        gt_segs = parse_label(gt_label)

        Aiou, Aiou_ = metrics.aligned_IoU(gt_label, pred_label, bg_id=self.bg_class[0])
        M['AIoU'] = Aiou
        M['AIoU#'] = Aiou_

        M["slip_Amof"] = (slip_pred == slip_label).mean()
        ml = slip_label == 0
        mp = slip_pred == 0
        true_start = np.where(ml)[0].min() if ml.sum() > 0 else ml.size
        pred_start = np.where(mp)[0].min() if mp.sum() > 0 else mp.size
        M["slip_dist"] = np.abs(true_start-pred_start) / slip_label.size

        M["lapse_Amof"] = (lapse_pred == lapse_label).mean()

        M['edit'] = metrics.mstcn_edit_score(pred_segs, gt_segs, bg_class=self.bg_class) # HACK
        if np.isnan(M['edit']):
            M['edit'] = 0

        pred_aset = np.unique(pred_label)
        true_aset = np.unique(gt_label)
        num_hit = [ p for p in pred_aset if p in true_aset ]
        M['action_recall'] = len(num_hit) / len(true_aset)

        M['num_segs'] = len(pred_segs)


        return M

    def _f1(self, gt_list, pred_list):
        gt_ = np.concatenate(gt_list)
        pred_ = np.concatenate(pred_list)

        gt_ = 1 - gt_
        pred_ = 1 - pred_

        p, r, f, s = precision_recall_fscore_support(gt_, pred_, average='binary', zero_division=0)
        return p, r, f, s

    def _dataset_level_localization_metrics(self, gt_list, pred_list):
        M = OrderedDict()

        # Dmof
        gt_ = np.concatenate(gt_list)
        pred_ = np.concatenate(pred_list)

        correct = (gt_ == pred_)
        fg_loc = np.array([ True if g not in self.bg_class else False for g in gt_ ])
        M['Dmof'] = correct.mean()
        M['Dmof#'] = correct[fg_loc].mean()

        num_bg = sum([ 1 for p in pred_ if p in self.bg_class ])
        M["bg_freq"] = num_bg / len(pred_)

        # F Score
        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        for gt, pred in zip(gt_list, pred_list):
            gt_segs = parse_label(gt)
            pred_segs = parse_label(pred)
            for s in range(len(overlap)):
                tp1, fp1, fn1 = metrics.mstcn_f_score(pred_segs, 
                                gt_segs, overlap[s], bg_class=self.bg_class)
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1
                
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])
        
            f1 = 2.0 * (precision*recall) / max(precision+recall, 1e-10)

            f1 = np.nan_to_num(f1) #*100
            M['F1@%0.2f' % overlap[s]] = f1

        return M

    def compute_metrics(self):
        """
        compute metrics of all videos in the ckpt
        NOTE: video must have a `pred` and a `gt_label` attribute
        """
        gt_list, pred_list = [], []
        for vname, video in self.videos.items():
            video.gt_segs = parse_label(video.gt_label)
            if video.pred.size < video.gt_label.size:
                video.pred_label = expand_frame_label(video.pred, len(video.gt_label))
                video.lapse_pred_label = expand_frame_label(video.lapse_pred, len(video.gt_label))
            else:
                video.pred_label = video.pred
                video.lapse_pred_label = video.lapse_pred
            video.pred_segs = parse_label(video.pred_label)
            video.slip_pred_label = video.slip_pred

            video.metrics = self._localization_metrics(video.gt_label, video.pred_label, 
                                                        video.slip_label, video.slip_pred_label, 
                                                        video.lapse_label, video.lapse_pred_label)
            gt_list.append(video.gt_label)
            pred_list.append(video.pred_label)

        metric_list = [ video.metrics for video in self.videos.values() ]
        self.metrics = easy_reduce(metric_list, skip_nan=True) # for AIoU#

        vnames = list(self.videos.keys())
        slip_label = [ self.videos[v].slip_label for v in vnames ]
        slip_pred = [ self.videos[v].slip_pred_label for v in vnames ]
        p, r, f, s = self._f1(slip_label, slip_pred)
        self.metrics['slip_Pre'] = p
        self.metrics['slip_Rec'] = r
        self.metrics['slip_F1']  = f

        lapse_label = [ self.videos[v].lapse_label for v in vnames ]
        lapse_pred = [ self.videos[v].lapse_pred_label for v in vnames ]
        p, r, f, s = self._f1(lapse_label, lapse_pred)
        self.metrics['lapse_Pre'] = p
        self.metrics['lapse_Rec'] = r
        self.metrics['lapse_F1']  = f

        m = self._dataset_level_localization_metrics(gt_list, pred_list)
        self.metrics.update(m)

class Experiment():

    __VERSION__ = 1.0
    __DATE__    = "12-20" # date is not synced to version, but last update time

    def __init__(self, expfdr_dict, expgroup=None):
        self.expgroup = expgroup  
        self.expfdr_dict = expfdr_dict

        self.splits = list(expfdr_dict.keys())
        self.splits.sort()

        self.run_cfg = None

        self.run_iterations = {} # { split: run: {} }
        self.run_ckpts = {}

    def __str__(self):
        string = '< ExpGroup[%s] split: %s >'
        split_str = ''
        for s in self.splits:
            split_str += '%s-' % s
            split_str += '[%s] ' % ','.join(map(str, self.expfdr_dict[s].keys()))
        split_str = split_str[:-1] # remove last whitespace
        string = string % (self.expgroup, split_str)
        return string

    def __repr__(self):
        return self.__str__()

    def add_test_ckpt(self, split:int, run:int, ckpt):
        if split not in self.run_iterations:
            self.run_iterations[split] = {}
            self.run_ckpts[split] = {}
        if run not in self.run_iterations[split]:
            self.run_iterations[split][run] = []
            self.run_ckpts[split][run] = {}

        iteration = ckpt.iteration
        self.run_iterations[split][run].append(iteration)
        self.run_ckpts[split][run][iteration] = ckpt
        self.run_iterations[split][run].sort()

    def _get_best_ckpt(self, metric, iterations, ckpts, patience=0.0):
        """
        patience: ignore the first X percent of test iterations
        """
        iterations.sort()
        if metric == "latest":
            return ckpts[iterations[-1]]
        else:
            start_step = int( max(iterations) * patience )
            iterations = [i for i in iterations if i >=start_step]
            scores = [ ckpts[i].metrics[metric] for i in iterations ]
            i = np.argmax(scores)
            ckpt = ckpts[iterations[i]]
            return ckpt

    def get_best_test_ckpt(self, split, run, metric: str, patience: float=0.0) -> Checkpoint :
        """
        get the test metrics with the highest score of `metric`.
        """
        return self._get_best_ckpt(metric, 
            self.run_iterations[split][run], 
            self.run_ckpts[split][run], 
            patience=patience)

    def get_best_of_each_metric(self, split, run, patience=0.0):
        """
        get the highest score for each metric from all test ckpts
        """
        metrics = {}
        mlist = list(self.run_ckpts[split][run].values())[0].metrics.keys()
        for m in mlist:
            metrics[m] = self.get_best_test_ckpt(split, run, m, patience=patience).metrics[m]
        return metrics

    def get_test_metric_history(self, split, run, metric):
        return np.array([ self.run_ckpts[split][run][i].metrics[metric] 
                                for i in self.run_iterations[split][run] ])

    def get_id_of_best_run(self, split, metric, patience=0.0):
        runs = self.run_ckpts[split]
        scores =  [ self.get_best_test_ckpt(split, r, metric, patience=patience) for r in runs ] 
        idx = np.argmax(scores).item()
        return runs[idx]
    
    def average_metrice(self, func, splits=None, mode="mean"):
        """
        pass in a function to compute a metric on a run of a split
        collect the score for all exp and average over runs and splits

        func: 
            input: exp object, split, run,
            output: a score or a list/dict/tuple of scores
        
        mode: mean   - mean over runs in the same split
              max    - max  over runs in the same split
              median - median over runs in the same split
              metric_name - use it to select the best run and return results
              callable function - customized avarege

        return : a final score and a dict containing each run's score
        """

        if splits is None:
            splits = list(self.run_ckpts.keys())

        metrics_complete = {}
        split_metrics = []
        for split in splits:
            all_metrics = []
            mdict = {}
            for run in self.run_ckpts[split]:
                score = func(self, split, run)
                mdict[run] = score
                all_metrics.append(score)
            
            metrics_complete[split] = mdict

            if callable(mode):
                aggregate = mode(all_metrics)
            elif mode in ["mean", "max", "median"]:
                aggregate = intelligent_aggregrate(all_metrics, mode=mode)
            else: # mode should be a metric name and func should return a dict
                assert isinstance(all_metrics[0], dict)
                select_metric = [ m[mode] for m in all_metrics ]
                best_idx = np.argmax(select_metric)
                aggregate = all_metrics[best_idx]
            split_metrics.append(aggregate)

        final_score = intelligent_aggregrate(split_metrics, mode='mean')
        return final_score, metrics_complete

    def load_ckpts(self, splits=None, metrics_only=True):
        """
        split: load all splits or one of the split
        """
        def _load_ckpt(fname):
            with gzip.open(fname, 'rb') as fp:
                ckpt = pickle.load(fp)
            if not ckpt.__version__ == Checkpoint.__VERSION__:
                logging.warning("old version checkpoint found %s" % ckpt.__version__)
                logging.warning(f)
            return ckpt
        
        def _load_metric(ckpt_fname):
            metric_fname = ckpt_fname[:-2] + 'json'
            if os.path.exists(metric_fname):
                with open(metric_fname) as fp:
                    metrics = json.load(fp)
                iteration = int(os.path.basename(ckpt_fname)[:-3])
                ckpt = Checkpoint(iteration)
                ckpt.metrics = metrics
            else:
                ckpt = _load_ckpt(ckpt_fname)
                try:
                    with open(metric_fname, 'w') as fp:
                        json.dump(ckpt.metrics, fp)
                except Exception as e:
                    # print(metric_fname)
                    # print(e)
                    pass
            return ckpt

        if splits is None:
            splits = self.splits
        elif isinstance(splits, int):
            splits = [ splits ]

        for split in splits:
            self.run_iterations[split] = {}
            self.run_ckpts[split] = {}

            for runid, fdr in self.expfdr_dict[split].items():
                rsltdir = os.path.join(fdr, 'saves')
                ckpt_files = os.listdir(rsltdir)
                ckpt_files = [ os.path.join(rsltdir, fname) for fname in ckpt_files if fname.endswith('.gz') ]

                tmp = {}
                for f in ckpt_files:
                    ckpt = _load_metric(f) if metrics_only else _load_ckpt(f)
                    tmp[ckpt.iteration] = ckpt

                iterations = list(tmp.keys())
                iterations.sort()

                ckpts = OrderedDict()
                for i in iterations:
                    ckpts[i] = tmp[i]

                self.run_iterations[split][runid] = iterations
                self.run_ckpts[split][runid] = ckpts


def intelligent_aggregrate(scores, mode="mean", skip_nan=False):
    assert isinstance(scores, list), type(scores)
    if isinstance(scores[0], float) or isinstance(scores[0], int) or (isinstance(scores[0], np.ndarray) and len(scores[0].shape)==0):
        if skip_nan:
            scores = [ s for s in scores if not np.isnan(s) ]
        if mode == "mean":
            average = np.mean(scores)
        elif mode == "max":
            average = np.max(scores)
        elif mode == "median":
            average = np.median(scores)
    elif isinstance(scores[0], list):
        average = []
        L = len(scores[0])
        for i in range(L):
            average.append( intelligent_aggregrate([s[i] for s in scores ], mode=mode) )
    elif isinstance(scores[0], np.ndarray):
        assert len(scores[0].shape) == 1
        stack = np.stack(scores, axis=0)
        average = stack.mean(0)
    elif isinstance(scores[0], tuple):
        average = []
        L = len(scores[0])
        for i in range(L):
            average.append( intelligent_aggregrate([s[i] for s in scores ], mode=mode) )
        average = tuple(average)
    elif isinstance(scores[0], dict):
        average = {}
        for k in scores[0]:
            average[k] = intelligent_aggregrate([s[k] for s in scores], mode=mode)
    else:
        raise TypeError("Unsupport Data Type %s" % type(scores[0]) )

    return average

easy_reduce = intelligent_aggregrate # HACK


