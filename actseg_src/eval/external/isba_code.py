# All of the below code were copied from:
# https://github.com/Zephyr-D/TCFPN-ISBA/blob/3794c0684689fa68a815bc99f3c99c9d4f4cacc7/utils/metrics.py
# very small modifications have been made:
# namely bg_class can be a list of classes now.
# also removed the "* 100"

import numpy as np


def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    return Yi_split


def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
    return intervals


def IoD(P, Y, bg_class=None):
    # From ICRA paper:
    # Learning Convolutional Action Primitives for Fine-grained Action Recognition
    # Colin Lea, Rene Vidal, Greg Hager
    # ICRA 2016

    def overlap_d(p, y, bg_class):
        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        if bg_class is not None:
            true_intervals = np.array(
                [t for t, l in zip(true_intervals, true_labels) if l not in bg_class]
            )
            true_labels = np.array([l for l in true_labels if l not in bg_class])
            pred_intervals = np.array(
                [t for t, l in zip(pred_intervals, pred_labels) if l not in bg_class]
            )
            pred_labels = np.array([l for l in pred_labels if l not in bg_class])

        n_true_segs = true_labels.shape[0]
        n_pred_segs = pred_labels.shape[0]
        seg_scores = np.zeros(n_true_segs, float)

        for i in range(n_true_segs):
            for j in range(n_pred_segs):
                if true_labels[i] == pred_labels[j]:
                    intersection = min(
                        pred_intervals[j][1], true_intervals[i][1]
                    ) - max(pred_intervals[j][0], true_intervals[i][0])
                    union = pred_intervals[j][1] - pred_intervals[j][0]
                    score_ = float(intersection) / union
                    seg_scores[i] = max(seg_scores[i], score_)

        return seg_scores.mean()

    if type(P) == list:
        return np.mean([overlap_d(P[i], Y[i], bg_class) for i in range(len(P))])
    else:
        return overlap_d(P, Y, bg_class)


def IoU(P, Y, bg_class=None):
    # From ICRA paper:
    # Learning Convolutional Action Primitives for Fine-grained Action Recognition
    # Colin Lea, Rene Vidal, Greg Hager
    # ICRA 2016

    def overlap_(p, y, bg_class):
        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        if bg_class is not None:
            true_intervals = np.array(
                [t for t, l in zip(true_intervals, true_labels) if l not in bg_class]
            )
            true_labels = np.array([l for l in true_labels if l not in bg_class])
            pred_intervals = np.array(
                [t for t, l in zip(pred_intervals, pred_labels) if l not in bg_class]
            )
            pred_labels = np.array([l for l in pred_labels if l not in bg_class])

        n_true_segs = true_labels.shape[0]
        n_pred_segs = pred_labels.shape[0]
        seg_scores = np.zeros(n_true_segs, float)

        for i in range(n_true_segs):
            for j in range(n_pred_segs):
                if true_labels[i] == pred_labels[j]:
                    intersection = min(
                        pred_intervals[j][1], true_intervals[i][1]
                    ) - max(pred_intervals[j][0], true_intervals[i][0])
                    union = max(pred_intervals[j][1], true_intervals[i][1]) - min(
                        pred_intervals[j][0], true_intervals[i][0]
                    )
                    score_ = float(intersection) / union
                    seg_scores[i] = max(seg_scores[i], score_)

        return seg_scores.mean()

    if type(P) == list:
        return np.mean([overlap_(P[i], Y[i], bg_class) for i in range(len(P))])
    else:
        return overlap_(P, Y, bg_class)
