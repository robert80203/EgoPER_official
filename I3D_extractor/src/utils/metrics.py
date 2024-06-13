import numpy as np
from .utils import parse_label

####################################
# Middpoint Accuracy
def isba_midpoint(gt_seg, pred_seg, bg_id=None):
    unused = np.ones(len(gt_seg), np.bool)
    n_true = len(gt_seg)
    pointer = 0

    TP, FP, FN = 0, 0, 0
    # Go through each segment and check if it's correct.
    for i, pseg in enumerate(pred_seg):
        if bg_id is not None and pseg.action == bg_id:
            continue
        midpoint = int( (pseg.start + pseg.end)/2 )
        # Check each corresponding true segment
        for j in range(pointer, n_true):
            gseg = gt_seg[j]
            # If the midpoint is in this true segment
            if gseg.start <= midpoint <= gseg.end:
                pointer = j
                # If yes and it's correct
                if (gseg.action == pseg.action):
                    # Only a TP if it's the first occurance. Otherwise FP
                    if unused[j]:
                        TP += 1
                        unused[j] = False
                    else:
                        FP += 1
                # FN if it's wrong class
                else:
                    FN += 1
            elif midpoint < gseg.end:
                break

    prec = float(TP) / (TP+FP+1e-4) 
    recall = float(TP) / (TP+FN+1e-4) 
    
    return prec, recall



def compute_IoU_IoD_IoR(gt_label, pred, bg_id=0, as_dict=False):
    assert isinstance(gt_label, np.ndarray), type(gt_label)
    assert isinstance(pred,     np.ndarray), type(pred)

    # unique = set(gt_label.tolist()) | set(pred.tolist())
    # unique = list(unique)
    unique = np.unique(gt_label).tolist()

    iou = []
    iod = []
    ior = []
    iou_noBG = []
    iod_noBG = []
    ior_noBG = []
    for i in unique: # for each action
        recog_mask = pred == i
        gt_mask = gt_label == i
        union = np.logical_or(recog_mask, gt_mask).sum()
        intersect = np.logical_and(recog_mask, gt_mask).sum() # num of correct prediction
        num_recog = recog_mask.sum()
        num_gt = gt_mask.sum() 
        
        action_iou = intersect / (union+ 1e-8) 
        action_iod = intersect / (num_recog + 1e-8)
        action_ior = intersect / (num_gt + 1e-8)

        iou.append(action_iou)
        iod.append(action_iod)
        ior.append(action_ior)
        if i != bg_id:
            iou_noBG.append(action_iou)
            iod_noBG.append(action_iod)
            ior_noBG.append(action_ior)

    iou = np.mean(iou)
    iod = np.mean(iod)
    ior = np.mean(ior)
    iou_noBG = np.mean(iou_noBG)
    iod_noBG = np.mean(iod_noBG)
    ior_noBG = np.mean(ior_noBG)

    if as_dict:
        metrics = {}
        metrics["IoU"] = iou
        metrics["IoU#"] = iou_noBG
        metrics["IoD"] = iod
        metrics["IoD#"] = iod_noBG
        metrics['IoR'] = ior
        metrics['IoR#'] = ior_noBG
        return metrics
    else:
        return iou, iod, ior, iou_noBG, iod_noBG, ior_noBG

def compute_Dmof_IoX(label_list, pred_list, bg_id=0, as_dict=True):
    LABEL = np.concatenate(label_list)
    PRED = np.concatenate(pred_list)

    fg_mask = LABEL != bg_id
    correct = LABEL == PRED

    mof, mof_nobg = correct.mean(), correct[fg_mask].mean()

    all_iox = []
    for label, pred in zip(label_list, pred_list):
        iou_iod_ior = compute_IoU_IoD_IoR(label, pred, bg_id=bg_id, as_dict=False)
        all_iox.append(iou_iod_ior)

    iou = np.mean( [ iou_iod[0] for iou_iod in all_iox ] )
    iod = np.mean( [ iou_iod[1] for iou_iod in all_iox ] )
    ior = np.mean( [ iou_iod[2] for iou_iod in all_iox ] )
    iou_nobg = np.mean( [ iou_iod[3] for iou_iod in all_iox ] )
    iod_nobg = np.mean( [ iou_iod[4] for iou_iod in all_iox ] )
    ior_nobg = np.mean( [ iou_iod[5] for iou_iod in all_iox ] )

    if as_dict:
        return {
            "Dmof": mof,
            "Dmof#": mof_nobg,
            "IoU": iou,
            "IoU#": iou_nobg,
            "IoD" : iod,
            "IoD#" : iod_nobg,
            "IoR"  : ior,
            "IoR#" : ior_nobg,
        }
    else:
        return mof, mof_nobg, iou, iou_nobg, iod, iod_nobg, ior, ior_nobg


def aligned_IoU(gt_label, pred_label, bg_id=0, intermediate_result=False):
    from scipy.optimize import linear_sum_assignment

    gt_segs = parse_label(gt_label)
    pred_segs = parse_label(pred_label)
    matrix = np.zeros([len(gt_segs), len(pred_segs)])
    
    for i, gseg in enumerate(gt_segs):
        for j, pseg in enumerate(pred_segs):
            if gseg.action != pseg.action:
                continue
            if gseg.end <= pseg.start or pseg.end <= gseg.start:
                continue
            intersect = min(gseg.end, pseg.end) - max(gseg.start, pseg.start)
            union     = max(gseg.end, pseg.end) - max(gseg.start, pseg.start)
            iou = intersect / union if union > 0 else 0
            matrix[i, j] = iou
            
    rid, cid = linear_sum_assignment(-matrix)
    
    gt_new = np.zeros_like(gt_label) - 1
    pd_new = np.zeros_like(pred_label) - 2
    for r, c in zip(rid, cid):
        if matrix[r, c] == 0:
            continue
        gseg = gt_segs[r]
        gt_new[gseg.start:gseg.end+1] = gseg.action
        
        pseg = pred_segs[c]
        pd_new[pseg.start:pseg.end+1] = pseg.action

    
    unique = set(gt_label.tolist()) | set(pred_label.tolist())
    unique = list(unique)
    
    iou = []
    iou_noBG = []
    for i in unique: # for each action
        gt_mask = gt_label == i
        union = np.logical_or(pred_label==i, gt_mask).sum()
        intersect = np.logical_and(pd_new==i, gt_mask).sum() # num of correct prediction
        action_iou = intersect / union if union > 0 else 0

        iou.append(action_iou)
        if i != bg_id:
            iou_noBG.append(action_iou)
            
    iou = np.mean(iou)
    iou_noBG = np.mean(iou_noBG)        
    if np.isnan(iou_noBG):
        iou_noBG = 0
    
    if not intermediate_result:
        return iou, iou_noBG
    else:
        return iou, iou_noBG, [matrix, rid, cid, gt_new, pd_new]


##############################################
## edit distance 

def edit_distance(a, b):
    distance_dict = {}
    la, lb = len(a), len(b)
    return _dist_helper(a, b, la, lb, distance_dict)
    
def _dist_helper(a, b, la, lb, distance_dict):
    if (la, lb) not in distance_dict:
        if la == 0:
            score = lb
        elif lb == 0:
            score = la
        else:
            if a[la-1] == b[lb-1]: # match and continue to the next location
                score = _dist_helper(a, b, la-1, lb-1, distance_dict)
            else:
                score = min([
                    _dist_helper(a, b, la-1, lb, distance_dict) + 1,  # delete
                    _dist_helper(a, b, la, lb-1, distance_dict) + 1,  # insert
                    _dist_helper(a, b, la-1, lb-1, distance_dict) + 1, # replace
                ])  # insert and delete are the same in essential

        distance_dict[(la, lb)] = score
        
    return distance_dict[(la, lb)]

#################################################
## MSTCN Metrics

def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) #* 100
    else:
        score = D[-1, -1]

    return score    

def segs_to_labels_start_end_time(seg_list, bg_class):
    seg_list = [ s for s in seg_list if s.action not in bg_class ]
    labels = [ p.action for p in seg_list ]
    start  = [ p.start for p in seg_list ]
    end    = [ p.end+1 for p in seg_list ]
    return labels, start, end

def mstcn_edit_score(pred_segs, gt_segs, norm=True, bg_class=["background"]):
    P, _, _ = segs_to_labels_start_end_time(pred_segs, bg_class)
    Y, _, _ = segs_to_labels_start_end_time(gt_segs, bg_class)

    return levenstein(P, Y, norm)

def mstcn_f_score(pred_segs, gt_segs, overlap, bg_class=["background"]):
    p_label, p_start, p_end = segs_to_labels_start_end_time(pred_segs, bg_class)
    y_label, y_start, y_end = segs_to_labels_start_end_time(gt_segs, bg_class)

    if len(y_label) == 0: # HACK
        return 0, 0, 0 

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)
