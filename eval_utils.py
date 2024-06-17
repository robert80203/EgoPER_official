import numpy as np
import copy
import json
import Levenshtein
from collections import defaultdict
from actseg_src.eval import IoU

def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
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
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends

def mstcn_edit_score(pred, gt, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(pred, bg_class)
    Y, _, _ = get_labels_start_end_time(gt, bg_class)
    return levenstein(P, Y, norm)

def mstcn_f_score(pred_segs, gt_segs, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(pred_segs, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(gt_segs, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    per_action_stats = defaultdict(lambda: np.array([0, 0, 0]))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
            per_action_stats[p_label[j]][0] += 1
        else:
            fp += 1
            per_action_stats[p_label[j]][1] += 1

    fn = len(y_label) - sum(hits)

    for j, h in enumerate(hits):
        if h == 0:
            per_action_stats[y_label[j]][2] += 1

    return float(tp), float(fp), float(fn), per_action_stats


class Video():

    def __init__(self, vname='', pred=[], gt=[]):
        self.vname = vname
        self.pred_label = pred
        self.gt_label = gt

    def __str__(self):
        return "< Video %s >" % self.vname

    def __repr__(self):
        return "< Video %s >" % self.vname

class Checkpoint():

    def __init__(self, iteration=-1, bg_class=[0]):

        # self.rslt_file = None
        self.iteration = iteration
        self.metrics = None
        self.videos = {}
        self.bg_class = bg_class

    def add_videos(self, videos):
        for v in videos:
            self.videos[v.vname] = v

    def __str__(self):
        return "< Checkpoint[%d] %d videos >" % (self.iteration, len(self.videos))

    def __repr__(self):
        return str(self)

    def single_video_loc_metrics(self, v):

        if not hasattr(v, 'metrics'):
            v.metrics = {}

        # pred_label = v.pred_label = expand_pred_to_gt_len(v.pred, len(v.gt_label))
        assert len(v.pred_label) == len(v.gt_label)
        pred_label = v.pred_label 

        m = IoU(self.bg_class)
        m.add(v.gt_label, v.pred_label)
        v.metrics['IoU'] = m.summary()

        # v.metrics['edit'] = metrics2.mstcn_edit_score(pred_segs, gt_segs, bg_class=self.bg_class) / 100
        v.metrics['edit'] = mstcn_edit_score(v.pred_label, v.gt_label, bg_class=self.bg_class) / 100

        tp1, fp1, fn1, pas = mstcn_f_score(
                    v.pred_label, v.gt_label, 0.5, bg_class=self.bg_class)
        precision = tp1 / float(tp1+fp1)
        recall = tp1 / float(tp1+fn1)
        if precision+recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * (precision*recall) / (precision+recall)
        f1 = np.nan_to_num(f1)
        
        v.metrics['F1@0.50'] = f1


    def joint_video_acc(self, video_list):
        gt_list = [v.gt_label for v in video_list]
        pred_list = [v.pred_label for v in video_list]
        gt_ = np.concatenate(gt_list)
        pred_ = np.concatenate(pred_list)

        correct = (gt_ == pred_)
        fg_loc = np.array([ True if g not in self.bg_class else False for g in gt_ ])
        acc = correct[fg_loc].mean()
        return acc

    def compute_metrics(self):
        for vname, video in self.videos.items():
            video.metrics = {}
            self.single_video_loc_metrics(video)

        metric_keys = video.metrics.keys()
        metrics = { k: np.mean([ v.metrics[k] for v in self.videos.values() ])  
                            for k in metric_keys }

        acc = self.joint_video_acc(list(self.videos.values()))
        metrics['acc'] = acc

        self.metrics = metrics

        return self.metrics



class Graph:

	# Constructor
	def __init__(self, edges, N):

		# A List of Lists to represent an adjacency list
		self.adjList = [[] for _ in range(N)]

		# stores in-degree of a vertex
		# initialize in-degree of each vertex by 0
		self.indegree = [0] * N

		# add edges to the undirected graph
		for (src, dest) in edges:

			# add an edge from source to destination
			self.adjList[src].append(dest)

			# increment in-degree of destination vertex by 1
			self.indegree[dest] = self.indegree[dest] + 1

# all topological orderings of a given DAG
def findAllTopologicalOrders(graph, path, discovered, N, path_list):

    # do for every vertex
    for v in range(N):

        # proceed only if in-degree of current node is 0 and
        # current node is not processed yet
        if graph.indegree[v] == 0 and not discovered[v]:

            # for every adjacent vertex u of v, 
            # reduce in-degree of u by 1
            for u in graph.adjList[v]:
                graph.indegree[u] = graph.indegree[u] - 1

            # include current node in the path 
            # and mark it as discovered
            path.append(v)
            discovered[v] = True

            # recur
            findAllTopologicalOrders(graph, path, discovered, N, path_list)

            # backtrack: reset in-degree 
            # information for the current node
            for u in graph.adjList[v]:
                graph.indegree[u] = graph.indegree[u] + 1

            # backtrack: remove current node from the path and
            # mark it as undiscovered
            path.pop()
            discovered[v] = False

    # print the topological order if 
    # all vertices are included in the path
    if len(path) == N:
        path_list.append(copy.deepcopy(path))

# Print all topological orderings of a given DAG
def printAllTopologicalOrders(graph):

    # get number of nodes in the graph
    N = len(graph.adjList)

    # create an auxiliary space to keep track of whether vertex is discovered
    discovered = [False] * N

    # list to store the topological order
    path = []
    path_list = []

    # find all topological ordering and print them
    findAllTopologicalOrders(graph, path, discovered, N, path_list)

    # print('global_path', path_list)

    return path_list

def computeIoU_acc(s_gt, s_pred, s_graph, last):
    
    union_graph_pred = (set(s_graph) - set([0, last])) | (set(s_pred) - set([0, last]))

    omitted_pred = union_graph_pred - (set(s_pred) - set([0, last]))
    omitted_gt = set(s_gt) - set([0, last])
    union = omitted_pred | omitted_gt
    intersection = omitted_pred & omitted_gt
    
    # print('omitted prediction', omitted_pred)
    # print('GT', omitted_gt)

    if len(intersection) == 0 and len(union) == 0:
        iou = 1.0
    elif len(intersection) != 0 and len(union) == 0:
        iou = 0.0
    else:
        iou = len(intersection) / len(union)
    if len(intersection) == 0 and len(omitted_gt) == 0:
        acc = 1.0
    elif len(intersection) != 0 and len(omitted_gt) == 0:
        acc = 0.0
    else:
        acc = len(intersection) / len(omitted_gt)

    return iou, acc

def eval_omission_error(dataset_name, s_preds, s_gts):
    print('===============Dataset:', dataset_name)
    if dataset_name == 'tea':
        edges = [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (0, 3), (3, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
        N = 12
        last = 11
    elif dataset_name == 'pinwheels':
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14)]
        N = 15
        last = 14
    elif dataset_name == 'oatmeal':
        edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12)]
        N = 13
        last = 12
    elif dataset_name == 'quesadilla':
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6), (6, 7), (7, 8), (8, 9)]
        N = 10
        last = 9
    elif dataset_name == 'coffee':
        edges = [(0, 1), (1, 2), (2, 13), (0, 5), (5, 13), (0, 6), (6, 7), (7, 8), (8, 12), (0, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 3), (3, 4), (4, 16)]
        N = 17
        last = 16

    # use GT task graph
    graph = Graph(edges, N)
    global_paths = printAllTopologicalOrders(graph)
    
    nonrepeat_IoU_list = []
    acc_list = []
    ordering_IoU_list = []
    edit_score_list = []
    video_id = 1
    for s_pred, s_gt in zip(s_preds, s_gts):
        min_dist = -1
        best_path = None
        idx = 0
        new_s_pred = [0]
        new_s_gt = [0]
        for s in s_pred:
            if s != 0:
                new_s_pred.append(s)
        for s in s_gt:
            if s != 0:
                new_s_gt.append(s)

        new_s_pred.append(last)
        new_s_gt.append(last)
        
        for global_path in global_paths:
            #dist = editDistance(global_path, new_s_pred, len(global_path), len(new_s_pred))
            dist = Levenshtein.distance(global_path, new_s_pred)
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                best_path = global_path
            idx += 1

        nonrepeat_IoU, acc = computeIoU_acc(new_s_gt, new_s_pred, best_path, last=last)
        nonrepeat_IoU_list.append(nonrepeat_IoU)
        acc_list.append(acc)
        video_id += 1
    print('Nonrepeat IoU:', sum(nonrepeat_IoU_list) / len(nonrepeat_IoU_list))
    print('Acc:', sum(acc_list) / len(acc_list))

########## usage
# v1 = Video('v1')
# v1.gt_label = [1,2,3]
# v1.pred_label = [1,2,3]
#
# ckpt = Checkpoint(bg_class=[0])
# ckpt.add_videos([v1])
# ckpt.compute_metrics()