import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def to_numpy(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    else:
        return x

class Segment():
    def __init__(self, action, start, end):
        assert start >= 0
        self.action = action
        self.start = start
        self.end = end
        self.len = end - start + 1
    
    def __repr__(self):
        return "<%r %d-%d>" % (self.action, self.start, self.end)
    
    def intersect(self, s2):
        s = max([self.start, s2.start])
        e = min([self.end, s2.end])
        return max(0, e-s+1)

    def union(self, s2):
        s = min([self.start, s2.start])
        e = max([self.end, s2.end])
        return e-s+1
    
def parse_label(label_list):
    """
    convert framewise labels into segments
    """
    current = label_list[0]
    start = 0
    anno = []
    for i, l in enumerate(label_list):
        if l == current:
            pass
        else:
            anno.append(Segment(current, start, i-1))
            current = l
            start = i
    anno.append(Segment(current,start, len(label_list)-1))
    
    return anno

def expand_frame_label(label, target_len: int):
    if len(label) == target_len:
        return label

    import torch

    is_numpy = isinstance(label, np.ndarray)
    if is_numpy:
        label = torch.from_numpy(label).float()

    label = label.view([1, 1, -1])
    resized = torch.nn.functional.interpolate(
        label, size=target_len, mode="nearest"
    ).view(-1)
    resized = resized.long()
    
    if is_numpy:
        resized = resized.detach().numpy()

    return resized

def shrink_frame_label(label: list, clip_len: int) -> list:
    num_clip = ((len(label) - 1) // clip_len) + 1
    new_label = []
    for i in range(num_clip):
        s = i * clip_len
        e = s + clip_len
        l = label[s:e]
        ct = Counter(l)
        l = ct.most_common()[0][0]
        new_label.append(l)

    return new_label

##########################################################################
def compute_barh(label, begin=0):
    N = len(label)
    current_state = None
    start = -1
    state = []
    for i, l in enumerate(label):
        if l != current_state:
            if i != 0:
                state.append([current_state, start, i])
            start = i
            current_state = l
    state.append([current_state, start, i])
    trans = [ s[0] for s in state ]

    plotx = [ [ begin + s[1]/N, (s[2]-s[1])/N ] for s in state ]
    return trans, plotx


def visualize_prediction(*segments, index2label=None, plot=True, bg_token="SIL", 
                                    seg_label=None, title=None, figsize=None, legend=True):
    from matplotlib.lines import Line2D
    from collections.abc import Iterable
    # L = [ len(s) for s in segments ]
    # assert np.std(L) == 0

    if figsize:
        fig = plt.figure('visualize_prediction', figsize=figsize)

    ncol_legend = 4
    if index2label is not None:
        bg_token = 0
        ncol_legend = 2

    states = set()
    for s in segments:
        unique = set(s)
        states = states | unique

    states = list(states)
    if bg_token in states:
        states.pop(states.index(bg_token))
        states.insert(0, bg_token)

    mapdit = {}
    for i, s in enumerate(states):
        if not isinstance(s, str) and s < 0: # negative number means ignored segment/frame
            mapdit[s] = -1
        else:
            mapdit[s] = i


    cm = plt.cm.tab20 if len(states) > 10 else plt.cm.tab10
    def colormap(x):
        def _element_convert(t):
            return cm(t) if t >= 0 else [0.0, 0.0, 0.0, 1] # for ignored segment/frame use black color

        if isinstance(x, Iterable):
            return [ _element_convert(e) for e in x ]
        else:
            return _element_convert(x)
        
    
    for i, s in enumerate(segments):
        trans, plot_x = compute_barh(s)
        remap_trans = [ mapdit[a] for a in trans ]
        color = colormap(remap_trans)
        plt.broken_barh(plot_x, [-0.3*i, 0.2], facecolor=color)
        if seg_label is not None:
            plt.text(-0.03, -0.3*i+0.1, seg_label[i], ha='right', fontsize="large")

    if legend:
        color=[ colormap(mapdit[s]) for s in states ]
        lines = [ Line2D([0], [0], color=color[i], lw=4) for i, a in enumerate(states) ]
        if index2label is not None:
            ticks = [ index2label[s] for s in states ]
        else:
            ticks = states
        plt.legend(lines, ticks, 
                   frameon=False, loc='upper center', ncol=ncol_legend, bbox_to_anchor=[0.5, -0.0], fontsize="large")  
    plt.axis("off")
    if title is not None:
        plt.title(title)
    if plot:
        plt.show()

def create_groundtruth_prob(gt_label, num_classes):
    label = np.zeros( [len(gt_label), num_classes] )
    label[ np.arange(label.shape[0]), gt_label ] = 1
    return label

def plot_probability(prob_list, label_list, index2label=None, title=None, plot=True, legend=True):
    if len(label_list) == 1 and len(label_list) != len(prob_list) :
        label_list = label_list * len(prob_list)
    assert len(prob_list) == len(label_list)

    actions = []
    for label in label_list:
        actions.extend(np.unique(label).tolist())
        # actions.extend(np.where(label)[0].tolist())
    actions = list(set(actions))
    actions.sort()
    index2rel = {j:i for i, j in enumerate(actions)}
    if index2label is None:
        index2label={i:str(i) for i in actions}

    nrows = len(prob_list)
    fig, axes = plt.subplots(nrows=nrows, sharex=False, figsize=[7, 2.5*nrows+0.5])

    if nrows == 1:
        axes = [axes]
    
    for i, att in enumerate(prob_list):
        action = np.unique(label_list[i])
        for a in action:
            axes[i].plot(att[:, a], c=plt.cm.tab20(index2rel[a]), label=index2label[a])
    if legend:
        axes[0].legend()

    if title is not None:
        plt.suptitle(title)

    if plot:
        plt.show()
    else:
        return fig

def easy_prob_plot(video_object):
    from scipy.special import softmax
    C = video_object.action_logit.shape[1]
    gtp = create_groundtruth_prob(video_object.gt_label, C)
    pdp = create_groundtruth_prob(video_object.pred_label, C)
    prob = softmax(video_object.action_logit, axis=1)
    plot_probability([gtp, pdp, prob], [video_object.gt_label])


#==================================================

def modify_actions(train_label, nmodify, nclasses, hide_bg=False):
    segs = parse_label(train_label)
    if hide_bg:
        segs = [ s for s in segs if s.action != 0 ]
    transcript = [s.action for s in segs] 
    T = len(train_label)

    slip_label = np.array([1]*T)
    modify_loc = np.random.choice(len(transcript), nmodify, replace=False)
    modify_loc = np.sort(modify_loc)
    for n in modify_loc:
        slip_label[segs[n].start:segs[n].end+1] = 0
    for i in range(nmodify):
        n = modify_loc[i]
        if np.random.rand() > 0.5: # delete
            transcript.pop(n)
            modify_loc = modify_loc - 1
        else: # swap
            c = transcript[n]
            while c == transcript[n]:
                if hide_bg:
                    c = np.random.choice(nclasses-1)+1 # do not choose bg
                else:
                    c = np.random.choice(nclasses)
            transcript[n] = c

    return transcript, slip_label

def modify_actions_v2(train_label, nmodify, nclasses, hide_bg=False, generate_seglabel=False):

    def check_same(transcript, c, n):
        same_as_before = (transcript[n-1] == c) if n > 0 else False
        same_as_n = transcript[n] == c
        same_as_after = (transcript[n+1] == c) if n < len(transcript) - 1 else False

        r = same_as_before or same_as_n or same_as_after
        return r

    segs = parse_label(train_label)
    if hide_bg:
        segs = [ s for s in segs if s.action != 0 ]
    transcript = [s.action for s in segs] 
    T = len(train_label)

    slip_label = np.array([1]*T)
    modify_loc = np.random.choice(len(transcript), nmodify, replace=False)
    modify_loc = np.sort(modify_loc)
    for n in modify_loc:
        slip_label[segs[n].start:segs[n].end+1] = 0

    for i in range(nmodify):
        n = modify_loc[i]
        if np.random.rand() > 0.5: # delete
            transcript.pop(n)
            segs.pop(n)
            modify_loc = modify_loc - 1
        else: # swap
            c = transcript[n]
            while check_same(transcript, c, n):
                if hide_bg:
                    c = np.random.choice(nclasses-1)+1 # do not choose bg
                else:
                    c = np.random.choice(nclasses)
            transcript[n] = c
            segs[n] = None

    if not generate_seglabel:
        return transcript, slip_label
    
    seglabel = np.zeros([len(train_label), len(segs)], dtype=np.int)
    for i, s in enumerate(segs):
        if s is None:
            continue
        seglabel[s.start:s.end+1, i] = 1
    return transcript, slip_label, seglabel


