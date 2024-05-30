import yaml


DEFAULTS = {
    # random seed for reproducibility, a large number is preferred
    "init_rand_seed": 0,
    # dataset loader, specify the dataset here
    "dataset_name": "EgoPER",
    "devices": ['cuda:3'], # default: single gpu
    "train_split": 'training',
    "val_split": 'validation',
    "model_name": "LocPointTransformer",
    "dataset": {
        "default_fps": 10,
        # number of classes
        "num_classes": 97,
        # max sequence length during training
        "max_seq_len": 2304,
        # threshold for truncating an action
        "trunc_thresh": 0.5,
        "crop_ratio": None,
        # height of frame
        "height": 720,
        # width of frame
        "width": 1280,
        # ratio of BG segments
        "background_ratio": 0.3,
        # num of graph node
        "num_node": 20,
        "use_gcn": False,
        "task": 'coffee',
    },
    "loader": {
        "batch_size": 2,
        "num_workers": 4,
    },
    # network architecture
    "model": {
        # type of backbone (convTransformer | conv)
        "backbone_type": 'convTransformer',
        # type of FPN (fpn | identity)
        "fpn_type": "identity",
        "backbone_arch": (2, 2, 5),
        # scale factor between pyramid levels
        "scale_factor": 2,
        # regression range for pyramid levels
        "regression_range": [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        # number of heads in self-attention
        "n_head": 4,
        # window size for self attention; <=1 to use full seq (ie global attention)
        "n_mha_win_size": -1,
        # kernel size for embedding network
        "embd_kernel_size": 3,
        # I3D feature
        "input_dim": 2048,
        # (output) feature dim for embedding network
        "embd_dim": 512,
        # if attach group norm to embedding network
        "embd_with_ln": True,
        # feat dim for FPN
        "fpn_dim": 512,
        # if add ln at the end of fpn outputs
        "fpn_with_ln": True,
        # starting level for fpn
        "fpn_start_level": 0,
        # feat dim for head
        "head_dim": 512,
        # kernel size for reg/cls/center heads
        "head_kernel_size": 3,
        # number of layers in the head (including the final one)
        "head_num_layers": 3,
        # if attach group norm to heads
        "head_with_ln": True,
        # defines the max length of the buffered points
        "max_buffer_len_factor": 6.0,
        # disable abs position encoding (added to input embedding)
        "use_abs_pe": False,
        # use rel position encoding (added to self-attention)
        "use_rel_pe": False,
        # if > 0, we include background segment during training and inference
        # "num_bg_clusters": 1, # always include background
        # if > 1, we use k-means to generate multiple normal prototypes
        "num_normal_clusters": 1,
    },
    "train_cfg": {
        # radius | none (if to use center sampling)
        "center_sample": "radius",
        "center_sample_radius": 1.5,
        "loss_weight": 1.0, # on reg_loss, use -1 to enable auto balancing
        "cls_prior_prob": 0.01,
        "init_loss_norm": 2000,
        # gradient cliping, not needed for pre-LN transformer
        "clip_grad_l2norm": -1,
        # cls head without data (a fix to epic-kitchens / thumos)
        "head_empty_cls": [],
        # dropout ratios for tranformers
        "dropout": 0.0,
        # ratio for drop path
        "droppath": 0.1,
        # if to use label smoothing (>0.0)
        "label_smoothing": 0.0,
        "contrastive": False,
        # "pretrain_contrastive": False,
        # "cl_level": 1,
        "cl_weight": 0.1,
        # "clsemb_drop_rate": 0.0,
        # "cl_clip_thres": 1000.0, # if larger than 1, this parameter means nothing
        "num_negative_segments": 2,
        # "use_equal_focal": False,
        # "use_smooth_loss": False,
    },
    "test_cfg": {
        "pre_nms_thresh": 0.001,
        "pre_nms_topk": 5000,
        "iou_threshold": 0.1,
        "min_score": 0.01,
        "max_seg_num": 1000,
        "nms_method": 'soft', # soft | hard | none
        "nms_sigma" : 0.5,
        "duration_thresh": 0.05,
        "multiclass_nms": True,
        "ext_score_file": None,
        "voting_thresh" : 0.75,
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "AdamW", # SGD or AdamW
        # solver params
        "momentum": 0.9,
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
        # excluding the warmup epochs
        "epochs": 30,
        # lr scheduler: cosine / multistep
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    }
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_default_config():
    config = DEFAULTS
    return config

def _update_config(config):
    # fill in derived fields
    # config["model"]["input_dim"] = config["dataset"]["input_dim"]
    config["model"]["num_node"] = config["dataset"]["num_node"]
    config["model"]["num_classes"] = config["dataset"]["num_classes"]
    config["model"]["max_seq_len"] = config["dataset"]["max_seq_len"]
    config["model"]["use_gcn"] = config["dataset"]["use_gcn"]
    # config["model"]["num_bg_clusters"] = config["dataset"]["num_bg_clusters"]
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    return config

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    config = _update_config(config)
    return config