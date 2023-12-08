import math
import numpy
import torch
import random
from torch import nn
from torch.nn import functional as F
from torch_kmeans import KMeans
from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss
from ..utils import batched_nms


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )
        # fpn_masks remains the same
        return out_logits


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )
        


    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), )
        # fpn_masks remains the same
        return out_offsets


@register_meta_arch("LocPointTransformer_CSPL_GCN")
class PtTransformer_CSPL_GCN(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines #layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        fpn_start_level,       # start level of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        num_classes,           # number of action classes
        num_normal_clusters,   # number of normal step clusters
        use_gcn,               # use AOD
        num_node,
        train_cfg,             # other cfg for training
        test_cfg               # other cfg for testing
    ):
        super().__init__()
         # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(
            fpn_start_level, backbone_arch[-1]+1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*(1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            #print(stride)
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']
        self.cl_weight = train_cfg['cl_weight']
        self.num_negative_segments = train_cfg['num_negative_segments']
        self.use_contrastive = train_cfg['contrastive']
        # self.cl_fpn_strides = [self.fpn_strides[i] for i in range(train_cfg['cl_level'])]
        self.tao_bias = nn.Parameter(torch.tensor(0.0))
        self.pre_cl_loss = None
        ##################################################
        # background setting
        # self.num_bg_clusters = num_bg_clusters
        # if num_bg_clusters > 1:
        #     self.bg_cluster_model = KMeans(n_clusters=self.num_bg_clusters,verbose=False)
        # # normal class setting
        # self.num_normal_clusters = num_normal_clusters
        # if num_normal_clusters > 1:
        #     self.normal_cluster_models = {}
        #     for i in range(1, num_classes):
        #         self.normal_cluster_models[str(i)] = KMeans(n_clusters=num_normal_clusters, verbose=False)
        
        self.num_normal_clusters = num_normal_clusters
        if num_normal_clusters > 1:
            self.normal_cluster_models = {}
            for i in range(num_classes):
                self.normal_cluster_models[str(i)] = KMeans(n_clusters=num_normal_clusters, verbose=False)
        ##################################################
        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        self.use_gcn = use_gcn
        assert backbone_type in ['convGCNTransformer', 'convTransformer']
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                backbone_type,
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch' : backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe,
                }
            )
        else:
            self.backbone = make_backbone(
                backbone_type,
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch' : backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe,
                    'num_node' : num_node,
                }
            )
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.fpn_type = fpn_type
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel' : fpn_dim,
                'scale_factor' : scale_factor,
                'start_level' : fpn_start_level,
                'with_ln' : fpn_with_ln
            }
        )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : max_seq_len * max_buffer_len_factor,
                'fpn_strides' : self.fpn_strides,
                'regression_range' : self.reg_range
            }
        )

        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

        self.fpn_dim = fpn_dim
        self.head_dim = head_dim
        self.head_kernel_size = head_kernel_size
        self.head_with_ln = head_with_ln
        self.head_num_layers = head_num_layers
        self.train_cfg = train_cfg
        self.cosine_sim = nn.CosineSimilarity(dim=0, eps=1e-6)


    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]
    
    def update_head(self, num_classes=14):
        self.cls_head = PtTransformerClsHead(
            self.fpn_dim, self.head_dim, num_classes,
            kernel_size=self.head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=self.head_with_ln,
            num_layers=self.head_num_layers,
            empty_cls=self.train_cfg['head_empty_cls']
        )

    def init_prototypes(self):
        self.prototypes = {}
        self.threshold_dict = {}
        for cls_idx in range(self.num_classes):
            # self.prototypes[str(cls_idx)] = [None for i in range(len(self.cl_fpn_strides))]
            self.prototypes[str(cls_idx)] = None
            self.threshold_dict[str(cls_idx)] = None

    def flush_prototypes(self):
        for key, value in self.prototypes.items():
            if value is None:
                continue
            if self.num_normal_clusters > 1:
                frame_feat = value.permute(1, 0)
                feat_list = frame_feat.unsqueeze(0)
                normal_labels = self.normal_cluster_models[key].fit_predict(feat_list)
                normal_labels = normal_labels.squeeze(0)
                normal_clusters = []
                for i in range(self.num_normal_clusters):
                    selected_frame_feat = frame_feat[normal_labels == i].permute(1, 0)
                    normal_clusters.append(selected_frame_feat.mean(dim=1, keepdim=True))
                self.prototypes[key] = normal_clusters
            else:
                self.prototypes[key] = value.mean(dim=1, keepdim=True)    
    

    def generate_prototypes(self, fpn_feats, segments, labels):
        for i in range(len(segments)):
            for j in range(len(segments[i])):
                start = int(segments[i][j, 0])
                end = int(segments[i][j, 1])
                label = labels[i][j].item()
                if str(label) in self.prototypes:
                    if start >= end: # prevent nan problem (only happen when using predicted boundary)
                        continue

                    feats = fpn_feats[0][i, :, start:end]
                    
                    if self.prototypes[str(label)] is None:
                        self.prototypes[str(label)] = feats.detach()
                    else:
                        self.prototypes[str(label)] = torch.cat((self.prototypes[str(label)], feats.detach()), dim=1)

    def get_thresholds(self, fpn_feats, segments, labels):
        for i in range(len(segments)):
            output_labels = []
            output_sim = []
            for j in range(len(segments[i])):
                start = int(segments[i][j, 0])
                end = int(segments[i][j, 1])
                label = labels[i][j].item()

                if start == end: # prevent nan problem (only happen when using predicted boundary)
                    continue
                
                feats = fpn_feats[0][i, :, start:end]


                if self.prototypes[str(label)] is None:
                    continue
                
                if self.num_normal_clusters > 1:
                    best_sims_mean = -1 #0
                    best_sims = None
                    best_std = None
                    for normal_cluster_idx in range(self.num_normal_clusters):
                        sims = self.cosine_sim(self.prototypes[str(label)][normal_cluster_idx], feats)
                        if sims.mean() > best_sims_mean:
                            best_sims_mean = sims.mean()
                            best_sims = sims
                            best_std = sims.std()
                    sims = best_sims
                    std = best_std
                else:
                    sims = self.cosine_sim(self.prototypes[str(label)], feats)
                
                if self.threshold_dict[str(label)] is None:
                    self.threshold_dict[str(label)] = sims
                else:
                    self.threshold_dict[str(label)] = torch.cat((self.threshold_dict[str(label)], sims), dim=0)

    # compute the similarity only with the prototype of predictied class
    def compute_similarity(self, fpn_feats, segments, labels, video_id, threshold=0.5):
        b_output = []
        for i in range(len(segments)):
            output_labels = []
            output_sim = []
            for j in range(len(segments[i])):
                start = int(segments[i][j, 0])
                end = int(segments[i][j, 1])
                sim_list_pyramid = [] # shape: (pyramid level, keys of the classes, frame similarities)
                label = labels[i][j].item()
                if start >= end: # prevent nan problem (only happen when using predicted boundary)
                    continue
                
                feats = fpn_feats[0][i, :, start:end]

                # if prototype does not exist, then assign the segment to BG
                if self.prototypes[str(label)] is None:
                    output_labels.append(-1)
                    output_sim.append(0)
                    continue
                
                if self.num_normal_clusters > 1:
                    best_sims_mean = -1 #0
                    best_sims = None
                    for normal_cluster_idx in range(self.num_normal_clusters):
                        sims = self.cosine_sim(self.prototypes[str(label)][normal_cluster_idx], feats)
                        if sims.mean() > best_sims_mean:
                            best_sims_mean = sims.mean()
                            best_sims = sims
                    sims = best_sims
                else:
                    sims = self.cosine_sim(self.prototypes[str(label)], feats)
                
                # pre-computed threshold with mean and std
                value = self.threshold_dict[str(label)]
                if value is None:
                    thres = (threshold + 2.0) / 4.0
                else:
                    std = value.std(0)
                    thres = value.mean(0) + threshold * std
                
                thres_cond = sims < thres
                action_list = torch.ones(sims.size()).long()
                action_list[thres_cond] = -1
                
                # majority voting for seen classes
                num_sample = torch.sum(action_list == 1)
                sim_mean = torch.mean(sims[action_list == 1])
                best_num_sample = num_sample
                best_action = str(label)
                best_similarity = sim_mean
                
                # majority voting for unseen class
                num_sample = torch.sum(action_list == -1)
                if num_sample > best_num_sample:
                    best_action = '-1'
                    best_similarity = 0 

                output_labels.append(int(best_action))
                output_sim.append(best_similarity)
                

            b_output.append(
                {
                    'video_id': video_id[i],
                    'segments': segments[i].cpu(), 
                    'labels': torch.tensor(output_labels),
                    'scores': torch.tensor(output_sim)
                }
            )
        return b_output


    def forward(self, video_list, mode = 'none', threshold=0.5):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        
        batched_ofs, batched_bboxes, batched_bbox_classes, \
            batched_edge_maps, batched_inputs, batched_masks = self.preprocessing_gcn(video_list)


        feats, masks = self.backbone(batched_ofs, batched_bboxes, batched_bbox_classes,
                                    batched_edge_maps, batched_inputs, batched_masks)
        
        fpn_feats, fpn_masks = self.neck(feats, masks)

        if mode == 'clustering':
            b_segments = []
            b_labels = []
            for i in range(len(video_list)):
                b_segments.append(video_list[i]['segments'])
                b_labels.append(video_list[i]['labels'])
            self.generate_prototypes(fpn_feats, b_segments, b_labels)
            return
        elif mode == 'clustering_flush':
            self.flush_prototypes()
            return 
        elif mode == 'clustering_init':
            self.init_prototypes()
            return 
        elif 'similarity' in mode:
            b_segments = []
            b_labels = []
            b_video_id = []
            for i in range(len(video_list)):
                b_segments.append(video_list[i]['segments'])
                b_labels.append(video_list[i]['labels'])
                b_video_id.append(video_list[i]['video_id'])
            b_output = self.compute_similarity(fpn_feats, b_segments, b_labels, b_video_id, threshold)
            return b_output
        elif 'get_thresholds' in mode:
            b_segments = []
            b_labels = []
            for i in range(len(video_list)):
                b_segments.append(video_list[i]['segments'])
                b_labels.append(video_list[i]['labels'])
            self.get_thresholds(fpn_feats, b_segments, b_labels)
            return 

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)
        
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels)
            # compute the loss and return
            losses = self.losses(
                fpn_feats, fpn_masks, video_list,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets
            )

            return losses

        else:
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets
            )
            return results

    @torch.no_grad()
    def preprocessing_gcn(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        ################################
        # graph input
        ################################
        if 'bbox' not in video_list[0]:
            bboxes = None
        else:
            bboxes = [x['bbox'].permute(1, 2, 0) for x in video_list]
        if 'bbox_class' not in video_list[0]:
            bbox_classes = None
        else:
            bbox_classes = [x['bbox_class'].permute(1, 0) for x in video_list]
        if 'edge_map' not in video_list[0]:
            edge_maps = None
        else:
            edge_maps = [x['edge_map'].permute(1, 2, 0) for x in video_list]

        if 'of' not in video_list[0]:
            ofs = None
        else:
            ofs = [x['of'].permute(1, 0) for x in video_list]
        
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            # an empty batch with padding
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            # refill the batch
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
            
            # batch bbox shape B, num_node, 4, T
            if bboxes is not None:
                batch_bboxes_shape = [len(bboxes), bboxes[0].shape[0], bboxes[0].shape[1], max_len]
                batched_bboxes = bboxes[0].new_full(batch_bboxes_shape, padding_val)
                for bbox, pad_bbox in zip(bboxes, batched_bboxes):
                    pad_bbox[..., :bbox.shape[-1]].copy_(bbox)
            else:
                batched_bboxes = None
            
            # batch bbox_class shape B, num_node, T
            if bbox_classes is not None:
                batch_bbox_classes_shape = [len(bbox_classes), bbox_classes[0].shape[0], max_len]
                batched_bbox_classes = bbox_classes[0].new_full(batch_bbox_classes_shape, padding_val)
                for bbox_class, pad_bbox_class in zip(bbox_classes, batched_bbox_classes):
                    pad_bbox_class[..., :bbox_class.shape[-1]].copy_(bbox_class)
            else:
                batched_bbox_classes = None
            
            # batch edge map shape B, num_node, num_node, T
            if edge_maps is not None:
                batch_edge_maps_shape = [len(edge_maps), edge_maps[0].shape[0], edge_maps[0].shape[1], max_len]
                batched_edge_maps = edge_maps[0].new_full(batch_edge_maps_shape, padding_val)
                for edge_map, pad_edge_map in zip(edge_maps, batched_edge_maps):
                    pad_edge_map[..., :edge_map.shape[-1]].copy_(edge_map)
            else:
                batched_edge_maps = None
                
            # batch of shape B, num_node, T
            if ofs is not None:
                batch_ofs_shape = [len(ofs), ofs[0].shape[0], max_len]
                batched_ofs = ofs[0].new_full(batch_ofs_shape, padding_val)
                for of, pad_of in zip(ofs, batched_ofs):
                    pad_of[..., :of.shape[-1]].copy_(of)
            else:
                batched_ofs = None
           
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            padding_bboxes_size = [0, max_len - feats_lens[0]]
            padding_bbox_classes_size = [0, max_len - feats_lens[0]]
            padding_edge_maps_size = [0, max_len - feats_lens[0]]
            padding_ofs_size = [0, max_len - feats_lens[0]]

            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

            if bboxes is not None:
                batched_bboxes = F.pad(
                    bboxes[0], padding_bboxes_size, value=padding_val).unsqueeze(0)
            else:
                batched_bboxes = None

            if bbox_classes is not None:
                batched_bbox_classes = F.pad(
                    bbox_classes[0], padding_bbox_classes_size, value=padding_val).unsqueeze(0)
            else:
                batched_bbox_classes = None

            if edge_maps is not None:
                batched_edge_maps = F.pad(
                    edge_maps[0], padding_edge_maps_size, value=padding_val).unsqueeze(0)
            else:
                batched_edge_maps = None
            
            if ofs is not None:
                batched_ofs = F.pad(
                    ofs[0], padding_ofs_size, value=padding_val).unsqueeze(0)
            else:
                batched_ofs = None
        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        if batched_bboxes is not None:
            batched_bboxes = batched_bboxes.to(self.device)
        if batched_bbox_classes is not None:
            batched_bbox_classes = batched_bbox_classes.to(self.device)
        if batched_edge_maps is not None:
            batched_edge_maps = batched_edge_maps.to(self.device)
        if batched_ofs is not None:
            batched_ofs = batched_ofs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        # return batched_bboxes, batched_bbox_classes, batched_edge_maps, batched_inputs, batched_masks
        return batched_ofs, batched_bboxes, batched_bbox_classes, batched_edge_maps, batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset


    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            # print('cernter', center_pts, center_pts.size())
            # print('concat', concat_points[:, 3, None], concat_points[:, 3, None].size())
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets


    def losses(
        self, fpn_feats, fpn_masks, video_list,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets
    ):     
        if self.use_contrastive:
            cl_loss = self.contrastive_loss(fpn_feats, video_list)
        else:
            cl_loss = torch.tensor(0.0)

        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # focal loss
        #print(gt_target, gt_target.size())

        # print(out_cls_logits[0].size(), out_cls_logits[1].size())
        # cls_loss = sigmoid_focal_loss(
        #     torch.cat(out_cls_logits, dim=1)[valid_mask],
        #     gt_target,
        #     reduction='sum',
        # )

        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum',
        )

        # if self.use_smooth_loss:
        #     sm_loss = smooth_loss(out_cls_logits, valid_mask)
        # else:
        #     sm_loss = torch.tensor(0.0)

        
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        
        final_loss = cls_loss + reg_loss * loss_weight + self.cl_weight * cl_loss
        
        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                'cl_loss'    : cl_loss,
                'final_loss' : final_loss}

    def contrastive_loss(self, fpn_feats, video_list):
        cl_loss = 0
        num_negative_samples = self.num_negative_segments

        for i in range(len(video_list)):
            num_segments = len(video_list[i]['segments'])
            num_sample = num_segments
            # non-repeative sampling
            sample_segments = random.sample(range(num_segments), num_sample)
            
            # traverse segments
            for j in sample_segments:
                p_start = int(video_list[i]['segments'][j][0])
                p_end = int(video_list[i]['segments'][j][1])

                p_label = video_list[i]['labels'][j].item()
                n_start_list = []
                n_end_list = []
                n_label_list = []
                n_v_idx_list = []
                
                # select negative segments across batch
                while len(n_start_list) < num_negative_samples:
                    n_v_idx = random.randint(0, len(video_list)-1)
                    n_idx = random.randint(0, len(video_list[n_v_idx]['labels'])-1)
                    n_label = video_list[n_v_idx]['labels'][n_idx].item()
                    n_start = int(video_list[n_v_idx]['segments'][n_idx][0])
                    n_end = int(video_list[n_v_idx]['segments'][n_idx][1])
                    
                    # the selected segment cannot be the same label as positive one and start time != end time
                    if n_label != p_label and n_start != n_end:
                        n_start_list.append(n_start)
                        n_end_list.append(n_end)
                        n_label_list.append(n_label)
                        n_v_idx_list.append(n_v_idx)
                
                # traverse every level of fpn
                numerator = None
                denominator = None
                
                # avoid nan problem
                if p_start >= p_end:
                    continue
                
                pos_feats = fpn_feats[0][i, :, p_start:p_end]
                
                # compute positive scores
                if self.prototypes[str(p_label)] is None:
                    continue

                if self.num_normal_clusters > 1:
                    best_sims_mean = -1
                    best_sims = None
                    best_normal_cluster_idx = -1
                    for normal_cluster_idx in range(self.num_normal_clusters):
                        sims = self.cosine_sim(self.prototypes[str(p_label)][normal_cluster_idx], pos_feats)
                        if sims.mean() > best_sims_mean:
                            best_sims_mean = sims.mean()
                            best_sims = sims
                            best_normal_cluster_idx = normal_cluster_idx
                    pos_scores = best_sims
                else:
                    pos_scores = self.cosine_sim(self.prototypes[str(p_label)], pos_feats)

                if numerator is None:
                    numerator = torch.exp(pos_scores / F.relu(self.tao_bias + 1.0))
                else:
                    numerator = torch.cat((numerator, torch.exp(pos_scores / F.relu(self.tao_bias + 1.0))), dim=0)
                
                # compute negative scores
                for l in range(len(n_start_list)):
                    n_start = n_start_list[l]
                    n_end = n_end_list[l]
                    
                    n_label = n_label_list[l]
                    n_v_idx = n_v_idx_list[l]

                    # avoid nan problem
                    if n_start == n_end:
                        continue

                    neg_feats = fpn_feats[0][n_v_idx, :, n_start:n_end]
                    

                    if self.num_normal_clusters > 1:
                        neg_scores = self.cosine_sim(self.prototypes[str(p_label)][best_normal_cluster_idx], neg_feats)
                    else:
                        neg_scores = self.cosine_sim(self.prototypes[str(p_label)], neg_feats)
                    
                    if denominator is None:
                        denominator = torch.exp(neg_scores / (F.relu(self.tao_bias) + 1.0))
                    else:
                        denominator = torch.cat((denominator, torch.exp(neg_scores / (F.relu(self.tao_bias) + 1.0))), dim=0)
                # infoNCE loss
                cl_loss += torch.log(numerator.sum(dim=0) / (numerator.sum(dim=0) + denominator.sum(dim=0)))
        
        if self.pre_cl_loss == None:
            self.pre_cl_loss = torch.abs(cl_loss)
            factor = 1.0
        else:
            factor = torch.abs(self.pre_cl_loss - torch.abs(cl_loss)) / self.pre_cl_loss
            factor = torch.clamp(factor, min=1.0).detach()
            self.pre_cl_loss = torch.abs(cl_loss)

        return - cl_loss / factor

    @torch.no_grad()
    def inference(
        self,
        video_list,
        points, fpn_masks,
        out_cls_logits, out_offsets
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        # vid_ft_stride = [x['feat_stride'] for x in video_list]
        # vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        # for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
        #     zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        # ):
        for idx, (vidx, fps, vlen) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid
            )
            
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            # results_per_vid['feat_stride'] = stride
            # results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []
        org_scores_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
            ):
            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()
            # 1004
            org_scores_all.append(pred_prob)

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs =  torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])
        
        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all,
                   'scores_all': org_scores_all} # 1004

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            # stride = results_per_vid['feat_stride']
            # nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            # 1004
            scores_all = results_per_vid['scores_all']
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms = (self.test_nms_method == 'soft'),
                    multiclass = self.test_multiclass_nms,
                    sigma = self.test_nms_sigma,
                    voting_thresh = self.test_voting_thresh
                )

                # 3: convert from feature grids to seconds
                if segs.shape[0] > 0:
                    # segs = (segs * stride + 0.5 * nframes) / fps
                    segs = segs / fps
                    # truncate all boundaries within [0, duration]
                    segs[segs<=0.0] *= 0.0
                    segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen

            else: # just return the results of original length
                num_frames = int(vlen * fps)
                #print(segs.size(), scores.size(), labels.size(), num_frames)
                segs = segs[:num_frames,:]
                scores = scores[:num_frames]
                labels = labels[:num_frames]
            
            # 4: repack the results
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels'   : labels,
                 'scores_all': scores_all} #1004
            )

        return processed_results
