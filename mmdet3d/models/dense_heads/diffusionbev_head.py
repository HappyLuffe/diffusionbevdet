from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch


from mmdet3d.models.builder import HEADS
from mmrotate.core import rbbox2roi
from mmrotate.models.roi_heads import OrientedStandardRoIHead
from mmrotate.core import rbbox2result

from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init

import numpy as np


@HEADS.register_module()
class DiffusionBEVHead(nn.Module):
    def __init__(self,
                 fuse_img=False,
                 num_views=0,
                 in_channels_img=64,
                 out_size_factor_img=4,
                 num_proposals=128,
                 auxiliary=True,
                 in_channels=256,
                 hidden_channel=128,
                 num_classes=4,
                 # config for Transformer
                 num_decoder_layers=3,
                 num_heads=8,
                 learnable_query_pos=False,
                 initialize_by_heatmap=False,
                 nms_kernel_size=1,
                 ffn_channel=256,
                 dropout=0.1,
                 bn_momentum=0.1,
                 activation='relu',
                 # config for FFN
                 common_heads=dict(),
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_iou=dict(type='VarifocalLoss', use_sigmoid=True, iou_weighted=True, reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,):
        super(DiffusionBEVHead, self).__init__()

        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base


    def forward_train(self, 
                      x,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      t,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        
        """
            x: list[ [BS, C, H, W] ] lenth: num_level_feat
            proposal_list: list[ [num_proposal, 5] ] lenth: BS
            gt_bboxes: list[ [num_gt, 5] ] lenth: BS
            gt_labels: list[ [num_gt] ] lenth: BS
        """

        for i in range(len(proposal_list)):
            bboxes = proposal_list[i].numpy()
            lenth = np.size(bboxes, 0)
            temp = np.full((lenth), i)
            bboxes = np.insert(bboxes, 0, temp, axis=1)
            proposal_list[i] = torch.from_numpy(bboxes)


        # * rois[BS*num_proposal, 6] bbox_feats[BS*num_proposal, 512, 7, 7]
        rois = rbbox2roi(proposal_list)
        query_feat = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        
        batch_size = x.shape[0]
        lidar_feat = self.shared_conv(x)
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)




"""

# if self.with_bbox:
        #     num_pts = len(proposal_list)
        #     if gt_bboxes_ignore is None:
        #         gt_bboxes_ignore = [None for _ in range(num_pts)]
        #     sampling_results = []
        #     for i in range(num_pts):
        #         assign_result = self.bbox_assigner.assign(
        #             proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
        #             gt_labels[i])
        #         sampling_result = self.bbox_sampler.sample(
        #             assign_result,
        #             proposal_list[i],
        #             gt_bboxes[i],
        #             gt_labels[i],
        #             feats=[lvl_feat[i][None] for lvl_feat in x])

        #         if gt_bboxes[i].numel() == 0:
        #             sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
        #                 (0, gt_bboxes[0].size(-1))).zero_()
        #         else:
        #             sampling_result.pos_gt_bboxes = \
        #                 gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

        #         sampling_results.append(sampling_result)

        # losses = dict()
        # # bbox head forward and loss
        # if self.with_bbox:
        #     bbox_results = self._bbox_forward_train(x, sampling_results,
        #                                             gt_bboxes, gt_labels, t)
        #     losses.update(bbox_results['loss_bbox'])

        # return losses
    




    # def _bbox_forward(self, x, rois, t):
        # ""Box head forward function used in both training and testing.

    #     Args:
    #         x (list[Tensor]): list of multi-level img features.
    #         rois (list[Tensors]): list of region of interests.

    #     Returns:
    #         dict[str, Tensor]: a dictionary of bbox_results.
    #     ""
    #     # * rois[BS*num_proposal, 6] bbox_feats[BS*num_proposal, 512, 7, 7]
    #     bbox_feats = self.bbox_roi_extractor(
    #         x[:self.bbox_roi_extractor.num_inputs], rois)
    #     if self.with_shared_head:
    #         bbox_feats = self.shared_head(bbox_feats)
    #     # todo  时间还需要处理
    #     cls_score, bbox_pred = self.bbox_head(bbox_feats, t)

    #     bbox_results = dict(
    #         cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
    #     return bbox_results

    # def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, t):
    #     ""Run forward function and calculate loss for box head in training.

    #     Args:
    #         x (list[Tensor]): list of multi-level img features.
    #         sampling_results (list[Tensor]): list of sampling results.
    #         gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
    #             shape (num_gts, 5) in [cx, cy, w, h, a] format.
    #         gt_labels (list[Tensor]): class indices corresponding to each box
            
    #     Returns:
    #         dict[str, Tensor]: a dictionary of bbox_results.
    #     ""
    #     rois = rbbox2roi([res.bboxes for res in sampling_results])
    #     bbox_results = self._bbox_forward(x, rois, t)

    #     bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
    #                                               gt_labels, self.train_cfg)
    #     loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
    #                                     bbox_results['bbox_pred'], rois,
    #                                     *bbox_targets)

    #     bbox_results.update(loss_bbox=loss_bbox)
    #     return bbox_results

    # def simple_test(self, x, proposal_list, t, rescale=False):
    #     assert self.with_bbox, 'Bbox head must be implemented.'

    #     det_bboxes, det_scores = self.simple_test_bboxes(x, proposal_list, t, self.test_cfg, rescale=rescale)

    #     # bbox_results = [
    #     #     rbbox2result(det_bboxes[i], det_labels[i],
    #     #                  self.bbox_head.num_classes)
    #     #     for i in range(len(det_bboxes))
    #     # ]

    #     # *三个变量都为List类型，长度为batch_size
    #     # *List中的元素大小：
    #     # *label=[num_boxes, 5], score=[num_boxes, ], bbox=[num_boxes, ]
    #     return det_scores, det_bboxes

    # def simple_test_bboxes(self, 
    #                        x, 
    #                        proposals, 
    #                        t,
    #                        test_cfg, 
    #                        rescale=False):
    #     rois = rbbox2roi(proposals)
    #     bbox_results = self._bbox_forward(x, rois, t)

    #     cls_score = bbox_results['cls_score']
    #     bbox_pred = bbox_results['bbox_pred']
    #     num_proposals_per_img = tuple(len(p) for p in proposals)
    #     rois = rois.split(num_proposals_per_img, 0)
    #     cls_score = cls_score.split(num_proposals_per_img, 0)

    #     # some detector with_reg is False, bbox_pred will be None
    #     if bbox_pred is not None:
    #         # the bbox prediction of some detectors like SABL is not Tensor
    #         if isinstance(bbox_pred, torch.Tensor):
    #             bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
    #         else:
    #             bbox_pred = self.bbox_head.bbox_pred_split(
    #                 bbox_pred, num_proposals_per_img)
    #     else:
    #         bbox_pred = (None, ) * len(proposals)

    #     det_bboxes = []
    #     det_scores = []

    #     batch_size = len(proposals)
    #     w, h = 1600, 1408
    #     img_shapes = tuple([w, h])
    #     # scale_factors = tuple([1., 1., 1., 1.] for i in range(batch_size))

    #     for i in range(len(proposals)):
    #         # *bboxes=[n, 5], scores=[n, num_class+1]
    #         det_bbox, det_score = self.bbox_head.get_bboxes(
    #             rois[i],
    #             cls_score[i],
    #             bbox_pred[i],
    #             img_shapes,
    #             rescale=rescale,
    #             cfg=test_cfg)
            
    #         det_bboxes.append(det_bbox)
    #         det_scores.append(det_score)
        
    #     det_bboxes = torch.stack(det_bboxes)
    #     det_scores = torch.stack(det_scores)
    #     return det_bboxes, det_scores

    


"""


        