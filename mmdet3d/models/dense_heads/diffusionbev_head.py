from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


from mmdet3d.models.builder import HEADS
from mmrotate.core import rbbox2roi
from mmrotate.models.roi_heads import OrientedStandardRoIHead




@HEADS.register_module()
class DiffusionBEVHead(OrientedStandardRoIHead):

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            
        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def forward_train(self, 
                      x,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      t=None):
        if self.with_bbox:
            num_pts = len(proposal_list)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_pts)]
            sampling_results = []
            for i in range(num_pts):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels)
            losses.update(bbox_results['loss_bbox'])

        return losses, bbox_results