import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector
from .mvx_two_stage import MVXTwoStageDetector
import math
import numpy as np
from collections import namedtuple
from mmcv.ops.nms import batched_nms

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def lidabev2img(bboxes):
    bev_bboxes = bboxes

    batch_size = len(bev_bboxes)
    for i in range(batch_size):
        bev_bbox = bev_bboxes[i].cpu().numpy()
        lenth = np.size(bev_bbox, 0)
        for j in range(lenth):
            w, h = bev_bbox[j][2], bev_bbox[j][3]
            x = 1600 - bev_bbox[j][1]
            y = 1408 - bev_bbox[j][0]
            bev_bbox[j][:4] = np.asarray([x, y, h, w])
        bev_bboxes[i] = torch.from_numpy(bev_bbox).cuda()

    return bev_bboxes

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)

@DETECTORS.register_module()
class DiffusionBEVDetector(MVXTwoStageDetector):
    def __init__(self, **kwargs):
        super(DiffusionBEVDetector, self).__init__(**kwargs)
        # self.voxel_layer = Voxelization((**voxel_layer))
        timesteps = 1000
        sampling_timesteps = 1 # 后续需要改为在配置文件中说明

        self.device = torch.device('cuda')

        self.num_proposals = 200
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.scale = 2.0

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.box_renewal = True
        self.use_ensemble = True
        self.use_focal = True
        self.use_fed_loss = False
        self.use_nms = True
        

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        x_start = x_start.cuda()

        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape).type(torch.float32)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape).type(torch.float32)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = x_boxes * images_whwh[:, None, :]        

         # *outputs_class=[bs, num_boxes, 5], outputs_score=[bs, num_boxes, ], outputs_bbox=[bs, num_boxes, ]
        outputs_class, outputs_score, outputs_coord = self.pts_bbox_head.simple_test(backbone_feats, x_boxes, t)

        x_start = outputs_coord
        x_start = x_start / images_whwh[:, None, :]
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_score, outputs_coord

        


    @torch.no_grad()
    def ddim_sample(self, backbone_feats, images_whwh, points, clip_denoised=True, do_postprocess=True):
        w, h = 1600, 1408
        batch = points.shape[0]
        images_whwh = torch.tensor([w, h, w, h, 2 * math.pi]).repeat(batch, 1)

        shape = (batch, self.num_proposals, 5)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # *[-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # *[(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # *初始噪声框
        img = torch.randn(shape, device=self.device)

        # ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_scores, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img, time_cond, self_cond, clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:
                score_per_image, box_per_image = outputs_class[0], outputs_coord[0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]

            if time_next < 0:
                img = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            # img_sizes = torch.tensor([w, h]).repeat(batch,1)
            if self.box_renewal:
                # *补充盒子数量
                img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)

        # todo 还需要将二维BEV的bbox转换为三维的bbox
        # todo 需要对outputs_coord进行处理，加上高度数据
        bbox_results = [
            bbox3d2result(outputs_coord[i], outputs_scores[i], outputs_class[i])
            for i in range(batch)
        ]
        return bbox_results
        
            
    # def inference(self, box_cls, box_pred, image_sizes):
    #     assert len(box_cls) == len(image_sizes)
    #     results = []
    #     if self.use_focal or self.use_fed_loss:
    #         scores = torch.sigmoid(box_cls)
    #         labels = torch.arange(self.num_classes, device=self.device). \
    #             unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

    #         for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
    #                 scores, box_pred, image_sizes
    #         )):
    #             # result = Instances(image_size)
    #             scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
    #             labels_per_image = labels[topk_indices]
    #             box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
    #             box_pred_per_image = box_pred_per_image[topk_indices]

    #             if self.use_ensemble and self.sampling_timesteps > 1:
    #                 return box_pred_per_image, scores_per_image, labels_per_image

    #             if self.use_nms:
    #                 _, keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, dict(type='nms', iou_threshold=0.5))
    #                 box_pred_per_image = box_pred_per_image[keep]
    #                 scores_per_image = scores_per_image[keep]
    #                 labels_per_image = labels_per_image[keep]


    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_img_feat(self, img, img_metas):
        pass

    def extract_pts_feat(self, pts, img_feats, img_metas):
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        return x

    def forward_single(self, gt_bboxes_3d, gt_labels_3d):
        h, w = 1408, 1600
        point_cloud_range=torch.tensor([0, -40, -3, 70.4, 40, 1]) # x0, y0, z0, x1, y1, z1
        voxel_size = torch.tensor([0.05, 0.05, 0.1])

        # 原始点云坐标
        gt_bev_bboxes = gt_bboxes_3d.bev # [22, 5] [XYWHR]
        # gt_nearest_bev_bboxes = gt_bboxes_3d.nearest_bev
        gt_classes = gt_labels_3d 
        # 真值框的个数
        num_gt_bboxes = gt_bev_bboxes.shape[0]
        # 将原始点云坐标转换为体素坐标
        for i in range(num_gt_bboxes):
            bbox = gt_bev_bboxes[i]
            bbox[:2] = (bbox[:2] - point_cloud_range[:2]) / voxel_size[:2]
            bbox[2:4] = bbox[2:4] / voxel_size[:2]
            bbox[:4] = torch.round(bbox[:4]).long()
            gt_bev_bboxes[i] = bbox

        image_size_xyxy = torch.as_tensor([h, w, h, w, 1.], dtype=torch.float)

        gt_bev_bboxes = gt_bev_bboxes / image_size_xyxy


        target = {}
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 5, device=self.device)

        
        if num_gt_bboxes < self.num_proposals:
            # 生成噪声框，坐标为[XYWHR]形式
            box_placeholder = torch.randn(self.num_proposals - num_gt_bboxes, 5) / 6 + 0.5
            box_placeholder[:, 2:4] = torch.clip(box_placeholder[:, 2:4], min=1e-4)
            # x_start为噪声框与真值框的混合
            x_start = torch.cat((gt_bev_bboxes, box_placeholder), dim=0)
        elif num_gt_bboxes > self.num_proposals:
            pass
        else:
            x_start = gt_bev_bboxes

        # padding  
        x_start = (x_start * 2. - 1.) * self.scale

        #noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.


        x = x * image_size_xyxy.cuda()
        gt_bev_bboxes = gt_bev_bboxes * image_size_xyxy

        return x, noise, t, gt_bev_bboxes, gt_classes

    def forward_train(self, 
                      points=None, 
                      img_metas=None, 
                      gt_bboxes_3d=None, 
                      gt_labels_3d=None, 
                      gt_labels=None, 
                      gt_bboxes=None, 
                      img=None, 
                      proposals=None, 
                      gt_bboxes_ignore=None):
        
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        
        # 暂时只使用单模态的数据
        # pts_feats.shape = [2, 256, 200, 176]
        fuse_feats = pts_feats
        fuse_feats = [fuse_feats]
        
        res = multi_apply(self.forward_single, gt_bboxes_3d, gt_labels_3d)
        # d_boxes, d_noise, d_t为list，大小为batch size
        # [XYWHR]
        d_boxes = [i.cuda() for i in res[0]] # proposal
        d_noise = [i.cuda() for i in res[1]]
        d_t = [i.cuda() for i in res[2]]
        gt_bev_boxes = [i.cuda() for i in res[3]]
        gt_labels = [i.cuda() for i in res[4]]

        batch_size = len(gt_labels)
        for i in range(batch_size):
            gt_label = gt_labels[i]
            gt_bev_box = gt_bev_boxes[i]
            index = 0
            for label in gt_label:
                if label.item() == -1:
                    gt_label = del_tensor_ele(gt_label, index)
                    gt_bev_box = del_tensor_ele(gt_bev_box, index)
                index += 1
            gt_labels[i] = gt_label
            gt_bev_boxes[i] = gt_bev_box

        # todo 航向角的坐标系还需要处理
        gt_bev_boxes = lidabev2img(gt_bev_boxes)

        losses = dict()

        roi_losses = self.pts_bbox_head.forward_train(fuse_feats, d_boxes, gt_bev_boxes, gt_labels, d_t)
        losses.update(roi_losses)
        return losses
        
    def simple_test(self, points, img_metas, img=None, rescale=False):
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        
        fuse_feats = pts_feats
        fuse_feats = [fuse_feats]

        results = self.ddim_sample(fuse_feats, None, points)
        return results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        return super().aug_test(points, img_metas, imgs, rescale)
