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
from mmrotate.core import multiclass_nms_rotated
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import matplotlib.pyplot as plt

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

avg_height = [1.73, 1.73, 1.56, 0]

def drawimg(xy):
    xy = xy.cpu().numpy()
    x = xy[:, 0].flatten()
    y = xy[:, 1].flatten()
    plt.figure(num='haha')
    plt.scatter(x, y)
    plt.xlim((0, 1600))
    plt.ylim((0, 1408))
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')  #将x轴的位置设置在顶部
    # ax.invert_xaxis()  #x轴反向
    ax.yaxis.set_ticks_position('left')  #将y轴的位置设置在左边
    ax.invert_yaxis()  #y轴反向

    plt.savefig('out/img.png')

def drawlidar(xy):
    xy = xy.cpu().numpy()
    x = xy[:, 0].flatten()
    y = xy[:, 1].flatten()
    plt.figure(num='hoho')
    plt.scatter(y, x)
    plt.xlim((-800, 800))
    plt.ylim((0, 1408))
    ax = plt.gca()
    # ax.xaxis.set_ticks_position('top')  #将x轴的位置设置在顶部
    ax.invert_xaxis()  #x轴反向
    # ax.yaxis.set_ticks_position('left')  #将y轴的位置设置在左边
    # ax.invert_yaxis()  #y轴反向

    plt.savefig('out/lidar.png')

# todo有bug需要修改
def addheight(bboxes, labels):
    batch_size = bboxes.shape[0]
    bboxes = bboxes.cpu().numpy()
    bboxes_t = []
    for i in range(batch_size):
        bbox = bboxes[i]
        lenth = np.size(bbox, 0)

        temp = np.ones(lenth)
        bbox = np.insert(bbox, 2, temp, axis=1)
        bbox = np.insert(bbox, 5, temp, axis=1)

        bbox[:, :6] = bbox[:, :6] * 0.05

        for j in range(lenth):
            label = labels[i, j].item()
            height = avg_height[label]
            bbox[j][2] = height / 2 - 2.5
            bbox[j][5] = height
        bbox = LiDARInstance3DBoxes(bbox)    
        bboxes_t.append(bbox)
    # bboxes = torch.from_numpy(bboxes_t)

    return bboxes_t

def lidarbev2img(bboxes):
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
            r = bev_bbox[j][4]
            if r > math.pi :
                r = r - math.pi
            bev_bbox[j][4] = r - math.pi / 2
        bev_bboxes[i] = torch.from_numpy(bev_bbox).cuda()
    return bev_bboxes

def img2lidarbev(bev_bboxes):
    bboxes = bev_bboxes
    batch_size = bev_bboxes.shape[0]
    for i in range(batch_size):
        bbox = bboxes[i].cpu().numpy()
        lenth = np.size(bbox, 0)
        for j in range(lenth):
            w, h = bbox[j][2], bbox[j][3]
            x = 1408 - bbox[j][1]
            y = 800 - bbox[j][0]
            bbox[j][:4] = np.asarray([x, y, h, w])
            r = bbox[j][4] + math.pi / 2
            bbox[j][4] = r
        bboxes[i] = torch.from_numpy(bbox).cuda()
    return bboxes

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

    def model_predictions(self, backbone_feats, images_whwhr, x, t, x_self_cond=None, clip_x_start=False):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = x_boxes * images_whwhr[:, None, :]        

         # *outputs_score=[bs, num_boxes, num_class+1], outputs_coord=[bs, num_boxes, 5]
        
        # drawimg(x_boxes[0])

        outputs_coord = x_boxes
        
        for i in range(6):
            outputs_score, outputs_coord = self.pts_bbox_head.simple_test(backbone_feats, outputs_coord, t)
            
        # drawimg(outputs_coord[0])

        x_start = outputs_coord
        x_start = x_start / images_whwhr[:, None, :]
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_score, outputs_coord


    @torch.no_grad()
    def ddim_sample(self, backbone_feats, images_whwhr, points, clip_denoised=True, do_postprocess=True):
        

        w, h = 1600, 1408
        batch_size = len(points)
        images_whwhr = torch.tensor([w, h, 100, 100, 2 * math.pi]).repeat(batch_size, 1).cuda()

        shape = (batch_size, self.num_proposals, 5)
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
            time_cond = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            
            # *outputs_score=[bs, num_boxes, num_class+1], outputs_coord=[bs, num_boxes, 5]
            preds, outputs_scores, outputs_coords = self.model_predictions(backbone_feats, images_whwhr, img, time_cond, self_cond, clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start


            # todo  这里需要修改
            if self.box_renewal:
                score_per_image, box_per_image = outputs_scores[0], outputs_coords[0]
                threshold = 0.5
                # score_per_image = torch.sigmoid(score_per_image)
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

        # *nms
        bbox_t, score_t, label_t = [], [], []
        for i in range(batch_size):
            scores, bboxes = outputs_scores[i].cpu(), outputs_coords[i].cpu()
            det_bboxes, det_labels = multiclass_nms_rotated(
                bboxes, scores, self.test_cfg['pts']['score_thr'], self.test_cfg['pts']['nms'], self.test_cfg['pts']['max_per_img'])
            det_scores = det_bboxes[:, 5:].cuda().squeeze(1)
            det_bboxes = det_bboxes[:, :5].cuda()
            
            # drawimg(det_bboxes)

            bbox_t.append(det_bboxes)
            score_t.append(det_scores)
            label_t.append(det_labels)

        outputs_coords = torch.stack(bbox_t).cuda()
        outputs_scores = torch.stack(score_t).cuda()
        outputs_labels = torch.stack(label_t).cuda()

        # *将坐标系进行转换，角度进行转换
        outputs_coords = img2lidarbev(outputs_coords)

        # drawlidar(outputs_coords[0])

        

        # *将二维BEV的bbox转换为三维的bbox, 对outputs_coord进行处理，加上高度数据
        outputs_coords = addheight(outputs_coords, outputs_labels)

        bbox_results = [
            bbox3d2result(outputs_coords[i], outputs_scores[i], outputs_labels[i])
            for i in range(batch_size)
        ]
        return bbox_results
        


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
        batch_size = coors[-1, 0].item() + 1        
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def noise_boxes_gen(self, gt_bboxes_3d, gt_labels_3d):
        w, h = 1408, 1600
        point_cloud_range=torch.tensor([0, -40, -3, 70.4, 40, 1]) # x0, y0, z0, x1, y1, z1
        voxel_size = torch.tensor([0.05, 0.05, 0.1])

        # *原始点云坐标
        gt_bev_bboxes = gt_bboxes_3d.bev # [22, 5] [XYWHR]
        # gt_nearest_bev_bboxes = gt_bboxes_3d.nearest_bev
        gt_labels = gt_labels_3d 

        # * 真值清洗
        index = 0
        for lable in gt_labels:
            if lable.item() == -1:
                gt_labels = del_tensor_ele(gt_labels, index)
                gt_bev_bboxes = del_tensor_ele(gt_bev_bboxes, index)
            else:
                index += 1
        

        # *真值框的个数
        num_gt_bboxes = gt_bev_bboxes.shape[0]
        # *将原始点云坐标转换为体素坐标
        for i in range(num_gt_bboxes):
            bbox = gt_bev_bboxes[i]
            bbox[:2] = (bbox[:2] - point_cloud_range[:2]) / voxel_size[:2]
            bbox[2:4] = bbox[2:4] / voxel_size[:2]
            bbox[:4] = torch.round(bbox[:4]).long()
            gt_bev_bboxes[i] = bbox

        image_size_xyxyr = torch.as_tensor([w, h, 100, 100, 1.], dtype=torch.float)

        gt_bev_bboxes = gt_bev_bboxes / image_size_xyxyr


        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 5, device=self.device)

        
        if num_gt_bboxes < self.num_proposals:
            # *生成噪声框，坐标为[XYWHR]形式
            box_placeholder = torch.randn(self.num_proposals - num_gt_bboxes, 5) / 6 + 0.5
            box_placeholder[:, 2:4] = torch.clip(box_placeholder[:, 2:4], min=1e-4)
            # *x_start为噪声框与真值框的混合
            x_start = torch.cat((gt_bev_bboxes, box_placeholder), dim=0)
        elif num_gt_bboxes > self.num_proposals:
            pass
        else:
            x_start = gt_bev_bboxes

        # *padding  
        x_start = (x_start * 2. - 1.) * self.scale

        # *noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.


        x = x * image_size_xyxyr.cuda()
        gt_bev_bboxes = gt_bev_bboxes * image_size_xyxyr

        return x, noise, t, gt_bev_bboxes, gt_labels

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
        
        # *暂时只使用单模态的数据
        # *pts_feats.shape = [bs, 256, 200, 176]
        fuse_feats = pts_feats[0]
        fuse_feats = [fuse_feats]
        
        res = multi_apply(self.noise_boxes_gen, gt_bboxes_3d, gt_labels_3d)
        # *d_boxes, d_noise, d_t为list，大小为batch size
        # *[XYWHR]
        d_boxes = [i.cuda() for i in res[0]] # proposal
        d_noise = [i.cuda() for i in res[1]]
        d_t = [i.cuda() for i in res[2]]
        gt_bev_boxes = [i.cuda() for i in res[3]]
        gt_labels = [i.cuda() for i in res[4]]


        # *航向角的坐标系还需要处理
        gt_bev_boxes = lidarbev2img(gt_bev_boxes)

        losses = dict()

        d_t = torch.stack(d_t).flatten()
        
        # drawimg(d_boxes[0])
        # drawimg(gt_bev_boxes[0])

        roi_losses = self.pts_bbox_head.forward_train(fuse_feats, d_boxes, gt_bev_boxes, gt_labels, d_t)
        losses.update(roi_losses)
        return losses
        
    def simple_test(self, points, img_metas, img=None, rescale=False):
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        
        fuse_feats = pts_feats[0]
        fuse_feats = [fuse_feats]

        results = self.ddim_sample(fuse_feats, None, points)
        return results

    # def aug_test(self, points, img_metas, imgs=None, rescale=False):
    #     return super().aug_test(points, img_metas, imgs, rescale)
