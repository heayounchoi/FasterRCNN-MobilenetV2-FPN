from __future__ import absolute_import
import torch as t
from torch import nn
from torchvision.ops import RoIPool

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt
from model.mobilenetv2 import mbnet

class FasterRCNNMbnetFPN(FasterRCNN):
    feat_stride = [4, 8, 16, 32, 64]

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[2, 4, 8, 16, 32]
                 ):

        extractor = mbnet()

        rpn = RegionProposalNetwork(
            256, 256,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = MBNetRoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / 16),
        )

        super(FasterRCNNMbnetFPN, self).__init__(
            extractor,
            rpn,
            head,
        )

class MBNetRoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier=None):
        super(MBNetRoIHead, self).__init__()

        self.top = nn.Sequential(
                            nn.Conv2d(256, 1024, kernel_size=7, stride=7, padding=0),
                            nn.ReLU(True),
                            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
                            nn.ReLU(True)
                            )
    
        self.cls_loc = nn.Linear(1024, n_class * 4)
        self.score = nn.Linear(1024, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size

    def _PyramidRoI_Feat(self, feat_maps, rois, img_size, roi_indices):
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()
        
        img_area = img_size[0] * img_size[1]
        h = rois[:, 2] - rois[:, 0] + 1
        w = rois[:, 3] - rois[:, 1] + 1
        roi_level = t.log(t.sqrt(h * w) / 224.0)
        roi_level = t.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5

        roi_pool_feats = []
        box_to_levels = []
        for i, l in enumerate(range(2, 6)):
            if (roi_level == l).sum() == 0:
                continue
            idx_l = (roi_level == l).nonzero().squeeze()
            if idx_l.dim() == 0:
                idx_l = idx_l.unsqueeze(0)
            box_to_levels.append(idx_l)
            scale = feat_maps[i].size(2) / img_size[0]

            indices = indices_and_rois[idx_l]
            if indices.dim() == 1:
                indices = indices.unsqueeze(0)
            feat = RoIPool((self.roi_size, self.roi_size), scale)(feat_maps[i], indices)

            roi_pool_feats.append(feat)
        roi_pool_feat = t.cat(roi_pool_feats, 0)
        box_to_level = t.cat(box_to_levels, 0)
        idx_sorted, order = t.sort(box_to_level)
        roi_pool_feat = roi_pool_feat[order]
        return roi_pool_feat

    def _head_to_tail(self, pool5):
        block5 = self.top(pool5)
        fc7 = block5.mean(3).mean(2)
        return fc7

    def forward(self, features, rois, roi_indices, img_size):
        roi_pool_feat = self._PyramidRoI_Feat(features, rois, img_size, roi_indices)
        pooled_feat = self._head_to_tail(roi_pool_feat)
        roi_cls_locs = self.cls_loc(pooled_feat)
        roi_scores = self.score(pooled_feat)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()

    