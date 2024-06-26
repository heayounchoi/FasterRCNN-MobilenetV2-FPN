import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    def __init__(
            self, in_channels, mid_channels, ratios,
            anchor_scales,
            feat_stride,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = [generate_anchor_base(anchor_scales=anchor_scale, ratios=ratios)
                            for anchor_scale in anchor_scales]
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = len(self.anchor_base[0])
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, features, img_size, scale=1.):
        anchors = []
        rpn_locs = []
        rpn_fg_scores = []
        rpn_scores = []
        for i, x in enumerate(features):
            n, _, hh, ww = x.shape
            anchor = _enumerate_shifted_anchor(
                np.array(self.anchor_base[i]),
                self.feat_stride[i], hh, ww)

            n_anchor = anchor.shape[0] // (hh * ww)
            h = F.relu(self.conv(x))

            rpn_loc = self.loc(h)

            rpn_loc = rpn_loc.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
            rpn_score = self.score(h)
            rpn_score = rpn_score.permute(0, 2, 3, 1).contiguous()
            rpn_softmax_scores = F.softmax(
                rpn_score.view(n, hh, ww, n_anchor, 2), dim=4)

            rpn_fg_score = rpn_softmax_scores[:, :, :, :, 1].contiguous()
            rpn_fg_score = rpn_fg_score.view(n, -1)
            rpn_score = rpn_score.view(n, -1, 2)

            anchors.append(anchor)
            rpn_locs.append(rpn_loc)
            rpn_fg_scores.append(rpn_fg_score)
            rpn_scores.append(rpn_score)

        anchors_all = np.concatenate(anchors)

        rpn_locs_all = t.cat(rpn_locs, 1)
        rpn_fg_scores_all = t.cat(rpn_fg_scores, 1)
        rpn_scores_all = t.cat(rpn_scores, 1)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs_all[i].cpu().data.numpy(),
                rpn_fg_scores_all[i].cpu().data.numpy(),
                anchors_all, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)

        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs_all, rpn_scores_all, rois, roi_indices, anchors_all


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
        shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def normal_init(m, mean, stddev, truncated=False):
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()
