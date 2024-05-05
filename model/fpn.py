import torch.nn as nn
import torch.nn.functional as F


class _FPN(nn.Module):
    def __init__(self):
        super(_FPN, self).__init__()
        self.maxpool2d = nn.MaxPool2d(1, stride=2)

        self._init_modules()
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev):
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.RCNN_toplayer, 0, 0.01)
        normal_init(self.RCNN_smooth1, 0, 0.01)
        normal_init(self.RCNN_smooth2, 0, 0.01)
        normal_init(self.RCNN_smooth3, 0, 0.01)
        normal_init(self.RCNN_latlayer1, 0, 0.01)
        normal_init(self.RCNN_latlayer2, 0, 0.01)
        normal_init(self.RCNN_latlayer3, 0, 0.01)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, im_data):
        batch_size = im_data.size(0)

        # Bottom-up
        c1 = self.RCNN_layer0(im_data)
        c2 = self.RCNN_layer1(c1)
        c3 = self.RCNN_layer2(c2)
        c4 = self.RCNN_layer3(c3)
        c5 = self.RCNN_layer4(c4)
        # Top-down
        p5 = self.RCNN_toplayer(c5)
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p4 = self.RCNN_smooth1(p4)
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p3 = self.RCNN_smooth2(p3)
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p2 = self.RCNN_smooth3(p2)

        p6 = self.maxpool2d(p5)

        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        return rpn_feature_maps, mrcnn_feature_maps
