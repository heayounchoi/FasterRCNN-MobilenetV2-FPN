from model.fpn import _FPN

import torch.nn as nn
import torch.nn.functional as F


from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchinfo import summary
import torch as t


def mobilenetv2():
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    features = model.features

    for layer in features[:2]:
        for p in layer.parameters():
            p.requires_grad = False

    for name, module in features.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for p in module.parameters():
                p.requires_grad = False

    return features


class MobileNetV2(nn.Module):
    def __init__(self):
        model = mobilenetv2()
        super(MobileNetV2, self).__init__()
        self.layer0 = model[:2]
        self.layer1 = model[2:4]
        self.layer2 = model[4:7]
        self.layer3 = model[7:14]
        self.layer4 = model[14:]

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class mbnet(_FPN):
    def __init__(self):
        _FPN.__init__(self)

    def _init_modules(self):
        mbnet = MobileNetV2()

        self.RCNN_layer0 = mbnet.layer0
        self.RCNN_layer1 = mbnet.layer1
        self.RCNN_layer2 = mbnet.layer2
        self.RCNN_layer3 = mbnet.layer3
        self.RCNN_layer4 = mbnet.layer4

        # Top layer
        self.RCNN_toplayer = nn.Conv2d(
            1280, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.RCNN_smooth1 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth2 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth3 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.RCNN_latlayer1 = nn.Conv2d(
            96, 256, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer2 = nn.Conv2d(
            32, 256, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer3 = nn.Conv2d(
            24, 256, kernel_size=1, stride=1, padding=0)

    def _head_to_tail(self, pool5):
        block5 = self.RCNN_top(pool5)
        fc7 = block5.mean(3).mean(2)
        return fc7
