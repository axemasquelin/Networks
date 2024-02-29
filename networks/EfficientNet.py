# coding: utf-8
""" MIT License """
'''
    Project: Personal Backups
    Authors: Axel Masquelin
    Description:
'''
# Libraries
# ---------------------------------------------------------------------------- #
from torch import Tensor

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import numpy as np
import math
import utils
# ---------------------------------------------------------------------------- #



def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))

class Swish(nn.Module):
    '''
    Desc:
    '''
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class BasicConv2d(nn.Module):
    '''
    Desc: Basic 2D Convolutional Block
    '''
    def __init__(self, chn_in: int, chn_out: int, nxn: int, stride: int, groups: int):
        pad = self.__get_padding(nxn, stride)
        super(BasicConv2d, self).__init__(
            nn.ZeroPad2d(pad),
            nn.Conv2d(chn_in, chn_out, nxn, stride, padding = 0, groups=groups, bias = False),
            nn.BatchNorm2d(chn_out),
            Swish(),
        )

    
    def __get_padding(self, nxn: int, stride: int) -> int:
        p = max(nxn-stride,0)
        return ([p // 2, p-p // 2, p-p // 2])

class SqueezeExcit(nn.Module):
    '''
    Desc:
    '''
    def __init__(self, chn_in: int, reduce_dim: int):
        super(SqueezeExcit, self).__init__()
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(chn_in, reduce_dim, 1),
            Swish,
            nn.Conv2d(reduce_dim, chn_in, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):

    def __init__(self,
                 chn_in,
                 chn_out,
                 expand_ratio,
                 nxn,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = chn_in == chn_out and stride == 1
        assert stride in [1, 2]
        assert nxn in [3, 5]

        hidden_dim = chn_in * expand_ratio
        reduced_dim = max(1, int(chn_in / reduction_ratio))

        layers = []
        # pw
        if chn_in != hidden_dim:
            layers += [BasicConv2d(chn_in, hidden_dim, 1)]

        layers += [
            # dw
            BasicConv2d(hidden_dim, hidden_dim, nxn, stride=stride, groups=hidden_dim),
            # se
            SqueezeExcit(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, chn_out, 1, bias=False),
            nn.BatchNorm2d(chn_out),
        ]

        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)

class EfficientNet(nn.Module):

    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000):
        super(EfficientNet, self).__init__()

        efnconfig = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]

        out_channels = _round_filters(32, width_mult)
        features = [BasicConv2d(3, out_channels, 3, stride=2)]

        in_channels = out_channels
        for t, c, n, s, k in efnconfig:
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                in_channels = out_channels

        last_channels = _round_filters(1280, width_mult)
        features += [BasicConv2d(in_channels, last_channels, 1)]

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
