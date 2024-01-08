# coding: utf-8
""" MIT License """
'''
    Project: Scratch
    Authors: Axel Masquelin
    Description: Implementation of MobileNetV1, MobileNetV2, and Custom iterations
'''
# Libraries
# ---------------------------------------------------------------------------- #

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.tensor
import torchvision
import torch

import numpy as np
import utils
# ---------------------------------------------------------------------------- #

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        
        def conv_full(x, x_out, stride):
            return nn.Sequential(
                nn.Conv2d(x, x_out, 3, stride, 1, bias = False),
                nn.BatchNorm2d(x_out),
                nn.ReLU(inplace = True)
            )
        
        def conv_depthwise(x, x_out, stride): 
            return nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(x, x, (3,3) , stride, 1, groups = x, bias = False),
                nn.BatchNorm2d(x),
                nn.ReLU(inplace = True),
                
                '''
                How would MobileNet behave within a non-square kernel?
                Are there situations where a non-square kernel would be ideal with a MobileNet?
                '''

                # Pointwise Convolution
                nn.Conv2d(x, x, (1,1), stride = 1, bias = False),
                nn.BatchNorm2d(x_out),
                nn.ReLU(inplace = True)
                )
        
        self.model = nn.Sequential(
            conv_full(chn_in, 32, 2),
            conv_depthwise(32, 64, 1),
            conv_depthwise(64, 128, 2),
            conv_depthwise(128, 128, 1),
            conv_depthwise(128, 256, 2),
            conv_depthwise(256,256, 1),
            conv_depthwise(256, 512, 2),
            conv_depthwise(512, 512, 1),
            conv_depthwise(512, 512, 1),
            conv_depthwise(512, 512, 1),
            conv_depthwise(512, 512, 1),
            conv_depthwise(512, 512, 1),
            conv_depthwise(512, 1024, 2),
            conv_depthwise(1024, 1024, 1),
            nn.AdaptiveAvgPool2dd(1)
        )

        self.fullyconnected = nn.Linear(256, n_class)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, max_size)
        x = self.fullyconnected(x)
        
        return x


class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            #dw
            dwise_conv(hidden_dim, stride=stride),
            #pw
            conv1x1(hidden_dim, ch_out)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

class MobileNetV2(nn.Module):
    def __init__(self, ch_in=3, n_classes=1000):
        super(MobileNetV2, self).__init__()

        self.configs=[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.stem_conv = conv3x3(ch_in, 32, stride=2)

        layers = []
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c

        self.layers = nn.Sequential(*layers)

        self.last_conv = conv1x1(input_channel, 1280)

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Linear(1280, n_classes)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).view(-1, 2)
        x = self.classifier(x)
        return x


        