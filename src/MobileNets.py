# coding: utf-8
""" MIT License """
'''
    Project: Personal Backups
    Authors: Axel Masquelin
    Description:
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


        