# coding: utf-8
""" MIT License """
'''
    Project: Scratch
    Authors: Axel Masquelin
    Description: Implementation of Inception modules and exploring various design
    implementation. 

'''
# Libraries
# ---------------------------------------------------------------------------- #

from re import S
from turtle import forward
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.Tensor as Tensor
import torchvision
import torch

import numpy as np
import utils
# ---------------------------------------------------------------------------- #

class BasicConv2d(nn.Module):
    def __init__(self, chn_in: int, chn_out: int, nxn: int):
        super.__init__()

        self.Conv = nn.Conv2d(chn_in, chn_out, nxn, bias = False)
        self.BatchNorm = nn.BatchNorm2d(chn_out, eps = 0.001)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.BatchNorm(x)

        return F.relu(x, inplace=True)

class NaiveInception(nn.Module):
    def __init__(self, chn_in: int, chn_out: int):
        super().__init__()

        self.branch1x1 = BasicConv2d(chn_in= chn_in, chn_out= chn_out, nxn = 1) 
        self.branch3x3 = BasicConv2d(chn_in= chn_in, chn_out= chn_out, nxn = 3)
        self.branch5x5 = BasicConv2d(chn_in= chn_in, chn_out= chn_out, nxn = 5)
        self.branchpool = nn.MaxPool2d(kernel_size = 3)
    
    def _forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branchpool = self.branchpool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branchpool]

        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class DRInceptionblock(nn.Module):
    def __init__(self, chn_in: int, chn_out: int) -> None:
        super().__init__()

        self.branch1x1 = BasicConv2d(chn_in= chn_in, chn_out= chn_out, nxn= 1) 
        self.branch3x3 = BasicConv2d(chn_in= chn_in, chn_out= chn_out, nxn= 3) 
        self.branch5x5 = BasicConv2d(chn_in= chn_in, chn_out= chn_out, nxn= 5)
        self.branchpool = nn.MaxPool2d(kernel_size = 3)

    def _forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(self.branch1x1(x))
        branch5x5 = self.branch5x5(self.branch1x1(x))
        branchpool = self.branch1x1(self.branchpool(x))

        outputs = [branch1x1, branch3x3, branch5x5, branchpool]

        return outputs
    
    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs,1)


class Miniception(nn.Module):
    def __init__(self, n, ch_in, n_classes = 2):
        super(Miniception, self).__init__()
        
        self.config=[
            # chn_in, chn_out
            [1, 4]
            [16, 4]
        ]

        layers = []
        for c_in, c_out in self.config:
            layers.append(DRInceptionblock(chn_in=c_in, chn_out=c_out))
        
        self.layers = nn.Sequential(*layers)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        x = self.layers(x)
        x = self.avg_pool(x)
        x = x.view(-1, 256)
        x = self.classifier(x)

        return x 