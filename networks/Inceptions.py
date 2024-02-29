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


class DRInception(nn.Module):
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


class DRception(nn.Module):
    def __init__(self, n, ch_in, n_classes = 2):
        super(DRception, self).__init__()
        
        self.DRconfig=[
            # Chn1xa_in, chn1xa_out, chn3xa_out, chn3xb_out, chn1xb_in, chn1xb_out, chnblock_out
            [1, 3, 5, 7, 7, 8],
            [8, 12, 24, 36, 12, 20],
            # [66, 12, 24, 36, 50, 116],
            # [116, 12, 24, 36, 50, 125],
        ]
        layers = []
        for  chn1xa_in, chn1xa_out, chn3xa_out, chn3xb_out, chn1xb_out, chnblock_out in self.DRconfig:
            layers.append(DRInception(chn1xa_in= chn1xa_in, chn1xa_out= chn1xa_out, 
                                      chn3xa_out= chn3xa_out, chn3xb_out= chn3xb_out,
                                      chn1xb_out= chn1xb_out, chnblock_out= chnblock_out)
                                      )
        
        
        self.layers = nn.Sequential(*layers)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(256, n_classes)
        )
        
    def init_weights(m):
        '''Initializes Model Weights using Xavier Uniform Function'''
        np.random.seed(2020)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def adjust_lr(optimizer, lrs, epoch):
        lr = lrs * (0.01 ** (epoch // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def forward(self, x):
        x = self.layers(x)
        x = self.avg_pool(x)
        x = x.view(-1, 256)
        x = self.classifier(x)

        return x 
 

class Naiveception(nn.Module):
    def __init__(self, n_classes:int = 2, avgpool_kernel:int = 16):
        super(Miniception, self).__init__()
        self.avgpool_kernel = avgpool_kernel
        self.Naiveconfig=[
            # 1ach_in, 1ach_out, 3ch_in, 3ch_out, 5ch_in, 5ch_out
            [1, 8, 1, 8, 1, 8],
            [25, 12, 25, 12, 25, 12]
        ]

        layers = []

        for c1x_in, c1x_out, c3x_in, c3x_out, c5x_in, c5x_out in self.Naiveconfig:
            layers.append(NaiveInception(chn1x_in=c1x_in, chn1x_out=c1x_out,
                                           chn3x_in=c3x_in, chn3x_out=c3x_out,
                                           chn5x_in=c5x_in, chn5x_out=c5x_out
                                           )
                        )
        
        self.dims = self.DRconfig[-1][-1]
        
        self.layers = nn.Sequential(*layers)

        self.avg_pool = nn.AdaptiveAvgPool2d(self.avgpool_kernel)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.dims*16**2, 250),
            nn.ReLU(inplace = True),
            nn.Dropout(0.75),
            nn.Linear(250, self.dims),
            nn.ReLU(inplace = True),
            nn.Dropout(0.75),
            nn.Linear(self.dims, n_classes),
            nn.Sigmoid()
        )

    def init_weights(m):
        '''Initializes Model Weights using Xavier Uniform Function'''
        np.random.seed(2020)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def adjust_lr(optimizer, lrs, epoch):
        lr = lrs * (0.01 ** (epoch // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def forward(self, x, flag = False):
        x = self.layers(x)
        x_avg = self.avg_pool(x)
        x = x_avg.view(-1, self.dims*self.avgpool_kernel**2)
        x = self.classifier(x)

        if flag:
            return x, x_avg
        
        return x 