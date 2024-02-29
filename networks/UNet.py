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


class ConvBlock(nn.Module):
    """
    Parameters
    ----------
    channel_in: int
        Number of channels going into convolution block
    channel_out: int
        Number of channels coming out of the convolution block
    mid_channels: int (Optional)
        Allows for multiple convolutional layers in a single block
    outconv: Bool
        Returns a single convolution without ReLU or batchnormalization
    Returns
    -------

    """
    def __init__(self, channel_in: int, channel_out: int, mid_channels = None, outconv = False) -> None:
        super().__init__()
        if outconv:
            self.convblock = nn.Sequential(
                nn.Conv2d(channel_in, channel_out, kernel_size = 1),
                nn.BatchNorm2d(channel_out),
                nn.ReLU(inplace = True),
                nn.Conv2d(channel_out, channel_out, kernel_size = 1))

        else:
            if not mid_channels:
                mid_channels = channel_out

            self.convblock = nn.Sequential(
                    nn.Conv2d(channel_in, mid_channels, kernel_size = 3, padding = 1, bias = False),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(mid_channels, channel_out, kernel_size = 3, padding = 1, bias = False),
                    nn.BatchNorm2d(channel_out),
                    nn.ReLU(inplace = True)
                    )

    def forward(self, x):        
        return self.convblock(x)


class Up(nn.Module):
    """
    """
    def __init__(self, channel_in, channel_out, bilinear = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode= 'bilinear', align_corners = True)
            self.conv = ConvBlock(channel_in, channel_out, mid_channels= channel_in // 2)
        else:
            self.up = nn.ConvTranspose2d(channel_in, channel_in//2, kernel_size= 2, stride= 2)
            self.conv = ConvBlock(channel_in, channel_out)

    def forward(self, x, x1):
        x = self.up(x)

        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x], dim=1)      
        x = self.conv(x)
        return x

class Down(nn.Module):
    """
    """
    def __init__(self, channel_in: int, channel_out: int) -> None:
        super().__init__()
        self.max_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(channel_in= channel_in, channel_out = channel_out)
            )
    def forward(self, x):
        return self.max_conv(x)

class Encoder(nn.Module):
    """
    Encoder for MAE
    """
    def __init__(self, n_channels: int, embeddim: int) -> None:
        super(Encoder,self).__init__()
        
        self.convblock = ConvBlock(channel_in= n_channels, channel_out = 256)
        self.down1 = Down(channel_in = 256, channel_out = 512)
        self.down2 = Down(channel_in = 512, channel_out = embeddim)
        
        self.model = nn.Sequential(
            ConvBlock(channel_in= n_channels, channel_out = 256),
            Down(channel_in = 256, channel_out = 512),
            Down(channel_in = 512, channel_out = embeddim)
            )

    def forward(self, x):
        x = self.convblock(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        return (x2, x1, x)


class Decoder(nn.Module):
    '''
    MAE Decoder
    '''
    def __init__(self, embeddim: int, outputsize: int) -> None:
        super(Decoder,self).__init__()
        
        self.up1 = Up(channel_in=embeddim +512, channel_out= 512)
        self.up2 = Up(channel_in= 512 + 256, channel_out= 256)
        self.outconv = ConvBlock(channel_in= 256, channel_out= outputsize, outconv=True)
        
        self.model = nn.Sequential(
            Up(channel_in=embeddim, channel_out= 512),
            Up(channel_in= 512, channel_out= 256),
            # Up(channel_in= 256, channel_out= 128),
            # Up(channel_in= 128, channel_out= 64),
            ConvBlock(channel_in= 256, channel_out= outputsize, outconv=True),
            )
    
    def forward(self, x, x1, x2):     
        x = self.up1(x,x1)    
        x = self.up2(x,x2)
        x = self.outconv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)