# utf-8
""" MIT License """
'''
    Project: PulmonaryMAE
    Authors: Axel Masquelin
    Description:
'''
# Libraries
# ---------------------------------------------------------------------------- #
from curses import KEY_REFERENCE
from doctest import OutputChecker
from turtle import forward
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np
# ---------------------------------------------------------------------------- #

class dw_conv(nn.Module):
    """
    Depthwise + LayerNorm + Pointwise Convolution
    """
    def __init__(self, x_in:int, x_out:int, nxn:int, stride:int, dilation:int):
        super(dw_conv).__init__()

        self.depthwise=nn.Conv2d(in_channels=x_in, out_channels=x_in, kernel_size=nxn,
                      stride=stride, padding=1, groups=x_in, dilation=dilation, bias=False)
        self.LN = nn.LayerNorm(x_in, eps=1e-6, sparse=True)
        self.pointwise = nn.Conv2d(in_channels=x_in, out_channels=x_out, kernel_size=1,
                                   stride=1, padding=0, dilation=1, bias=False)
    def forward(self,x:torch.Tensor):
        x = self.depthwise(x)
        x = self.LN(x)
        return self.pointwise(x)

class SparseEncoder(nn.Module):
    """
    Sparse U-Net Encoder using DropPatch
    """
    def __init__(self, num_channels:int=1, input_size:int=1, kernel:int=3):
        super(SparseEncoder,self).__init__()

        self.dwconv1 = dw_conv(x_in=num_channels, x_out=32, nxn=kernel)
        
    def forward(self,x:torch.Tensor):
        pass

class SparseDecoder(nn.Module):
    """
    Sparse U-Net Encoder using DropPatch
    """
    def __init__(self, num_channels:int=1, input_size:int=1):
        super(SparseEncoder,self).__init__()
        pass

    def forward(self,x:torch.Tensor):
        pass

class SparseUNet(nn.Module):
    '''
    Sparse Masked Autoencoder module
    '''
    def __init__(self, maskratio:int, inputsize = 64, patchsize = 8, 
                 chnin = 1, n_classes = 2, chnout = 1,
                 embeddim = 128, depth = 24) -> None:
        super(SparseUNet, self).__init__()
        self.patchsize = patchsize
        self.inputsize = inputsize
        self.patch_num = inputsize // patchsize
        self.mask_ratio = maskratio

    def pathify(self,imgs:torch.Tensor):
        '''
        Parameters
        ----------
        imgs: (N,1,H,W)
        
        Returns
        -------
        x: (N,p,patch_size, patch_size)
        '''
        print(imgs.shape)
        print(self.patch_num)
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % self.patch_num == 0

        h = w = imgs.shape[2] // self.patch_num
        x = imgs.reshape(shape=(imgs.shape[0],self.patch_num**2,h,w))

        return x

    def unpatch(self, imgs):
        '''
        Parameters
        ----------
        x: (N,L, patch_size, patch_size)
        Returns
        -------
        imgs: (N,H,W)
        '''
        h = w = imgs.shape[2]
        assert (h*self.patch_num == self.inputsize)

        imgs = imgs.reshape(shape=(imgs.shape[0], h*self.patch_num, w*self.patch_num))

        return imgs
    
    def random_masking(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            (N,L,patch_size,patch_size) tensor
        Returns
        -------
        x_masked: torch.Tensor
            (N,L,patch_size,patch_size) tensor
        masks: torch.tensor
        """
        N,L,W,H = x.shape
        # print("Shape of X originally: ", x.shape)
        
        len_keep=int(L*(1-self.mask_ratio))

        noise = torch.rand(N,L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim= 1)
        ids_restore = torch.argsort(ids_shuffle, dim= 1)

        ids_keep = ids_shuffle[:,:len_keep]
        # print(ids_keep.shape)
        # print(ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1,1,W,H).shape)
        x_masked = torch.gather(x, dim = 0, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1,1,W,H))

        # print(x_masked.shape)
        mask = torch.ones([N,L], device= x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim= 1, index= ids_restore)
        # print("Shape of X after dropping patches: ", x.shape)
        
        return x_masked, mask, ids_restore

    def forward(self, x:torch.Tensor):
        """
        Foward Pass of U-Net style encoder decoder with sparsity
        """
        
        #Patch Images
        patchx = self.pathify(x)
        print(patchx.shape)

        #Mask Filters
        maskx, mask, ids_restore = self.random_masking(patchx)

        #Images Pass Forward     
        embs = self.encoder(maskx)
        # print('Embedding Shape: ',embs.shape)
        x = self.decoder(embs)

        # Un-patch image to Larger images
        out_imgs = self.unpatch(x)
        
        maskx = self.unpatch(maskx)

        return (out_imgs, embs, maskx)
