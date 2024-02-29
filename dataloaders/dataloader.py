# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description:
'''
# Dependencies
# ---------------------------------------------------------------------------- #
from torch.utils.data import Dataset
from sklearn.utils import resample
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import glob
import os
import cv2

import ParenchymalAttention.utils.image as image 
# ---------------------------------------------------------------------------- #

class ZarrLoader(Dataset):
    """
    """
    def __init__(self, data, method= None, augmentations=None, masksize=None, norms='stand'):
        super().__init__()
        self.data = data
        self.augment= augmentations
        self.norms = norms

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index:int):
        pass

class CsvLoader(Dataset):
    """
    """
    def __init__(self, data, method= None, augmentations=None, masksize=None, norms='stand'):
        super().__init__()
        self.data = data
        self.augment= augmentations
        self.norms = norms

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index:int):
        pass

class DFLoader(Dataset):
    """
    """
    def __init__(self, data, method= None, augmentations=None, masksize=None, norms='stand'):
        super().__init__()
        self.data = data
        self.augment= augmentations
        self.masksize = masksize
        self.norms = norms
        self.method = method

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index:int):
        row = self.data.iloc[index]
        img = Image.open(row['uri'])
        img = np.asarray(img).T     

        maskmap = Image.open(row['segmented_uri'])
        maskmap = np.asarray(maskmap).T

        label = row['ca']
        im = np.zeros((1,64,64))
        if self.method != 'Original':
            im[0,:,:] = img[0,:,:]
            img = normalize_img(im, self.norms)
            img = seg_method(img, maskmap=maskmap, method= self.method, masksize = self.masksize)
            sample = {'image': img,
                    'label': label,
                    'id': row['pid']}

        else:
            im[0,:,:] = img[0,:,:]
            sample = {'image': normalize_img(im, self.norms),
                      'label': label,
                      'id': row['pid']}

        return sample         
