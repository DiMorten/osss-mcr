
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:45:38 2018

@author: laura
"""


import json
import logging
import os
import shutil
import numpy as np
import sys
import errno
from osgeo import gdal
import glob
import multiprocessing
import subprocess, signal
import gc
from sklearn import preprocessing as pp
import joblib
import pandas as pd
from itertools import groupby
from collections import Counter
from sklearn.metrics import accuracy_score, cohen_kappa_score,precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 5})

colormap_list = np.array([[40/255.0, 255/255.0, 40/255.0],
          [166/255.0, 206/255.0, 227/255.0],
          [31/255.0, 120/255.0, 180/255.0],
          [178/255.0, 223/255.0, 138/255.0],
          [51/255.0, 160/255.0, 44/255.0],
          [251/255.0, 154/255.0, 153/255.0],
          [227/255.0, 26/255.0, 28/255.0],
          [253/255.0,191/255.0,111/255.0],
          [255/255.0, 127/255.0, 0/255.0],
          [202/255.0, 178/255.0, 214/255.0],
          [106/255.0, 61/255.0, 154/255.0],
          [255/255.0,255/255.0,153/200.0],
          [255/255.0, 40/255.0, 255/255.0],
          [255/255.0, 146/255.0, 36/255.0],
          [177/255.0, 89/255.0, 40/255.0],
          [255/255.0, 255/255.0, 0/255.0],
          [0/255.0, 0/255.0, 0/255.0]])

 

def load_image(patch):
    # Read Image
    print (patch)
    gdal_header = gdal.Open(patch)
    # get array
    img = gdal_header.ReadAsArray()
    return img


def add_padding(img, psize, overl):
    '''Function to padding image
        input:
            patches_size: psize
            stride: stride
            img: image (row,col,bands)
    '''    

    try:
        row, col, bands = img.shape
    except:
        bands = 0
        row, col = img.shape
        
    # Percent of overlap between consecutive patches.
    # The overlap will be multiple of 2
    overlap = int(round(psize * overl))
    overlap -= overlap % 2
    stride = psize - overlap

    # Add Padding to the image to match with the patch size and the overlap
    row += overlap//2
    col += overlap//2
    step_row = (stride - row % stride) % stride
    step_col = (stride - col % stride) % stride
    
    if bands>0:
        npad_img = ((overlap//2, step_row), (overlap//2, step_col),(0,0))
    else:        
        npad_img = ((overlap//2, step_row), (overlap//2, step_col))  
        
    # padd with symetric (espelhado)    
    pad_img = np.pad(img, npad_img, mode='symmetric')

    # Number of patches: k1xk2
    k1, k2 = (row+step_row)//stride, (col+step_col)//stride
    print('Number of patches: %d' %(k1 * k2))

    return pad_img, stride, step_row, step_col, overlap

def seq_add_padding(img, psize, overl):
    '''Function to padding image
        input:
            patches_size: psize
            stride: stride
            img: image (row,col,bands)
    '''    
    try:
        t_len, row, col, bands = img.shape
    except:
        bands = 0
        t_len, row, col = img.shape
        
    # Percent of overlap between consecutive patches.
    # The overlap will be multiple of 2
    overlap = int(round(psize * overl))
    overlap -= overlap % 2
    stride = psize - overlap

    # Add Padding to the image to match with the patch size and the overlap
    row += overlap//2
    col += overlap//2
    step_row = (stride - row % stride) % stride
    step_col = (stride - col % stride) % stride
    
    if bands>0:
        npad_img = ((0,0),(overlap//2, step_row), (overlap//2, step_col),(0,0))
    else:        
        npad_img = ((0,0),(overlap//2, step_row), (overlap//2, step_col))  
    
    #pad_img = np.zeros((t_len,row, col, bands))
    # padd with symetric (espelhado)    
    #for t_step in t_len:
    #    pad_img[t_step] = np.pad(img[t_step], npad_img, mode='symmetric')
    pad_img = np.pad(img, npad_img, mode='symmetric')
    # Number of patches: k1xk2
    k1, k2 = (row+step_row)//stride, (col+step_col)//stride
    print('Number of patches: %d' %(k1 * k2))

    return pad_img, stride, step_row, step_col, overlap

