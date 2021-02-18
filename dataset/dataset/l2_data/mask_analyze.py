import numpy as np
import cv2
from osgeo import gdal
import pdb

def load_image(patch):
    # Read Image
    print (patch)
    gdal_header = gdal.Open(patch)
    # get array
    img = gdal_header.ReadAsArray()
    return img
lem2_mask = cv2.imread('TrainTestMask_large_coffee.tif', 0).astype(np.uint8)
#lem2_mask = cv2.imread('TrainTestMask.tif', 0).astype(np.uint8)

print(lem2_mask.shape)
lem2_label_name = 'labels/20200912_S1.tif'
lem2_label_name = 'labels/20200221_S1.tif'
lem2_label_name = 'labels/20191223_S1.tif'

lem2_label = load_image(lem2_label_name).astype(np.uint8)

lem2_label_train = lem2_label.copy()
lem2_label_test = lem2_label.copy()

lem2_label_train[lem2_mask!=1] = 0
lem2_label_test[lem2_mask!=2] = 0

print("np.unique(lem2_label, return_counts=True)",np.unique(lem2_label, return_counts=True))
print("np.unique(lem2_label_train, return_counts=True)",np.unique(lem2_label_train, return_counts=True))
print("np.unique(lem2_label_test, return_counts=True)",np.unique(lem2_label_test, return_counts=True))

lem1_mask = cv2.imread('../lm_data/TrainTestMask.tif', 0).astype(np.uint8)
lem1_label = load_image('../lm_data/labels/20170916_S1.tif').astype(np.uint8)

lem1_label_masked = lem1_label.copy()
lem1_label_masked[lem1_mask!=2] = 0

print("np.unique(lem1_label, return_counts=True)",np.unique(lem1_label, return_counts=True))
print("np.unique(lem1_label_masked, return_counts=True)",np.unique(lem1_label_masked, return_counts=True))
