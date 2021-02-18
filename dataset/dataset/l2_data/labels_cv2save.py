import cv2
import numpy as np
from osgeo import gdal
import glob
import pdb


def load_image(patch):
    # Read Image
    print (patch)
    gdal_header = gdal.Open(patch)
    # get array
    img = gdal_header.ReadAsArray()
    return img

im_names = []
for file_ in glob.glob('labels_qgis/*'):
    im_name = file_[12:-4]
    im_names.append(im_name)
    print(file_)

    label = load_image(file_).astype(np.uint8)
    print(im_name)
    print(np.unique(label,return_counts=True))

    print(label.shape)
    cv2.imwrite('labels/'+im_name+'.tif', label)
