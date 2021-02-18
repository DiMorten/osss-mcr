import cv2
import numpy as np
import matplotlib.pyplot as plt
path='../../../../dataset/dataset/lm_data/patches_bckndfixed/'

patches_in=np.load(path+'train/'+'patches_in.npy')
print(np.histogram(patches_in, bins=20))