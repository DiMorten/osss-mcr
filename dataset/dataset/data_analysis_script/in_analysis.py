 
import numpy as np
path="../lm_data/in_optical/"
im=np.load(path+'20170729_S2_10m.npy')
print(im.shape)
print(np.average(im),np.min(im),np.max(im))

im=im[:,:,(3,1,0)]
print(im.shape)
print(np.average(im),np.min(im),np.max(im))

print("=== per band analysis")
print(im[:,:,0].shape)
print(np.average(im[:,:,0]),np.min(im[:,:,0]),np.max(im[:,:,0]))
