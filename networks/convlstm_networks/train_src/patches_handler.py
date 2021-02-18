 
import numpy as np
import cv2
import pathlib
class PatchesArray(object):
    def __init__(self):
        pass
        #self.patches=patches
    def store(self, patches, path, filename='patches'):
        pathlib.Path(path+'ims/').mkdir(parents=True, exist_ok=True) 
        np.save(path+filename+'ims/_in.npy', patches['in']) #to-do: add polymorphism for other types of input 
        
        pathlib.Path(path+'labels/').mkdir(parents=True, exist_ok=True) 
        np.save(path+filename+'labels/_labels.npy', patches['in']) #to-do: add polymorphism for other types of input 




