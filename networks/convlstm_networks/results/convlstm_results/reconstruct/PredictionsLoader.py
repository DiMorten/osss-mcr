from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose
# from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.optimizers import Adam,Adagrad 
from keras.models import Model
from keras import backend as K
import keras

import numpy as np
from sklearn.utils import shuffle
import cv2
#from skimage.util import view_as_windows
import argparse
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import metrics
import sys
import glob
import pdb
sys.path.append('../../../train_src/')
import deb
import pickle
class PredictionsLoader():
	def __init__(self):
		pass


class PredictionsLoaderNPY(PredictionsLoader):
	def __init__(self):
		pass
	def loadPredictions(self,path_predictions, path_labels):
		return np.load(path_predictions, allow_pickle=True), np.load(path_labels, allow_pickle=True)

class PredictionsLoaderModel(PredictionsLoader):
	def __init__(self, path_test):
		self.path_test=path_test
	def loadPredictions(self,path_model):
		print("============== loading model =============")
		model=load_model(path_model, compile=False)
		print("Model", model)
		test_in=np.load(self.path_test+'patches_in.npy',mmap_mode='r')
		test_label=np.load(self.path_test+'patches_label.npy')

		test_predictions = model.predict(test_in)
		print(test_in.shape, test_label.shape, test_predictions.shape)
		print("Test predictions dtype",test_predictions.dtype)
		del test_in
		return test_predictions, test_label, model
	def loadModel(self,path_model):
		model=load_model(path_model, compile=False)
		return model
	def newLabel2labelTranslate(self, label, filename, bcknd_flag=True):
		print("Entering newLabel2labelTranslate")
		label = label.astype(np.uint8)
		# bcknd to 0
		deb.prints(np.unique(label,return_counts=True))
		deb.prints(np.unique(label)[-1])
		if bcknd_flag == True:
			label[label==np.unique(label)[-1]] = 255 # this -1 will be different for each dataset
		deb.prints(np.unique(label,return_counts=True))
		label = label + 1
		
		deb.prints(np.unique(label,return_counts=True))

		# translate 
		f = open(filename, "rb")
		new_labels2labels = pickle.load(f)
		deb.prints(new_labels2labels)

		classes = np.unique(label)
		deb.prints(classes)
		translated_label = label.copy()
		for j in range(len(classes)):
			print(classes[j])
			print("Translated",new_labels2labels[classes[j]])
			translated_label[label == classes[j]] = new_labels2labels[classes[j]]

		# bcknd to last
		##label = label - 1 # bcknd is 255
		##label[label==255] = np.unique(label)[-2]
		return translated_label 