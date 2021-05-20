import cv2
import os
import glob
import numpy as np
import pdb
from pathlib import Path
import csv
import threading
import keras
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, ConvLSTM2D, Activation, BatchNormalization, Bidirectional, TimeDistributed, AveragePooling2D, MaxPooling2D, Lambda, concatenate
from keras.models import Model
from keras.regularizers import l1,l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf


import sys
from keras import backend as K
from keras import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

import tensorflow
from keras.applications.vgg16 import VGG16
from icecream import ic
import matplotlib.pyplot as plt
import pdb

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, inputs, labels, batch_size=16, dim=(20,128,128), label_dim=(128,128),
				n_channels=3, n_classes=2, shuffle=True):
		'Initialization'
		self.inputs = inputs
		self.dim = dim
		self.batch_size = batch_size
		ic(self.batch_size)
		self.labels = labels

		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.label_dim = label_dim
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
#		return int(np.floor(len(self.list_IDs) / self.batch_size))
		n_batches = int(np.floor(self.inputs.shape[0] / self.batch_size))
		ic(n_batches)
		return n_batches
	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
#		list_IDs_temp = [self.list_IDs[k] for k in indexes]
		inputs_batch = self.inputs[indexes]
		labels_batch = self.labels[indexes]

		# Generate data
		X, y = self.__data_generation(inputs_batch, labels_batch)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
#		self.indexes = np.arange(len(self.list_IDs))
		self.indexes = np.arange(self.inputs.shape[0])

		if self.shuffle == True:
			np.random.shuffle(self.indexes)

#	def scalerApply(self, X):
#		X_shape = X.shape
#		X = np.reshape(X, (-1, X_shape[-1]))
#		X = self.scaler.transform(X)
#		X = np.reshape(X, X_shape)
#		return X

#	def __data_generation(self, list_IDs_temp):
	def __data_generation(self, inputs_batch, labels_batch):

		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
		Y = np.empty((self.batch_size, *self.label_dim, self.n_classes), dtype=int)

		# You: Uncomment this for N-to-N (Classify all frames)
		# Y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=np.float32)
		
		X = inputs_batch.copy()
		Y = labels_batch.copy()
		
		# Generate data
		'''
		for i, ID in enumerate(list_IDs_temp):
			ic(i,ID)
			# Store sample

#			X[i,] = np.load('data/' + ID + '.npy').astype(np.float32)/255.0
#			Y[i] = np.load('labels/' + ID + '.npy')[-1].astype(np.float32)/255.

		pdb.set_trace()
		'''

	  # You: Uncomment this for N-to-N (Classify all frames)
			#Y[i] = np.load('labels/' + ID + '.npy').astype(np.float32)/255.

		return X, Y
