import cv2
import os
import glob
import numpy as np
import pdb
from pathlib import Path
import csv
import threading
import keras
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, ConvLSTM2D, Activation, BatchNormalization, Bidirectional, TimeDistributed, AveragePooling2D
from keras.models import Model
from keras.regularizers import l1,l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, label_IDs, batch_size=6, dim=(20,128,128), label_dim=(128,128), n_channels=3,
				 n_classes=2, shuffle=True):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.label_IDs = label_IDs
		self.list_IDs = list_IDs
		#print("self.list_IDs",self.list_IDs)
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.label_dim = label_dim
		print("self.label_dim",self.label_dim)
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		print("Generating 1 batch...")
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]
		label_IDs_temp = [self.label_IDs[k] for k in indexes]
		# Generate data
		X, y = self.__data_generation(list_IDs_temp,label_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		print("EPOCH END",self.indexes)
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp,label_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		Y = np.empty((self.batch_size, *self.label_dim), dtype=int)
		print('Y shape',Y.shape)
		#print(list_IDs_temp)
		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample

			X[i,] = np.load(ID)

			# Store class
			#print(ID)
			#print(np.load('labels/' + ID + '.npy')[-1].shape)
			#Y[i,] = np.load(label_IDs_temp[i])[:,:,:,:-1] # Only last frame is segmented
			Y[i,] = np.expand_dims(np.load(label_IDs_temp[i]).argmax(axis=3),axis=3).astype(np.int8)
		#return X, keras.utils.to_categorical(Y, num_classes=self.n_classes)
		return X, Y
