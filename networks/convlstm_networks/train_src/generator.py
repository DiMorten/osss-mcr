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
import deb

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, inputs, labels, batch_size=16, dim=(20,128,128), label_dim=(128,128),
				n_channels=3, n_classes=2, shuffle=False, center_pixel = False,
				augm = False):
		'Initialization'
		self.inputs = inputs
		self.dim = dim
		self.patch_size = dim[1]
		ic(self.patch_size)
		self.batch_size = batch_size
		ic(self.batch_size)
		self.labels = labels

		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.label_dim = label_dim
		self.center_pixel = center_pixel
		self.augm = augm
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
		coords_print = False
		if coords_print == True:		
			ic(X.shape)
			ic(np.min(X), np.average(X), np.max(X))
			ic(Y.shape)
			ic(np.unique(Y, return_counts=True))

			pdb.set_trace()
		
	  # You: Uncomment this for N-to-N (Classify all frames)
			#Y[i] = np.load('labels/' + ID + '.npy').astype(np.float32)/255.
		if self.augm == True:
			for idx in range(X.shape[0]):
				input_patch = X[idx]
				label_patch = Y[idx]
				transf = np.random.randint(0,6,1)
				if transf == 0:
					# rot 90
					input_patch = np.rot90(input_patch,1,(1,2))
					label_patch = np.rot90(label_patch,1,(0,1))
					
				elif transf == 1:
					# rot 180
					input_patch = np.rot90(input_patch,2,(1,2))
					label_patch = np.rot90(label_patch,2,(0,1))
					
				elif transf == 2:
					# flip horizontal
					input_patch = np.flip(input_patch,1)
					label_patch = np.flip(label_patch,0)
					
				elif transf == 3:
					# flip vertical
					input_patch = np.flip(input_patch,2)
					label_patch = np.flip(label_patch,1)
					
				elif transf == 4:
					# rot 270
					input_patch = np.rot90(input_patch,3,(1,2))
					label_patch = np.rot90(label_patch,3,(0,1))

			X[idx] = input_patch
			Y[idx] = label_patch

		return X, Y

class DataGeneratorWithCoords(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, inputs, labels, coords, batch_size=16, dim=(20,128,128), label_dim=(128,128),
				n_channels=3, n_classes=2, shuffle=False, center_pixel = False, printCoords=False,
				augm = False):
		'Initialization'
		self.inputs = inputs
		self.dim = dim
		self.batch_size = batch_size
		ic(self.batch_size)
		self.patch_size = dim[1]
		ic(self.patch_size)

		self.labels = labels

		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.label_dim = label_dim
		self.coords = coords
		self.center_pixel = center_pixel
		self.printCoords = printCoords
		self.augm = augm
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
#		return int(np.floor(len(self.list_IDs) / self.batch_size))
		n_batches = int(np.floor(self.coords.shape[0] / self.batch_size))
		ic(n_batches)
		return n_batches
	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
#		list_IDs_temp = [self.list_IDs[k] for k in indexes]
		coords_batch = self.coords[indexes]
		# Generate data
		X, y = self.__data_generation(coords_batch)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
#		self.indexes = np.arange(len(self.list_IDs))
		self.indexes = np.arange(self.coords.shape[0])

		if self.shuffle == True:
			np.random.shuffle(self.indexes)


	def data_augmentation(self, X, Y):
		transf = np.random.randint(0,6,1)
		if transf == 0:
			# rot 90
			X = np.rot90(X,1,(0,1))
			Y = np.rot90(Y,1,(0,1))
			
		elif transf == 1:
			# rot 180
			X = np.rot90(X,2,(0,1))
			Y = np.rot90(Y,2,(0,1))
			
		elif transf == 2:
			# flip horizontal
			X = np.flip(X,0)
			Y = np.flip(Y,0)
			
		elif transf == 3:
			# flip vertical
			X = np.flip(X,1)
			Y = np.flip(Y,1)
			
		elif transf == 4:
			# rot 270
			X = np.rot90(X,3,(0,1))
			Y = np.rot90(Y,3,(0,1))
		return X, Y
	def __data_generation(self, coords_batch):
	
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
#		Y = np.empty((self.batch_size, *self.label_dim, self.n_classes), dtype=int)
		Y = np.empty((self.batch_size, *self.label_dim), dtype=int)

		#deb.prints(coords_batch)
		#deb.prints(coords_batch.shape[0])
		if self.printCoords:
			ic(coords_batch)
		for idx in range(coords_batch.shape[0]):
			'''
			print(idx, coords_batch[idx], coords_batch[idx][0])
			print(coords_batch[idx][0]-self.patch_size//2)
			print(coords_batch[idx][0]+self.patch_size//2+self.patch_size%2)
			print(coords_batch[idx][1]-self.patch_size//2)
			print(coords_batch[idx][1]+self.patch_size//2+self.patch_size%2)
			ic(self.inputs.shape)
			ic(self.labels.shape)

			#pdb.set_trace()
			'''
			if self.center_pixel == False:
				input_patch = self.inputs[:, coords_batch[idx][0]:coords_batch[idx][0]+self.patch_size,
					coords_batch[idx][1]:coords_batch[idx][1]+self.patch_size]

				label_patch = self.labels[coords_batch[idx][0]:coords_batch[idx][0]+self.patch_size,
					coords_batch[idx][1]:coords_batch[idx][1]+self.patch_size]
			else:
				input_patch = self.inputs[:, coords_batch[idx][0]-self.patch_size//2:coords_batch[idx][0]+self.patch_size//2+self.patch_size%2,
						coords_batch[idx][1]-self.patch_size//2:coords_batch[idx][1]+self.patch_size//2+self.patch_size%2]

				label_patch = self.labels[coords_batch[idx][0]-self.patch_size//2:coords_batch[idx][0]+self.patch_size//2+self.patch_size%2,
						coords_batch[idx][1]-self.patch_size//2:coords_batch[idx][1]+self.patch_size//2+self.patch_size%2]
		


				
#				coords_batch[idx][0]-self.patch_size//2:coords_batch[idx][0]+self.patch_size//2+self.patch_size%2,
#						  coords_batch[idx][1]-self.patch_size//2:coords_batch[idx][1]+self.patch_size//2+self.patch_size%2]
##			ic(coords_batch[idx])
##			ic(label_patch)
##			pdb.set_trace()
			#ic(input_patch.shape)
			#ic(label_patch.shape)
			#pdb.set_trace()
			#ic(X.shape, Y.shape)
			#X, Y = self.data_augmentation(X, Y)
			#ic(X.shape, Y.shape)
#			self.augm = True


			if self.augm == True:
				transf = np.random.randint(0,6,1)
				if transf == 0:
					# rot 90
					input_patch = np.rot90(input_patch,1,(1,2))
					label_patch = np.rot90(label_patch,1,(0,1))
					
				elif transf == 1:
					# rot 180
					input_patch = np.rot90(input_patch,2,(1,2))
					label_patch = np.rot90(label_patch,2,(0,1))
					
				elif transf == 2:
					# flip horizontal
					input_patch = np.flip(input_patch,1)
					label_patch = np.flip(label_patch,0)
					
				elif transf == 3:
					# flip vertical
					input_patch = np.flip(input_patch,2)
					label_patch = np.flip(label_patch,1)
					
				elif transf == 4:
					# rot 270
					input_patch = np.rot90(input_patch,3,(1,2))
					label_patch = np.rot90(label_patch,3,(0,1))

			'''
			# convert to one-hot
			def toOneHot(label):
				label_shape = label.shape
				ic(label[0])
				ic(label_shape)
#				ic((*label_shape[:-1], -1))
				label = np.reshape(label, -1)
				b = np.zeros((label.shape[0], self.n_classes))
				b[np.arange(label.shape[0], label)] = 1
				ic(b.shape)

				b = np.reshape(b, label_shape)
				ic(b.shape)
				ic(b[0])
				pdb.set_trace()

				return b
			'''
			X[idx] = input_patch
#			Y[idx] = toOneHot(label_patch)
			Y[idx] = label_patch
		coords_print = False
		if coords_print == True:
			ic(coords_batch)
			ic(X.shape)
			ic(np.min(X), np.average(X), np.max(X))
			ic(Y.shape)
			ic(np.unique(Y, return_counts=True))
			pdb.set_trace()
		
		return X, np.expand_dims(Y, axis=-1)

class DataGeneratorWithCoordsPatches(DataGeneratorWithCoords):
	def __init__(self, inputs_patches, labels_patches,
					inputs, labels, coords, batch_size=16, dim=(20,128,128), label_dim=(128,128),
					n_channels=3, n_classes=2, shuffle=False, center_pixel = False, printCoords=False,
					augm = False):

		self.inputs_patches = inputs_patches
		self.labels_patches = labels_patches
		super().__init__(inputs, labels, coords, batch_size, dim, label_dim,
					n_channels, n_classes, shuffle, center_pixel, printCoords,
					augm)
	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
#		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		inputs_batch = self.inputs_patches[indexes]
		labels_batch = self.labels_patches[indexes]
		coords_batch = self.coords[indexes]		
		ic(index)
		# Generate data
		X, y = self.__data_generation(inputs_batch, labels_batch, coords_batch)

		return X, y
	def __data_generation(self, inputs_batch, labels_batch, coords_batch):
#		X, Y = super().__data_generation(coords_batch)
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
		Y = np.empty((self.batch_size, *self.label_dim), dtype=int)

		#deb.prints(coords_batch)
		#deb.prints(coords_batch.shape[0])
		if self.printCoords:
			ic(coords_batch)
		for idx in range(coords_batch.shape[0]):

			if self.center_pixel == False:
				input_patch = self.inputs[:, coords_batch[idx][0]:coords_batch[idx][0]+self.patch_size,
					coords_batch[idx][1]:coords_batch[idx][1]+self.patch_size]

				label_patch = self.labels[coords_batch[idx][0]:coords_batch[idx][0]+self.patch_size,
					coords_batch[idx][1]:coords_batch[idx][1]+self.patch_size]
			else:
				input_patch = self.inputs[:, coords_batch[idx][0]-self.patch_size//2:coords_batch[idx][0]+self.patch_size//2+self.patch_size%2,
						coords_batch[idx][1]-self.patch_size//2:coords_batch[idx][1]+self.patch_size//2+self.patch_size%2]

				label_patch = self.labels[coords_batch[idx][0]-self.patch_size//2:coords_batch[idx][0]+self.patch_size//2+self.patch_size%2,
						coords_batch[idx][1]-self.patch_size//2:coords_batch[idx][1]+self.patch_size//2+self.patch_size%2]
		


				
#				coords_batch[idx][0]-self.patch_size//2:coords_batch[idx][0]+self.patch_size//2+self.patch_size%2,
#						  coords_batch[idx][1]-self.patch_size//2:coords_batch[idx][1]+self.patch_size//2+self.patch_size%2]
##			ic(coords_batch[idx])
##			ic(label_patch)
##			pdb.set_trace()
			#ic(input_patch.shape)
			#ic(label_patch.shape)
			#pdb.set_trace()
			#ic(X.shape, Y.shape)
			#X, Y = self.data_augmentation(X, Y)
			#ic(X.shape, Y.shape)
#			self.augm = True


			if self.augm == True:
				transf = np.random.randint(0,6,1)
				if transf == 0:
					# rot 90
					input_patch = np.rot90(input_patch,1,(1,2))
					label_patch = np.rot90(label_patch,1,(0,1))
					
				elif transf == 1:
					# rot 180
					input_patch = np.rot90(input_patch,2,(1,2))
					label_patch = np.rot90(label_patch,2,(0,1))
					
				elif transf == 2:
					# flip horizontal
					input_patch = np.flip(input_patch,1)
					label_patch = np.flip(label_patch,0)
					
				elif transf == 3:
					# flip vertical
					input_patch = np.flip(input_patch,2)
					label_patch = np.flip(label_patch,1)
					
				elif transf == 4:
					# rot 270
					input_patch = np.rot90(input_patch,3,(1,2))
					label_patch = np.rot90(label_patch,3,(0,1))

			'''
			# convert to one-hot
			def toOneHot(label):
				label_shape = label.shape
				ic(label[0])
				ic(label_shape)
#				ic((*label_shape[:-1], -1))
				label = np.reshape(label, -1)
				b = np.zeros((label.shape[0], self.n_classes))
				b[np.arange(label.shape[0], label)] = 1
				ic(b.shape)

				b = np.reshape(b, label_shape)
				ic(b.shape)
				ic(b[0])
				pdb.set_trace()

				return b
			'''
			X[idx] = input_patch
#			Y[idx] = toOneHot(label_patch)
			Y[idx] = label_patch
		coords_print = False
		if coords_print == True:
			ic(coords_batch)
			ic(X.shape)
			ic(np.min(X), np.average(X), np.max(X))
			ic(Y.shape)
			ic(np.unique(Y, return_counts=True))
			pdb.set_trace()

		Y = np.expand_dims(Y,axis=-1)
		X_patches = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
		Y_patches = np.empty((self.batch_size, *self.label_dim), dtype=int)
		X_patches = inputs_batch.copy()
		Y_patches = labels_batch.copy()

		ic(coords_batch)
		ic(X_patches.shape)
		ic(np.min(X_patches), np.average(X_patches), np.max(X_patches))
		ic(Y_patches.shape)
		ic(np.unique(Y_patches, return_counts=True))

		pdb.set_trace()
		return X, Y