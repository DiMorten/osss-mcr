
from colorama import init
init()
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose
# from tensorflow.keras.callbacks import ModelCheckpoint , EarlyStopping
from tensorflow.keras.optimizers import Adam,Adagrad 
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow.keras as keras

import numpy as np
from sklearn.utils import shuffle
import cv2
import argparse
import tensorflow as tf

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import metrics
import sys
import glob

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report
# Local
from densnet import DenseNetFCN
from densnet_timedistributed import DenseNetFCNTimeDistributed

#from metrics import fmeasure,categorical_accuracy
import deb
from keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy, sparse_accuracy_ignoring_last_label, weighted_categorical_crossentropy_ignoring_last_label, categorical_focal_ignoring_last_label, weighted_categorical_focal_ignoring_last_label
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ConvLSTM2D, UpSampling2D, multiply
from tensorflow.keras.regularizers import l1,l2
import time
import pickle
#from keras_self_attention import SeqSelfAttention
import pdb
import pathlib
from pathlib import Path, PureWindowsPath
from tensorflow.keras.layers import Conv3DTranspose, Conv3D

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from collections import Counter

import matplotlib.pyplot as plt
sys.path.append('../../../dataset/dataset/patches_extract_script/')
from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq

from icecream import ic
from tensorflow.keras import layers


class ModelArchitecture():

	def __init__(self, t_len, patch_len, channel_n):
		self.t_len = t_len
		self.model_t_len = t_len
		self.patch_len = patch_len
		self.channel_n = channel_n

	def __repr__(self):
		return "DefaultArchitecture"
	def build(self):
		deb.prints(self.t_len)
		deb.prints(self.model_t_len)

		self.in_im = Input(shape=(self.model_t_len,self.patch_len, self.patch_len, self.channel_n))
		self.weight_decay=1E-4

	def dilated_layer(self, x,filter_size,dilation_rate=1, kernel_size=3):
		x = TimeDistributed(Conv2D(filter_size, kernel_size, padding='same',
			dilation_rate=(dilation_rate, dilation_rate)))(x)
		x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
							beta_regularizer=l2(self.weight_decay))(x)
		x = Activation('relu')(x)
		return x


	def dilated_layer_3D(self, x,filter_size,dilation_rate=1, kernel_size=3):
		if isinstance(dilation_rate, int):
			dilation_rate = (dilation_rate, dilation_rate, dilation_rate)
		x = Conv3D(filter_size, kernel_size, padding='same',
			dilation_rate=dilation_rate)(x)
		x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
							beta_regularizer=l2(self.weight_decay))(x)
		x = Activation('relu')(x)
		return x
	def transpose_layer(self, x,filter_size,dilation_rate=1,
		kernel_size=3, strides=(2,2)):
		x = Conv2DTranspose(filter_size, 
			kernel_size, strides=strides, padding='same')(x)
		x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
											beta_regularizer=l2(self.weight_decay))(x)
		x = Activation('relu')(x)
		return x	

	def transpose_layer_3D(self, x,filter_size,dilation_rate=1,
		kernel_size=3, strides=(1,2,2)):
		x = Conv3DTranspose(filter_size,
			kernel_size, strides=strides, padding='same')(x)
		x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
											beta_regularizer=l2(self.weight_decay))(x)
		x = Activation('relu')(x)
		return x	
	def im_pooling_layer(self, x,filter_size):
		pooling=True
		shape_before=tf.shape(x)
		print("im pooling")
		deb.prints(K.int_shape(x))
		if pooling==True:
			mode=2
			if mode==1:
				x=TimeDistributed(GlobalAveragePooling2D())(x)
				deb.prints(K.int_shape(x))
				x=K.expand_dims(K.expand_dims(x,2),2)
			elif mode==2:
				x=TimeDistributed(AveragePooling2D((32,32)))(x)
				deb.prints(K.int_shape(x))
				
		deb.prints(K.int_shape(x))
		#x=dilated_layer(x,filter_size,1,kernel_size=1)
		deb.prints(K.int_shape(x))

		if pooling==True:
			x = TimeDistributed(Lambda(lambda y: K.tf.image.resize_bilinear(y,size=(32,32))))(x)
#				x = TimeDistributed(UpSampling2D(
#					size=(self.patch_len,self.patch_len)))(x)
		deb.prints(K.int_shape(x))
		print("end im pooling")
		# x=TimeDistributed(Lambda(
		# 	lambda y: tf.compat.v1.image.resize(
		# 		y, shape_before[2:4],
		# 		method='bilinear',align_corners=True)))(x)
		return x
	def spatial_pyramid_pooling(self, x,filter_size,
		max_rate=8,global_average_pooling=False):
		x=[]
		if max_rate>=1:
			x.append(self.dilated_layer(x,filter_size,1)) # (1,1,1)
		if max_rate>=2:
			x.append(self.dilated_layer(x,filter_size,2)) #6 (1,2,2)
		if max_rate>=4:
			x.append(self.dilated_layer(x,filter_size,4)) #12 (2,4,4)
		if max_rate>=8:
			x.append(self.dilated_layer(x,filter_size,8)) #18 (4,8,8)
		if global_average_pooling==True:
			x.append(im_pooling_layer(x,filter_size))
		out = keras.layers.concatenate(x, axis=-1)
		return out

	def spatial_pyramid_pooling_3D(self, x,filter_size,
		max_rate=8,global_average_pooling=False):
		x=[]
		if max_rate>=1:
			x.append(self.dilated_layer_3D(x,filter_size,1))
		if max_rate>=2:
			x.append(self.dilated_layer_3D(x,filter_size,(1,2,2))) #6
		if max_rate>=4:
			x.append(self.dilated_layer_3D(x,filter_size,(2,4,4))) #12
		if max_rate>=8:
			x.append(self.dilated_layer_3D(x,filter_size,(4,8,8))) #18
		if global_average_pooling==True:
			x.append(im_pooling_layer(x,filter_size))
		out = keras.layers.concatenate(x, axis=-1)
		return out

	def temporal_pyramid_pooling(self, x,filter_size,
		max_rate=4):
		x=[]
		if max_rate>=1:
			x.append(self.dilated_layer_3D(x,filter_size,1))
		if max_rate>=2:
			x.append(self.dilated_layer_3D(x,filter_size,(2,1,1))) #2
		if max_rate>=3:
			x.append(self.dilated_layer_3D(x,filter_size,(3,1,1))) #2
		if max_rate>=4:
			x.append(self.dilated_layer_3D(x,filter_size,(4,1,1))) #4
		if max_rate>=5:
			x.append(self.dilated_layer_3D(x,filter_size,(5,1,1))) #4
		out = keras.layers.concatenate(x, axis=4)
		return out

	def full_pyramid_pooling(self, x,filter_size,
		max_rate=4):
		x=[]
		if max_rate>=1:
			x.append(self.dilated_layer_3D(x,filter_size,1))
		if max_rate>=2:
			x.append(self.dilated_layer_3D(x,filter_size,(2,2,2))) #2
		if max_rate>=3:
			x.append(self.dilated_layer_3D(x,filter_size,(3,3,3))) #2
		if max_rate>=4:
			x.append(self.dilated_layer_3D(x,filter_size,(4,4,4))) #4
		out = keras.layers.concatenate(x, axis=4)
		return out

	def unetEncoder(self, x, fs):
		p1=self.dilated_layer(x,fs)			
		p1=self.dilated_layer(p1,fs)
		e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
		p2=self.dilated_layer(e1,fs*2)
		e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
		p3=self.dilated_layer(e2,fs*4)
		e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)
		return e3, p1, p2, p3

	def unetDecoder(self, x, p1, p2, p3):
		d3 = self.transpose_layer(x,fs*4)
		d3 = keras.layers.concatenate([d3, p3], axis=4)
		d2 = self.transpose_layer(d3,fs*4)
		d2 = keras.layers.concatenate([d2, p2], axis=4)
		d1 = self.transpose_layer(d2,fs*2)
		d1 = keras.layers.concatenate([d1, p1], axis=4)
		out=self.dilated_layer(d1,fs)
		return out

class UnetConvLSTM(ModelArchitecture):
	def build(self):
		super().build()		
		#fs=32
		fs=16

		e3, p1, p2, p3 = self.unetEncoder(self.in_im, fs)

		x = ConvLSTM2D(256,3,return_sequences=True,
				padding="same")(e3)

		out = self.unetDecoder(x, p1, p2, p3)

		out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
									padding='same'))(out)
		self.graph = Model(self.in_im, out)
		print(self.graph.summary())

class BUnetConvLSTM(ModelArchitecture):
	def build(self):
		super().build()		
		fs=16

		e3, p1, p2, p3 = self.unetEncoder(self.in_im, fs)

		x = Bidirectional(ConvLSTM2D(128,3,return_sequences=False,
				padding="same"),merge_mode='concat')(e3)
				
		out = self.unetDecoder(x, p1, p2, p3)

		out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
									padding='same'))(out)
		self.graph = Model(self.in_im, out)
		print(self.graph.summary())


# ==== N to 1 ======


class UnetConvLSTM_Skip(ModelArchitecture):
	def build(self):
		super().build()
		#fs=32
		fs=16

		p1=self.dilated_layer(self.in_im,fs)
		p1=self.dilated_layer(p1,fs)
		x_p1 = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
				padding="same"),merge_mode='concat')(p1)
		e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
		p2=self.dilated_layer(e1,fs*2)
		x_p2 = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
				padding="same"),merge_mode='concat')(p2)
		e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
		p3=self.dilated_layer(e2,fs*4)
		x_p3 = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
				padding="same"),merge_mode='concat')(p3)
		e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

		x = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
				padding="same"),merge_mode='concat')(e3)

		d3 = self.transpose_layer(x,fs*4)
		d3 = keras.layers.concatenate([d3, x_p3], axis=4)
		d3 = self.dilated_layer(d3,fs*4)
		d2 = self.transpose_layer(d3,fs*2)
		d2 = keras.layers.concatenate([d2, x_p2], axis=4)
		d2 = self.dilated_layer(d2,fs*2)
		d1 = self.transpose_layer(d2,fs)
		d1 = keras.layers.concatenate([d1, x_p1], axis=4)
		out = self.dilated_layer(d1,fs)
		out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
									padding='same'))(out)
		self.graph = Model(self.in_im, out)
		print(self.graph.summary())


class Unet3D(ModelArchitecture):
	def build(self):
		super().build()
		fs=32
		#fs=16

		p1=self.dilated_layer_3D(self.in_im,fs,kernel_size=(7,3,3))
		e1 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p1)
		p2=self.dilated_layer_3D(e1,fs*2,kernel_size=(7,3,3))
		e2 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p2)
		p3=self.dilated_layer_3D(e2,fs*4,kernel_size=(7,3,3))
		e3 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p3)

		d3 = self.transpose_layer_3D(e3,fs*4)
		d3 = keras.layers.concatenate([d3, p3], axis=4)
		d3 = self.dilated_layer_3D(d3,fs*4,kernel_size=(7,3,3))
		d2 = self.transpose_layer_3D(d3,fs*2)
		d2 = keras.layers.concatenate([d2, p2], axis=4)
		d2 = self.dilated_layer_3D(d2,fs*2,kernel_size=(7,3,3))
		d1 = self.transpose_layer_3D(d2,fs)
		d1 = keras.layers.concatenate([d1, p1], axis=4)
		out = self.dilated_layer_3D(d1,fs,kernel_size=(7,3,3))
		out = Conv3D(self.class_n, (1, 1, 1), activation=None,
									padding='same')(out)
		self.graph = Model(self.in_im, out)
		print(self.graph.summary())

class Unet3D_ATPP(ModelArchitecture):
	def build(self):
		super().build()
		fs=16
		max_rate=5
		#fs=16

		p1=self.temporal_pyramid_pooling(self.in_im,fs,max_rate)
		e1 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p1)
		p2=self.temporal_pyramid_pooling(e1,fs*2,max_rate)
		e2 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p2)
		p3=self.temporal_pyramid_pooling(e2,fs*4,max_rate)
		e3 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p3)

		d3 = self.transpose_layer_3D(e3,fs*4)
		d3 = keras.layers.concatenate([d3, p3], axis=4)
		d3 = temporal_pyramid_pooling(d3,fs*4,max_rate)
		d2 = self.transpose_layer_3D(d3,fs*2)
		d2 = keras.layers.concatenate([d2, p2], axis=4)
		d3 = temporal_pyramid_pooling(d2,fs*2,max_rate)
		d1 = self.transpose_layer_3D(d2,fs)
		d1 = keras.layers.concatenate([d1, p1], axis=4)
		out = temporal_pyramid_pooling(d1,fs,max_rate)
		out = Conv3D(self.class_n, (1, 1, 1), activation=None,
									padding='same')(out)
		self.graph = Model(self.in_im, out)
		print(self.graph.summary())

class BAtrousGAPConvLSTM(ModelArchitecture):
	def build(self):
		super().build()
		#fs=32
		fs=16
		
		#x=dilated_layer(self.in_im,fs)
		x=self.dilated_layer(self.in_im,fs)
		x=self.dilated_layer(x,fs)
		x=self.spatial_pyramid_pooling(x,fs*4,max_rate=8,
			global_average_pooling=True)
		
		x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"),merge_mode='concat')(x)

		out=self.dilated_layer(x,fs)
		out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
									padding='same'))(out)
		self.graph = Model(self.in_im, out)
		print(self.graph.summary())
class ModelArchitectureNto1(ModelArchitecture):

	def slice_tensor(self, x,output_shape):
		deb.prints(output_shape)
		deb.prints(K.int_shape(x))
#				res1 = Lambda(lambda x: x[:,:,:,-1], output_shape=output_shape)(x)
#				res2 = Lambda(lambda x: x[:,:,:,-1], output_shape=output_shape[1:])(x)
		res2 = Lambda(lambda x: x[:,-1])(x)

#				deb.prints(K.int_shape(res1))
		deb.prints(K.int_shape(res2))
		
		return res2

	def dilated_layer_Nto1(self, x,filter_size,dilation_rate=1, kernel_size=3):
		x = Conv2D(filter_size, kernel_size, padding='same',
			dilation_rate=(dilation_rate, dilation_rate))(x)
		x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
							beta_regularizer=l2(self.weight_decay))(x)
		x = Activation('relu')(x)
		return x
		
	def unetEncoderNto1(self, x, fs):

		p1=self.dilated_layer(x,fs)			
		p1=self.dilated_layer(p1,fs)
		e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
		p2=self.dilated_layer(e1,fs*2)
		e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
		p3=self.dilated_layer(e2,fs*4)
		e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)
		return e3, p1, p2, p3

	def unetDecoderNto1(self, x, p1, p2, p3, fs):
		d3 = self.transpose_layer(x,fs*4)
		p3 = self.slice_tensor(p3, output_shape = K.int_shape(d3))
		deb.prints(K.int_shape(p3))
		deb.prints(K.int_shape(d3))
		
		d3 = keras.layers.concatenate([d3, p3], axis=-1)
		d3=self.dilated_layer_Nto1(d3,fs*4)
		d2 = self.transpose_layer(d3,fs*2)
		p2 = self.slice_tensor(p2, output_shape = K.int_shape(d2))
		deb.prints(K.int_shape(p2))
		deb.prints(K.int_shape(d2))

		d2 = keras.layers.concatenate([d2, p2], axis=-1)
		d2=self.dilated_layer_Nto1(d2,fs*2)
		d1 = self.transpose_layer(d2,fs)
		p1 = self.slice_tensor(p1, output_shape = K.int_shape(d1))
		deb.prints(K.int_shape(p1))
		deb.prints(K.int_shape(d1))

		d1 = keras.layers.concatenate([d1, p1], axis=-1)
		out=self.dilated_layer_Nto1(d1,fs)
		return out

class UUnetConvLSTM(ModelArchitectureNto1):
	def __repr__(self):
		return "UUnetConvLSTM"
	def build(self):
		super().build()		
		concat_axis = 3
		#fs=32
		fs=16

		e3, p1, p2, p3 = self.unetEncoderNto1(self.in_im, fs)

		x = ConvLSTM2D(256,3,return_sequences=False,
				padding="same")(e3)

		out = self.unetDecoderNto1(x, p1, p2, p3, fs)

		out = Conv2D(self.class_n, (1, 1), activation=None,
									padding='same')(out)
		self.graph = Model(self.in_im, out)
		print(self.graph.summary())
		
class UnetSelfAttention(ModelArchitectureNto1):
	def build(self):
		super().build()		
		concat_axis = 3
		#fs=32
		fs=16

		e3, p1, p2, p3 = self.unetEncoderNto1(self.in_im, fs)

		key_dim = 8
		num_heads = 8
		dropout = 0.

		# shape (n_samples, t_len, h, w, channels)	
		deb.prints(K.int_shape(e3))	
		x = layers.MultiHeadAttention(
				key_dim=key_dim, num_heads=num_heads, dropout=dropout, 
				attention_axes=(2, 3)
			)(x, x)
		deb.prints(K.int_shape(x))	
		pdb.set_trace()
#		x = ConvLSTM2D(256,3,return_sequences=False,
#				padding="same")(e3)

		out = self.unetDecoderNto1(x, p1, p2, p3)

		out = Conv2D(self.class_n, (1, 1), activation=None,
									padding='same')(out)
		self.graph = Model(self.in_im, out)
		print(self.graph.summary())
		