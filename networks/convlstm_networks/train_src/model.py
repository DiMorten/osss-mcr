
from colorama import init
init()
from utils import *
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose
# from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.optimizers import Adam,Adagrad 
from keras.models import Model
from keras import backend as K
import keras

import numpy as np
from sklearn.utils import shuffle
import cv2
import argparse
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import metrics
import sys
import glob

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report
# Local
from densnet import DenseNetFCN
from densnet_timedistributed import DenseNetFCNTimeDistributed

#from metrics import fmeasure,categorical_accuracy
import deb
from keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy, sparse_accuracy_ignoring_last_label, weighted_categorical_crossentropy_ignoring_last_label, categorical_focal_ignoring_last_label, weighted_categorical_focal_ignoring_last_label
from keras.models import load_model
from keras.layers import ConvLSTM2D, UpSampling2D, multiply
from keras.utils.vis_utils import plot_model
from keras.regularizers import l1,l2
import time
import pickle
#from keras_self_attention import SeqSelfAttention
import pdb
import pathlib
from pathlib import Path, PureWindowsPath
from patches_handler import PatchesArray
from keras.layers import Conv3DTranspose, Conv3D

from keras.callbacks import EarlyStopping
import tensorflow as tf
from collections import Counter

from patches_storage import PatchesStorageEachSample,PatchesStorageAllSamples, PatchesStorageAllSamplesOpenSet
#from datagenerator import DataGenerator
from generator import DataGenerator, DataGeneratorWithCoords, DataGeneratorWithCoordsPatches

import matplotlib.pyplot as plt
sys.path.append('../../../dataset/dataset/patches_extract_script/')
from dataSource import DataSource, SARSource, OpticalSource, Dataset, LEM, LEM2, CampoVerde, OpticalSourceWithClouds, Humidity
from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq
from parameters.parameters_reader import ParamsTrain

from icecream import ic
from monitor import Monitor, MonitorNPY, MonitorGenerator, MonitorNPYAndGenerator
import natsort

def load_obj(name ):
	with open('obj/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class NetModel(object):
	def __init__(self, paramsTrain, ds, 
		*args, **kwargs):

		print("Initializing object...")
		print(paramsTrain.t_len, paramsTrain.channel_n)
		self.patch_len = paramsTrain.patch_len
		self.path = {"v": paramsTrain.path, 'train': {}, 'test': {}}
		self.image = {'train': {}, 'test': {}}
		self.patches = {'train': {}, 'test': {}}

		self.patches['train']['step']=paramsTrain.patch_step_train
		self.patches['test']['step']=paramsTrain.patch_step_test 
      
		self.path['train']['in'] = paramsTrain.path + 'train_test/train/ims/'
		self.path['test']['in'] = paramsTrain.path + 'train_test/test/ims/'
		self.path['train']['label'] = paramsTrain.path + 'train_test/train/labels/'
		self.path['test']['label'] = paramsTrain.path + 'train_test/test/labels/'

		# in these paths, the augmented train set and validation set are stored
		# they can be loaded after (flag decides whether estimating these values and storing,
		# or loading the precomputed ones)
		self.path_patches_bckndfixed = paramsTrain.path + 'patches_bckndfixed/' 
		self.path['train_bckndfixed']=self.path_patches_bckndfixed+'train/'
		self.path['val_bckndfixed']=self.path_patches_bckndfixed+'val/'
		self.path['test_bckndfixed']=self.path_patches_bckndfixed+'test/'
		self.path['test_loco'] = self.path_patches_bckndfixed+'test_loco/'

		self.channel_n = paramsTrain.channel_n
		deb.prints(self.channel_n)
		self.debug = paramsTrain.debug
		self.class_n = paramsTrain.class_n
		self.report={'best':{}, 'val':{}}
		self.report['exp_id']=paramsTrain.exp_id
		self.report['best']['text_name']='result_'+paramsTrain.exp_id+'.txt'
		self.report['best']['text_path']='../results/'+self.report['best']['text_name']
		self.report['best']['text_history_path']='../results/'+'history.txt'
		self.report['val']['history_path']='../results/'+'history_val.txt'
		
		self.t_len=paramsTrain.t_len
		deb.prints(self.t_len)
		self.dotys_sin_cos = paramsTrain.dotys_sin_cos
		self.dotys_sin_cos = np.expand_dims(self.dotys_sin_cos,axis=0) # add batch dimension
		self.dotys_sin_cos = np.repeat(self.dotys_sin_cos,16,axis=0)
		self.ds = ds

#		super().__init__(*args, **kwargs)
		if self.debug >= 1:
			print("Initializing Model instance")
		self.val_set = paramsTrain.val_set
		self.metrics = {'train': {}, 'test': {}, 'val':{}}
		self.batch = {'train': {}, 'test': {}, 'val':{}}
		self.batch['train']['size'] = paramsTrain.batch_size_train
		self.batch['test']['size'] = paramsTrain.batch_size_test
		self.batch['val']['size'] = paramsTrain.batch_size_test
		
		self.eval_mode = paramsTrain.eval_mode # legacy
		self.epochs = paramsTrain.epochs # legacy?
		self.early_stop={'best':0,
					'count':0,
					'signal':False,
					'patience':paramsTrain.patience}
		self.model_type=paramsTrain.model_type
		if self.model_type == 'UUnet4ConvLSTM_doty':		
			self.doty_flag = True
		else:
			self.doty_flag = False
		with open(self.report['best']['text_history_path'], "w") as text_file:
			text_file.write("epoch,oa,aa,f1,class_acc\n")

		with open(self.report['val']['history_path'], "w") as text_file:
			text_file.write("epoch,oa,aa,f1,class_acc\n")

		self.model_save=True
		self.time_measure=paramsTrain.time_measure
		self.mp=load_obj('model_params')
		deb.prints(self.mp)
		self.stop_epoch = paramsTrain.stop_epoch
		deb.prints(self.stop_epoch)

		self.model_t_len = paramsTrain.model_t_len
		self.mim = paramsTrain.mim
	def transition_down(self, pipe, filters):
		pipe = Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(pipe)
		pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		pipe = Activation('relu')(pipe)
		#pipe = Conv2D(filters, (1, 1), padding='same')(pipe)
		#pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		#pipe = Activation('relu')(pipe)
		
		return pipe

	def dense_block(self, pipe, filters):
		pipe = Conv2D(filters, (3, 3), padding='same')(pipe)
		pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		pipe = Activation('relu')(pipe)
		return pipe

	def transition_up(self, pipe, filters):
		pipe = Conv2DTranspose(filters, (3, 3), strides=(
			2, 2), padding='same')(pipe)
		pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		pipe = Activation('relu')(pipe)
		#pipe = Dropout(0.2)(pipe)
		#pipe = Conv2D(filters, (1, 1), padding='same')(pipe)
		#pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		#pipe = Activation('relu')(pipe)
		return pipe

	def concatenate_transition_up(self, pipe1, pipe2, filters):
		pipe = keras.layers.concatenate([pipe1, pipe2], axis=3)
		pipe = self.transition_up(pipe, filters)
		return pipe


	def build(self):
		deb.prints(self.t_len)
		deb.prints(self.model_t_len)

		in_im = Input(shape=(self.model_t_len,self.patch_len, self.patch_len, self.channel_n))
		weight_decay=1E-4
		def dilated_layer(x,filter_size,dilation_rate=1, kernel_size=3):
			x = TimeDistributed(Conv2D(filter_size, kernel_size, padding='same',
				dilation_rate=(dilation_rate, dilation_rate)))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			x = Activation('relu')(x)
			return x
		def dilated_layer_Nto1(x,filter_size,dilation_rate=1, kernel_size=3):
			x = Conv2D(filter_size, kernel_size, padding='same',
				dilation_rate=(dilation_rate, dilation_rate))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			x = Activation('relu')(x)
			return x

		def dilated_layer_3D(x,filter_size,dilation_rate=1, kernel_size=3):
			if isinstance(dilation_rate, int):
				dilation_rate = (dilation_rate, dilation_rate, dilation_rate)
			x = Conv3D(filter_size, kernel_size, padding='same',
				dilation_rate=dilation_rate)(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			x = Activation('relu')(x)
			return x
		def transpose_layer(x,filter_size,dilation_rate=1,
			kernel_size=3, strides=(2,2)):
			x = Conv2DTranspose(filter_size, 
				kernel_size, strides=strides, padding='same')(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(x)
			x = Activation('relu')(x)
			return x	

		def transpose_layer_3D(x,filter_size,dilation_rate=1,
			kernel_size=3, strides=(1,2,2)):
			x = Conv3DTranspose(filter_size,
				kernel_size, strides=strides, padding='same')(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(x)
			x = Activation('relu')(x)
			return x	
		def im_pooling_layer(x,filter_size):
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
		def spatial_pyramid_pooling(in_im,filter_size,
			max_rate=8,global_average_pooling=False):
			x=[]
			if max_rate>=1:
				x.append(dilated_layer(in_im,filter_size,1)) # (1,1,1)
			if max_rate>=2:
				x.append(dilated_layer(in_im,filter_size,2)) #6 (1,2,2)
			if max_rate>=4:
				x.append(dilated_layer(in_im,filter_size,4)) #12 (2,4,4)
			if max_rate>=8:
				x.append(dilated_layer(in_im,filter_size,8)) #18 (4,8,8)
			if global_average_pooling==True:
				x.append(im_pooling_layer(in_im,filter_size))
			out = keras.layers.concatenate(x, axis=-1)
			return out

		def spatial_pyramid_pooling_3D(in_im,filter_size,
			max_rate=8,global_average_pooling=False):
			x=[]
			if max_rate>=1:
				x.append(dilated_layer_3D(in_im,filter_size,1))
			if max_rate>=2:
				x.append(dilated_layer_3D(in_im,filter_size,(1,2,2))) #6
			if max_rate>=4:
				x.append(dilated_layer_3D(in_im,filter_size,(2,4,4))) #12
			if max_rate>=8:
				x.append(dilated_layer_3D(in_im,filter_size,(4,8,8))) #18
			if global_average_pooling==True:
				x.append(im_pooling_layer(in_im,filter_size))
			out = keras.layers.concatenate(x, axis=-1)
			return out

		def temporal_pyramid_pooling(in_im,filter_size,
			max_rate=4):
			x=[]
			if max_rate>=1:
				x.append(dilated_layer_3D(in_im,filter_size,1))
			if max_rate>=2:
				x.append(dilated_layer_3D(in_im,filter_size,(2,1,1))) #2
			if max_rate>=3:
				x.append(dilated_layer_3D(in_im,filter_size,(3,1,1))) #2
			if max_rate>=4:
				x.append(dilated_layer_3D(in_im,filter_size,(4,1,1))) #4
			if max_rate>=5:
				x.append(dilated_layer_3D(in_im,filter_size,(5,1,1))) #4
			out = keras.layers.concatenate(x, axis=4)
			return out

		def full_pyramid_pooling(in_im,filter_size,
			max_rate=4):
			x=[]
			if max_rate>=1:
				x.append(dilated_layer_3D(in_im,filter_size,1))
			if max_rate>=2:
				x.append(dilated_layer_3D(in_im,filter_size,(2,2,2))) #2
			if max_rate>=3:
				x.append(dilated_layer_3D(in_im,filter_size,(3,3,3))) #2
			if max_rate>=4:
				x.append(dilated_layer_3D(in_im,filter_size,(4,4,4))) #4
			out = keras.layers.concatenate(x, axis=4)
			return out

		if self.model_type=='DenseNet':


			#x = keras.layers.Permute((1,2,0,3))(in_im)
			x = keras.layers.Permute((2,3,1,4))(in_im)
			
			x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCN((self.patch_len, self.patch_len, self.t_len*self.channel_n), nb_dense_block=2, growth_rate=16, dropout_rate=0.2,
							nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='ConvLSTM_DenseNet':
			# convlstm then densenet

			#x = keras.layers.Permute((2,3,1,4))(in_im)
			x = ConvLSTM2D(32,3,return_sequences=False,padding="same")(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCN((self.patch_len, self.patch_len, self.channel_n), nb_dense_block=2, growth_rate=16, dropout_rate=0.2,
							nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='ConvLSTM':
			x = ConvLSTM2D(32,3,return_sequences=False,padding="same")(in_im)
			out = Conv2D(self.class_n, (1, 1), activation='softmax',
						 padding='same')(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		elif self.model_type=='FCN_ConvLSTM':
			x = TimeDistributed(Conv2D(32, (3, 3), padding='same'))(in_im)
			x = ConvLSTM2D(32,3,return_sequences=False,padding="same")(x)
			out = Conv2D(self.class_n, (1, 1), activation='softmax',
						 padding='same')(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='FCN_ConvLSTM2':
			e1 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				strides=(2, 2)))(in_im)
			e2 = TimeDistributed(Conv2D(32, (3, 3), padding='same',
				strides=(2, 2)))(e1)
			e3 = TimeDistributed(Conv2D(48, (3, 3), padding='same',
				strides=(2, 2)))(e2)

			x = ConvLSTM2D(80,3,return_sequences=False,padding="same")(e3)


			d3 = Conv2DTranspose(48, (3, 3), strides=(
				2, 2), padding='same')(x)
			#d2 = keras.layers.concatenate([d3, e2[:,-1,:,:,:]], axis=3)

			d2 = Conv2DTranspose(32, (3, 3), strides=(
				2, 2), padding='same')(d3)
			#d1 = keras.layers.concatenate([d2, e1[:,-1,:,:,:]], axis=3)
			
			d1 = Conv2DTranspose(16, (3, 3), strides=(
				2, 2), padding='same')(d2)
#			out = keras.layers.concatenate([d1, in_im[:,-1,:,:,:]], axis=3)

			out = Conv2D(self.class_n, (1, 1), activation='softmax',
						 padding='same')(d1)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		elif self.model_type=='BiConvLSTM_DenseNet':

			# convlstm then densenet

			#x = keras.layers.Permute((2,3,1,4))(in_im)
			x = Bidirectional(ConvLSTM2D(32,3,return_sequences=False,
				padding="same"),merge_mode='concat')(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCN((self.patch_len, self.patch_len, self.channel_n), nb_dense_block=2, growth_rate=16, dropout_rate=0.2,
							nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='ConvLSTM_seq2seq':
			x = ConvLSTM2D(256,3,return_sequences=True,padding="same")(in_im)
			x = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			x = Activation('relu')(x)
			self.graph = Model(in_im, x)
			print(self.graph.summary())
		elif self.model_type=='ConvLSTM_seq2seq_bi':
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(in_im)
#			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation='softmax',
#						 padding='same'))(x)

			x = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			out = Activation('relu')(x)						 
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='ConvLSTM_seq2seq_bi_60x2':
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(in_im)
#			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation='softmax',
#						 padding='same'))(x)

			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			x = Activation('relu')(x)						 
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='FCN_ConvLSTM_seq2seq_bi':
			e1 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				strides=(2, 2)))(in_im)
			e2 = TimeDistributed(Conv2D(32, (3, 3), padding='same',
				strides=(2, 2)))(e1)
			e3 = TimeDistributed(Conv2D(48, (3, 3), padding='same',
				strides=(2, 2)))(e2)

			x = Bidirectional(ConvLSTM2D(80,3,return_sequences=True,
				padding="same"),merge_mode='concat')(e3)


			d3 = TimeDistributed(Conv2DTranspose(48, (3, 3), strides=(
				2, 2), padding='same'))(x)
			#d2 = keras.layers.concatenate([d3, e2[:,-1,:,:,:]], axis=3)

			d2 = TimeDistributed(Conv2DTranspose(32, (3, 3), strides=(
				2, 2), padding='same'))(d3)
			#d1 = keras.layers.concatenate([d2, e1[:,-1,:,:,:]], axis=3)
			
			d1 = TimeDistributed(Conv2DTranspose(16, (3, 3), strides=(
				2, 2), padding='same'))(d2)
#			out = keras.layers.concatenate([d1, in_im[:,-1,:,:,:]], axis=3)

			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation='softmax',
						 padding='same'))(d1)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		if self.model_type=='DenseNetTimeDistributed':


			#x = keras.layers.Permute((1,2,0,3))(in_im)
			#x = keras.layers.Permute((2,3,1,4))(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCNTimeDistributed((self.t_len, self.patch_len, self.patch_len, self.channel_n), nb_dense_block=2, growth_rate=16, dropout_rate=0.2,
							nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=in_im,
							recurrent_filters=60)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='DenseNetTimeDistributed_128x2':


			#x = keras.layers.Permute((1,2,0,3))(in_im)
			#x = keras.layers.Permute((2,3,1,4))(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCNTimeDistributed((self.t_len, self.patch_len, self.patch_len, self.channel_n), nb_dense_block=self.mp['dense']['nb_dense_block'], growth_rate=self.mp['dense']['growth_rate'], dropout_rate=0.2,
							nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=in_im,
							recurrent_filters=128)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='DenseNetTimeDistributed_128x2_inconv':


			#x = keras.layers.Permute((1,2,0,3))(in_im)
			#x = keras.layers.Permute((2,3,1,4))(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			x=dilated_layer(in_im,16)
			out = DenseNetFCNTimeDistributed((self.t_len, self.patch_len, self.patch_len, self.channel_n), nb_dense_block=2, growth_rate=64, dropout_rate=0.2,
							nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=x,
							recurrent_filters=128)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='DenseNetTimeDistributed_128x2_3blocks':


			#x = keras.layers.Permute((1,2,0,3))(in_im)
			#x = keras.layers.Permute((2,3,1,4))(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCNTimeDistributed((self.t_len, self.patch_len, self.patch_len, self.channel_n), nb_dense_block=3, growth_rate=32, dropout_rate=0.2,
							nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=in_im,
							recurrent_filters=128)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='pyramid_dilated':

			d1 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				dilation_rate=(2, 2)))(in_im)
			d4 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				dilation_rate=(4, 4)))(in_im)
			d8 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				dilation_rate=(8, 8)))(in_im)

			x = keras.layers.concatenate([d1, d4, d8], axis=4)

			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		if self.model_type=='pyramid_dilated_bconvlstm':

			d1 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				dilation_rate=(2, 2)))(in_im)
			d1 = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(d1)
			d1 = Activation('relu')(d1)
			d4 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				dilation_rate=(4, 4)))(in_im)
			d4 = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(d4)
			d4 = Activation('relu')(d4)
			d8 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				dilation_rate=(8, 8)))(in_im)
			d8 = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(d8)
			d8 = Activation('relu')(d8)
			pdc = keras.layers.concatenate([d1, d4, d8], axis=4)
			r1 = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
							padding="same"))(pdc)
			r2 = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
							padding="same",dilation_rate=(2, 2)))(pdc)
			x = keras.layers.concatenate([r1, r2], axis=4)



			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		elif self.model_type=='bdepthconvlstm':

			e1 = TimeDistributed(Conv2D(16, (3, 3), padding='same'))(in_im)
			e1 = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(e1)
			e1 = Activation('relu')(e1)
		elif self.model_type=='deeplabv3':

			e1 = TimeDistributed(Conv2D(16, (3, 3), padding='same'))(in_im)
			e1 = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(e1)
			e1 = Activation('relu')(e1)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(e1)

			e1 = TimeDistributed(Conv2D(32, (3, 3), padding='same'))(e1)
			e1 = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(e1)
			e1 = Activation('relu')(e1)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(e1)



			dil1 = TimeDistributed(Conv2D(64, (3, 3), padding='same',
				dilation_rate=(2,2)))(e1)
			dil1 = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(dil1)
			dil1 = Activation('relu')(dil1)


			dil2 = TimeDistributed(Conv2D(64, (3, 3), padding='same',
				dilation_rate=(4,4)))(e1)
			dil2 = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(dil2)
			dil2 = Activation('relu')(dil2)

			dil3 = TimeDistributed(Conv2D(64, (3, 3), padding='same',
				dilation_rate=(8,8)))(e1)
			dil3 = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(dil3)
			dil3 = Activation('relu')(dil3)


			pdc = keras.layers.concatenate([dil1, dil2, dil3], axis=4)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(pdc)


			x = TimeDistributed(UpSampling2D(size=(4, 4)))(x)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			self.graph = Model(in_im, out)
		elif self.model_type=='deeplab_rs':



			d1=dilated_layer(in_im,16,1)
			d2=dilated_layer(in_im,16,2)
			d4=dilated_layer(in_im,16,4)
			d8=dilated_layer(in_im,16,8)

			pdc = keras.layers.concatenate([d1, d2, d4, d8], axis=4)
			pdc = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(pdc)

			d1=dilated_layer(pdc,32,1)
			d2=dilated_layer(pdc,32,2)
			d4=dilated_layer(pdc,32,4)

			pdc = keras.layers.concatenate([d1, d2, d4], axis=4)
			pdc = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(pdc)

			d1=dilated_layer(pdc,64,1)
			d2=dilated_layer(pdc,64,2)
			pdc = keras.layers.concatenate([d1, d2], axis=4)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(pdc)
			x = TimeDistributed(UpSampling2D(size=(4, 4)))(x)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='deeplab_rs_multiscale':

			d1=dilated_layer(in_im,16,1)
			d2=dilated_layer(in_im,16,2)
			d4=dilated_layer(in_im,16,4)
			d8=dilated_layer(in_im,16,8)

			pdc1 = keras.layers.concatenate([d1, d2, d4, d8], axis=4)
			pdc = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(pdc1)

			d1=dilated_layer(pdc,32,1)
			d2=dilated_layer(pdc,32,2)
			d4=dilated_layer(pdc,32,4)

			pdc2 = keras.layers.concatenate([d1, d2, d4], axis=4)
			pdc = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(pdc2)

			d1=dilated_layer(pdc,64,1)
			d2=dilated_layer(pdc,64,2)
			pdc = keras.layers.concatenate([d1, d2], axis=4)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(pdc)
	
			# Decoder V3+
			x = TimeDistributed(UpSampling2D(size=(2, 2)))(x)
			x2 = dilated_layer(pdc2,128,1,kernel_size=1)#Low level features
			x= keras.layers.concatenate([x, x2], axis=4)
			x= dilated_layer(x,64,1,kernel_size=3)
			x = TimeDistributed(UpSampling2D(size=(2, 2)))(x)
			
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
		elif self.model_type=='deeplabv3plus':

			e1 = dilated_layer(in_im,16,1)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(e1)
			p1 = dilated_layer(e1,32,1)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)



			dilAvg = im_pooling_layer(e1,64)
			dil0 = dilated_layer(e1,64,1,kernel_size=1)
			dil1 = dilated_layer(e1,64,2)
			dil2 = dilated_layer(e1,64,4)
			dil3 = dilated_layer(e1,64,8)

			pdc = keras.layers.concatenate([dilAvg, dil0, dil1, dil2, dil3], axis=4)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(pdc)

			# Decoder V3+
			x = TimeDistributed(UpSampling2D(size=(2, 2)))(x)
			x2 = dilated_layer(p1,128,1,kernel_size=1)#Low level features
			x= keras.layers.concatenate([x, x2], axis=4)
			x= dilated_layer(x,64,1,kernel_size=3)
			x = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			out = TimeDistributed(UpSampling2D(size=(2, 2)))(x)
			

			self.graph = Model(in_im, out)
		elif self.model_type=='FCN_ConvLSTM_seq2seq_bi_skip':

			#fs=32
			fs=16
			
			p1=dilated_layer(in_im,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d2 = transpose_layer(d3,fs*4)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs*2)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			
		if self.model_type=='BUnetConvLSTM':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d2 = transpose_layer(d3,fs*4)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs*2)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='Unet4ConvLSTM':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = ConvLSTM2D(256,3,return_sequences=True,
					padding="same")(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d2 = transpose_layer(d3,fs*4)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs*2)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='BUnet3ConvLSTM':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='BUnet4ConvLSTM':



			concat_axis = 3

			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=False,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=-1)
			d3=dilated_layer_Nto1(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=-1)
			d2=dilated_layer_Nto1(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=-1)
			out=dilated_layer_Nto1(d1,fs)
			out = Conv2D(self.class_n, (1, 1), activation=None,
										padding='same')(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		def slice_tensor(x,output_shape):
			deb.prints(output_shape)
			deb.prints(K.int_shape(x))
#				res1 = Lambda(lambda x: x[:,:,:,-1], output_shape=output_shape)(x)
#				res2 = Lambda(lambda x: x[:,:,:,-1], output_shape=output_shape[1:])(x)
			res2 = Lambda(lambda x: x[:,-1])(x)

#				deb.prints(K.int_shape(res1))
			deb.prints(K.int_shape(res2))
			
			return res2
		if self.model_type=='UUnet4ConvLSTM':

			
			
			concat_axis = 3

			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = ConvLSTM2D(256,3,return_sequences=False,
					padding="same")(e3)

			d3 = transpose_layer(x,fs*4)
			p3 = slice_tensor(p3, output_shape = K.int_shape(d3))
			deb.prints(K.int_shape(p3))
			deb.prints(K.int_shape(d3))
			
			d3 = keras.layers.concatenate([d3, p3], axis=-1)
			d3=dilated_layer_Nto1(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			p2 = slice_tensor(p2, output_shape = K.int_shape(d2))
			deb.prints(K.int_shape(p2))
			deb.prints(K.int_shape(d2))

			d2 = keras.layers.concatenate([d2, p2], axis=-1)
			d2=dilated_layer_Nto1(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			p1 = slice_tensor(p1, output_shape = K.int_shape(d1))
			deb.prints(K.int_shape(p1))
			deb.prints(K.int_shape(d1))

			d1 = keras.layers.concatenate([d1, p1], axis=-1)
			out=dilated_layer_Nto1(d1,fs)
			out = Conv2D(self.class_n, (1, 1), activation=None,
										padding='same')(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='UUnet4ConvLSTM_doty':
			#self.t_len = 20
			metadata_in = Input(shape=(self.model_t_len,2))
			
			concat_axis = 3

			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)
			metadata=Lambda(lambda x: K.expand_dims(x, 2))(metadata_in)
			metadata=Lambda(lambda x: K.expand_dims(x, 2))(metadata)
			#metadata=Lambda(lambda x: K.expand_dims(x, 2))(metadata)

			deb.prints(K.int_shape(metadata))

			metadata = TimeDistributed(Lambda(lambda y: K.tf.image.resize_bilinear(y,size=(4,4))))(metadata)
#				x = TimeDistributed(UpSampling2D(
#					size=(self.patch_len,self.patch_len)))(x)
			deb.prints(K.int_shape(metadata))

			x = keras.layers.concatenate([e3, metadata], axis = -1)
			x = ConvLSTM2D(256,3,return_sequences=False,
					padding="same")(x)

			d3 = transpose_layer(x,fs*4)
			p3 = slice_tensor(p3, output_shape = K.int_shape(d3))
			deb.prints(K.int_shape(p3))
			deb.prints(K.int_shape(d3))
			
			d3 = keras.layers.concatenate([d3, p3], axis=-1)
			d3=dilated_layer_Nto1(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			p2 = slice_tensor(p2, output_shape = K.int_shape(d2))
			deb.prints(K.int_shape(p2))
			deb.prints(K.int_shape(d2))

			d2 = keras.layers.concatenate([d2, p2], axis=-1)
			d2=dilated_layer_Nto1(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			p1 = slice_tensor(p1, output_shape = K.int_shape(d1))
			deb.prints(K.int_shape(p1))
			deb.prints(K.int_shape(d1))

			d1 = keras.layers.concatenate([d1, p1], axis=-1)
			out=dilated_layer_Nto1(d1,fs)
			out = Conv2D(self.class_n, (1, 1), activation=None,
										padding='same')(out)
			self.graph = Model(inputs=[in_im, metadata_in], outputs=out)
			print(self.graph.summary())
			#keras.utils.plot_model(model, show_shapes=True, to_file="model.png")	


		if self.model_type=='BUnet4ConvLSTM_SkipLSTM':
			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)
			p1=dilated_layer(p1,fs)
			x_p1 = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
					padding="same"),merge_mode='concat')(p1)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			x_p2 = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
					padding="same"),merge_mode='concat')(p2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			x_p3 = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
					padding="same"),merge_mode='concat')(p3)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, x_p3], axis=4)
			d3 = dilated_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, x_p2], axis=4)
			d2 = dilated_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, x_p1], axis=4)
			out = dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='BUnet6ConvLSTM':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3=dilated_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d2=dilated_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())			
		if self.model_type=='BUnet5ConvLSTM':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			#p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3=dilated_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d2=dilated_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='BUnet2ConvLSTM':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e2)

			d2 = transpose_layer(x,fs*4)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs*2)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='UnetTimeDistributed':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			#x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
			#        padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(e3,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d2 = transpose_layer(d3,fs*4)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs*2)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		elif self.model_type=='BAtrousConvLSTM':

			#fs=32
			fs=16
			
			#x=dilated_layer(in_im,fs)
			x=dilated_layer(in_im,fs)
			x=dilated_layer(x,fs)
			x=spatial_pyramid_pooling(x,fs*4,max_rate=8,
				global_average_pooling=False)
			
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(x)

			#out=dilated_layer(x,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='BAtrousGAPConvLSTM':

			#fs=32
			fs=16
			
			#x=dilated_layer(in_im,fs)
			x=dilated_layer(in_im,fs)
			x=dilated_layer(x,fs)
			x=spatial_pyramid_pooling(x,fs*4,max_rate=8,
				global_average_pooling=True)
			
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(x)

			out=dilated_layer(x,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='BUnetAtrousConvLSTM':

			#fs=32
			fs=16
			x=dilated_layer(in_im,fs)
			p1=spatial_pyramid_pooling(x,fs,max_rate=8)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=spatial_pyramid_pooling(e1,fs*2,max_rate=4)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=spatial_pyramid_pooling(e2,fs*4,max_rate=2)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d2 = transpose_layer(d3,fs*4)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs*2)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='BUnetAtrousConvLSTM_v3p':

			fs=16
			x=dilated_layer(in_im,fs)
			d1=dilated_layer(in_im,16,1)
			d2=dilated_layer(in_im,16,2)
			d4=dilated_layer(in_im,16,4)
			d8=dilated_layer(in_im,16,8)

			pdc = keras.layers.concatenate([d1, d2, d4, d8], axis=4)
			pdc = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(pdc)

			d1=dilated_layer(pdc,32,1)
			d2=dilated_layer(pdc,32,2)
			d4=dilated_layer(pdc,32,4)

			pdc = keras.layers.concatenate([d1, d2, d4], axis=4)
			pdc = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(pdc)

			d1=dilated_layer(pdc,64,1)
			d2=dilated_layer(pdc,64,2)
			pdc = keras.layers.concatenate([d1, d2], axis=4)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(pdc)
			x = TimeDistributed(UpSampling2D(size=(4, 4)))(x)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='Attention_DenseNetTimeDistributed_128x2':


			#x = keras.layers.Permute((1,2,0,3))(in_im)
			#x = keras.layers.Permute((2,3,1,4))(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCNTimeDistributedAttention((self.t_len, self.patch_len, self.patch_len, self.channel_n), nb_dense_block=self.mp['dense']['nb_dense_block'], growth_rate=self.mp['dense']['growth_rate'], dropout_rate=0.2,
							nb_layers_per_block=self.mp['dense']['nb_layers_per_block'], upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=in_im,
							recurrent_filters=128)
			self.graph = Model(in_im, out)


		if self.model_type=='fcn_lstm_temouri':


			#fs=32
			fs=64

			#e1=dilated_layer(in_im,fs)			
			e1=dilated_layer(in_im,fs) # 64
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(e1)
			e2=dilated_layer(e2,fs*2) # 128
			deb.prints(K.int_shape(e2))
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(e2)
			e3=dilated_layer(e3,fs*4) # 256
			deb.prints(K.int_shape(e3))

			e4 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(e3)


			e4=dilated_layer(e4,fs*8) # 512

			d3 = transpose_layer(e4,fs*4) #  256
			deb.prints(K.int_shape(d3))

			d3 = keras.layers.concatenate([d3, e3], axis=4) # 512
			d3=dilated_layer(d3,fs*4) # 256
			d2 = transpose_layer(d3,fs*2) # 128
			deb.prints(K.int_shape(d2))
			d2 = keras.layers.concatenate([d2, e2], axis=4) # 256
			d2=dilated_layer(d2,fs*2) # 128
			d1 = transpose_layer(d2,fs) # 64
			d1 = keras.layers.concatenate([d1, e1], axis=4) # 128
			d1=dilated_layer(d1,fs) # 64 # this would concatenate with the date
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(d1)			
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='fcn_lstm_temouri2':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x=dilated_layer(e3,fs*8)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3=dilated_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d2=dilated_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(out)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='Unet3D':
			fs=32
			#fs=16

			p1=dilated_layer_3D(in_im,fs,kernel_size=(7,3,3))
			e1 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p1)
			p2=dilated_layer_3D(e1,fs*2,kernel_size=(7,3,3))
			e2 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p2)
			p3=dilated_layer_3D(e2,fs*4,kernel_size=(7,3,3))
			e3 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p3)

			d3 = transpose_layer_3D(e3,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3 = dilated_layer_3D(d3,fs*4,kernel_size=(7,3,3))
			d2 = transpose_layer_3D(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d2 = dilated_layer_3D(d2,fs*2,kernel_size=(7,3,3))
			d1 = transpose_layer_3D(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out = dilated_layer_3D(d1,fs,kernel_size=(7,3,3))
			out = Conv3D(self.class_n, (1, 1, 1), activation=None,
										padding='same')(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		if self.model_type=='Unet3D_ATPP':
			fs=16
			max_rate=5
			#fs=16

			p1=temporal_pyramid_pooling(in_im,fs,max_rate)
			e1 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p1)
			p2=temporal_pyramid_pooling(e1,fs*2,max_rate)
			e2 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p2)
			p3=temporal_pyramid_pooling(e2,fs*4,max_rate)
			e3 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p3)

			d3 = transpose_layer_3D(e3,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3 = temporal_pyramid_pooling(d3,fs*4,max_rate)
			d2 = transpose_layer_3D(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d3 = temporal_pyramid_pooling(d2,fs*2,max_rate)
			d1 = transpose_layer_3D(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out = temporal_pyramid_pooling(d1,fs,max_rate)
			out = Conv3D(self.class_n, (1, 1, 1), activation=None,
										padding='same')(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		if self.model_type=='Unet3D_AFPP':
			fs=16
			max_rate=4
			#fs=16

			p1=full_pyramid_pooling(in_im,fs,max_rate)
			e1 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p1)
			p2=full_pyramid_pooling(e1,fs*2,max_rate)
			e2 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p2)
			p3=full_pyramid_pooling(e2,fs*4,max_rate)
			e3 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p3)

			d3 = transpose_layer_3D(e3,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3 = full_pyramid_pooling(d3,fs*4,max_rate)
			d2 = transpose_layer_3D(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d3 = full_pyramid_pooling(d2,fs*2,max_rate)
			d1 = transpose_layer_3D(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out = full_pyramid_pooling(d1,fs,max_rate)
			out = Conv3D(self.class_n, (1, 1, 1), activation=None,
										padding='same')(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())


# ==================================== ATTENTION ATTEMPTS =================================== #
		elif self.model_type=='ConvLSTM_seq2seq_bi_attention':
#			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation='softmax',
#						 padding='same'))(x)

			
			# attention

			# shape is (t,h,w,c). We want shape (c,h,w,t)
			# then timedistributed conv. 1x1 applies attention to t
			# then return 
			x = keras.layers.Permute((4,2,3,1))(in_im)
			x = TimeDistributed(Conv2D(self.t_len, (1, 1), activation=None,
						 padding='same'))(x)
			x = Activation('relu')(x)
			x = keras.layers.Permute((4,2,3,1))(x)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(x)

			
			x = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			out = Activation('relu')(x)

#			x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)

			self.graph = Model(in_im, out)
			print(self.graph.summary())

		elif self.model_type=='ConvLSTM_seq2seq_bi_attention2':
#			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation='softmax',
#						 padding='same'))(x)

			
			# attention

			# shape is (t,h,w,c). We want shape (c,h,w,t)
			# then timedistributed conv. 1x1 applies attention to t
			# then return 
			def attention_weights(x):
				att = keras.layers.Permute((4,2,3,1))(x)
				att = TimeDistributed(Conv2D(self.t_len, (1, 1), activation=None,
							 padding='same'))(att)
				att = Activation('relu')(att)
				att = keras.layers.Permute((4,2,3,1))(att)
				return att
			att = attention_weights(in_im)
			x = multiply([x,att])
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(x)

			
			x = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			out = Activation('relu')(x)

#			x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)

			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='ConvLSTM_seq2seq_bi_SelfAttention':
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(in_im)
#			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation='softmax',
#						 padding='same'))(x)
			x = Reshape((self.t_len,
					32*32*256))(x)
			x = SeqSelfAttention(
				kernel_regularizer=l2(1e-4),
				bias_regularizer=l1(1e-4),
				attention_regularizer_weight=1e-4,
				name='Attention')(x)
			x = Reshape((self.t_len,
					32,32,256))(x)
			x = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			out = Activation('relu')(x)						 
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='BUnet4ConvLSTM_SelfAttention':

			print(self.model_type)
			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)
			x = Reshape((self.t_len,
					4*4*256))(x)
			x = SeqSelfAttention(
				kernel_regularizer=l2(1e-4),
				bias_regularizer=l1(1e-4),
				attention_regularizer_weight=1e-4,
				name='Attention')(x)
			x = Reshape((self.t_len,
					4,4,256))(x)
			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3=dilated_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d2=dilated_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		if self.model_type=='Unet4ConvLSTM_SelfAttention':

			print(self.model_type)
			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			#x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
			#		padding="same"),merge_mode='concat')(e3)
			x = Reshape((self.t_len,
					4*4*fs*4))(e3)
			x = SeqSelfAttention(
				units=256,
				kernel_regularizer=l2(1e-4),
				bias_regularizer=l1(1e-4),
				attention_regularizer_weight=1e-4,
				name='Attention')(x)
			x = Reshape((self.t_len,
					4,4,fs*4))(x)
			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3=dilated_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d2=dilated_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		if self.model_type=='BUnet4_Standalone':
			print(self.model_type)
			in_shape = (self.t_len,self.patch_len, self.patch_len, self.channel_n)
			#in_im = Input(shape=in_shape)

			#x = Permute((1,2,3,0), input_shape = in_shape)(in_im)
			x = Permute((2,3,4,1), input_shape = in_shape)(in_im)

			#my_permute = lambda y: K.permute_dimensions(y, (None,1,2,3,0))
			#x = Lambda(my_permute)(in_im)
			x = Reshape((self.patch_len,self.patch_len,self.channel_n*self.t_len))(x)
			#fs=32
			def conv_layer(x,filter_size,dilation_rate=1, kernel_size=3):
				x = Conv2D(filter_size, kernel_size, padding='same',
					dilation_rate=(dilation_rate, dilation_rate))(x)
				x = BatchNormalization(gamma_regularizer=l2(weight_decay),
								beta_regularizer=l2(weight_decay))(x)
				x = Activation('relu')(x)
				return x		
			def transpose_layer(x,filter_size,dilation_rate=1, 
				kernel_size=3, strides=(2,2)):
				x = Conv2DTranspose(filter_size, 
					kernel_size, strides=strides, padding='same')(x)
				x = BatchNormalization(gamma_regularizer=l2(weight_decay),
													beta_regularizer=l2(weight_decay))(x)
				x = Activation('relu')(x)
				return x		
			fs=16

			p1=conv_layer(x,fs)			
			p1=conv_layer(p1,fs)
			e1 = AveragePooling2D((2, 2), strides=(2, 2))(p1)
			p2=conv_layer(e1,fs*2)
			e2 = AveragePooling2D((2, 2), strides=(2, 2))(p2)
			p3=conv_layer(e2,fs*4)
			e3 = AveragePooling2D((2, 2), strides=(2, 2))(p3)

			# This replaces convlstm. Check param count and increase filters if lacking
			x = conv_layer(e3, fs*16)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=3)
			d3=conv_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=3)
			d2=conv_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=3)
			out=conv_layer(d1,fs)
			out = Conv2D(self.class_n*self.t_len, (1, 1), activation=None,
										padding='same')(out)
			#deb.prints(out.output_shape)
			out_shape = (self.patch_len,self.patch_len,self.class_n,self.t_len)
			out = Reshape(out_shape)(out)

			#my_permute = lambda x: K.permute_dimensions(out, (None,3,0,1,2))
			#out = Lambda(my_permute)(x)

			#out = Permute((3,0,1,2), input_shape=out_shape)(out)
			out = Permute((4,1,2,3), input_shape=out_shape)(out)

			self.graph = Model(in_im, out)
			print(self.graph.summary())

		if self.model_type=='BUnet4ConvLSTM_64':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3=dilated_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d2=dilated_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		#self.graph = Model(in_im, out)
		print(self.graph.summary(line_length=125))



		with open('model_summary.txt','w') as fh:
			self.graph.summary(line_length=125,print_fn=lambda x: fh.write(x+'\n'))
		#self.graph.summary(print_fn=model_summary_print)
	def loss_weights_estimate(self,data):
		unique,count=np.unique(data.patches['train']['label'].argmax(axis=-1),return_counts=True)
		unique=unique[1:] # No bcknd
		count=count[1:].astype(np.float32)
		weights_from_unique=np.max(count)/count
		deb.prints(weights_from_unique)
		deb.prints(np.max(count))
		deb.prints(count)
		deb.prints(unique)
		self.loss_weights=np.zeros(self.class_n)
		for clss in range(1,self.class_n): # class 0 is bcknd. Leave it in 0
			
			if clss in unique:
				self.loss_weights[clss]=weights_from_unique[unique==clss]
			else:
				self.loss_weights[clss]=0
		deb.prints(self.loss_weights)
		self.loss_weights_ones=self.loss_weights.copy() # all weights are 1
		self.loss_weights_ones[1:]=1
		
		# no background weight
		self.loss_weights=self.loss_weights[1:]
		self.loss_weights_ones=self.loss_weights_ones[1:]

		deb.prints(self.loss_weights.shape)
		deb.prints(self.loss_weights)
		
	def test(self,data):
		data.patches['train']['batch_n'] = data.patches['train']['in'].shape[0]//self.batch['train']['size']
		data.patches['test']['batch_n'] = data.patches['test']['in'].shape[0]//self.batch['test']['size']

		batch = {'train': {}, 'test': {}}
		self.batch['train']['n'] = data.patches['train']['in'].shape[0] // self.batch['train']['size']
		self.batch['test']['n'] = data.patches['test']['in'].shape[0] // self.batch['test']['size']

		data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'])
		deb.prints(data.patches['test']['label'].shape)
		deb.prints(self.batch['test']['n'])

		self.metrics['test']['loss'] = np.zeros((1, 2))

		data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'])
		self.batch_test_stats=True

		for batch_id in range(0, self.batch['test']['n']):
			idx0 = batch_id*self.batch['test']['size']
			idx1 = (batch_id+1)*self.batch['test']['size']

			batch['test']['in'] = data.patches['test']['in'][idx0:idx1]
			batch['test']['label'] = data.patches['test']['label'][idx0:idx1]

			if self.batch_test_stats:
				self.metrics['test']['loss'] += self.graph.test_on_batch(
					batch['test']['in'], batch['test']['label'])		# Accumulated epoch

			data.patches['test']['prediction'][idx0:idx1]=self.graph.predict(batch['test']['in'],batch_size=self.batch['test']['size'])

		#====================METRICS GET================================================#
		deb.prints(data.patches['test']['label'].shape)		
		deb.prints(idx1)
		print("Epoch={}".format(epoch))	
		
		# Average epoch loss
		self.metrics['test']['loss'] /= self.batch['test']['n']
			
		# Get test metrics
		metrics=data.metrics_get(data.patches['test']['prediction'],data.patches['test']['label'],debug=1)
		print('oa={}, aa={}, f1={}, f1_wght={}'.format(metrics['overall_acc'],
			metrics['average_acc'],metrics['f1_score'],metrics['f1_score_weighted']))

	def train(self, data):

		# Random shuffle
		##data.patches['train']['in'], data.patches['train']['label'] = shuffle(data.patches['train']['in'], data.patches['train']['label'], random_state=0)

		# Normalize
		##data.patches['train']['in'] = normalize(data.patches['train']['in'].astype('float32'))
		##data.patches['test']['in'] = normalize(data.patches['test']['in'].astype('float32'))

		# Computing the number of batches
		data.patches['train']['batch_n'] = data.patches['train']['in'].shape[0]//self.batch['train']['size']
		data.patches['test']['batch_n'] = data.patches['test']['in'].shape[0]//self.batch['test']['size']
		data.patches['val']['batch_n'] = data.patches['val']['in'].shape[0]//self.batch['val']['size']

		deb.prints(data.patches['train']['batch_n'])

		self.train_loop(data)

	def early_stop_check(self,metrics,epoch,most_important='overall_acc'):

		if metrics[most_important]>=self.early_stop['best'] and self.early_stop["signal"]==False:
			self.early_stop['best']=metrics[most_important]
			self.early_stop['count']=0
			print("Best metric updated")
			self.early_stop['best_updated']=True
			#data.im_reconstruct(subset='test',mode='prediction')
		else:
			self.early_stop['best_updated']=False
			self.early_stop['count']+=1
			deb.prints(self.early_stop['count'])
			if self.early_stop["count"]>=self.early_stop["patience"]:
				self.early_stop["signal"]=True

			#else:
				#self.early_stop["signal"]=False
			


	def train_loop(self, data):
		print('Start the training')
		cback_tboard = keras.callbacks.TensorBoard(
			log_dir='../summaries/', histogram_freq=0, batch_size=self.batch['train']['size'], write_graph=True, write_grads=False, write_images=False)
		txt={'count':0,'val':{},'test':{}}
		txt['val']={'metrics':[],'epoch':[],'loss':[]}
		txt['test']={'metrics':[],'epoch':[],'loss':[]}
		
		
		#========= VAL INIT


		if self.val_set:
			count,unique=np.unique(data.patches['val']['label'].argmax(axis=-1),return_counts=True)
			print("Val label count,unique",count,unique)

		count,unique=np.unique(data.patches['train']['label'].argmax(axis=-1),return_counts=True)
		print("Train count,unique",count,unique)
		
		count,unique=np.unique(data.patches['test']['label'].argmax(axis=-1),return_counts=True)
		print("Test count,unique",count,unique)
		
		#==================== ESTIMATE BATCH NUMBER===============================#
		prediction_dtype=np.float16
#		prediction_dtype=np.int16
#		prediction_dtype=np.int8

		batch = {'train': {}, 'test': {}, 'val':{}}
		self.batch['train']['n'] = data.patches['train']['in'].shape[0] // self.batch['train']['size']
		self.batch['test']['n'] = data.patches['test']['in'].shape[0] // self.batch['test']['size']
		self.batch['val']['n'] = data.patches['val']['in'].shape[0] // self.batch['val']['size']

		batch['train']['size'] = self.batch['train']['size']
		batch['test']['size'] = self.batch['test']['size']
		batch['val']['size'] = self.batch['val']['size']
	
		data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'][...,:-1], dtype=prediction_dtype)
		deb.prints(data.patches['test']['label'].shape)
		deb.prints(self.batch['test']['n'])
		
		self.early_stop["signal"]=False
		#if self.train_mode==

		#data.im_reconstruct(subset='test',mode='label')
		#for epoch in [0,1]:
		init_time=time.time()

		#paramsTrain.model_t_len = 12
		batch, data, min_seq_len = self.mim.trainingInit(batch, data, self.t_len, 
									model_t_len=paramsTrain.seq_len)
		data = self.mim.valLabelSelect(data)

#		data.doty_flag=True
#		data.ds.doty_flag=True
		data.doty_flag=False
		data.ds.doty_flag=False
		deb.prints(self.mim)
		#==============================START TRAIN/TEST LOOP============================#
		for epoch in range(self.epochs):

			idxs=np.random.permutation(data.patches['train']['label'].shape[0])
			data.patches['train']['in']=data.patches['train']['in'][idxs]
			data.patches['train']['label']=data.patches['train']['label'][idxs]
			
			self.metrics['train']['loss'] = np.zeros((1, 2))
			self.metrics['test']['loss'] = np.zeros((1, 2))
			self.metrics['val']['loss'] = np.zeros((1, 2))

			# Random shuffle the data
			##data.patches['train']['in'], data.patches['train']['label'] = shuffle(data.patches['train']['in'], data.patches['train']['label'])
			if paramsTrain.seq_mode=='var' or paramsTrain.seq_mode=='var_label':
				label_date_id = -data.labeled_dates
			else:
				label_date_id = -1 # fixed
			#=============================TRAIN LOOP=========================================#
			for batch_id in range(0, self.batch['train']['n']):
				
				idx0 = batch_id*self.batch['train']['size']
				idx1 = (batch_id+1)*self.batch['train']['size']

				batch['train']['in'] = data.patches['train']['in'][idx0:idx1]
				batch['train']['label'] = data.patches['train']['label'][idx0:idx1]

				#pdb.set_trace()
				if self.time_measure==True:
					start_time=time.time()
				
				##deb.prints(batch['train']['in'].shape)
				# set label N to 1
				#if paramsTrain.seq_mode=='var' or paramsTrain.seq_mode=='var_label':
#				batch_seq_len = 12
				#deb.prints(self.mim)
				input_ = self.mim.batchTrainPreprocess(batch['train'], data.ds, 
								label_date_id, paramsTrain.seq_len)
				##deb.prints(input_[0].shape)
				##deb.prints(input_[1].shape)
				##deb.prints(batch['train']['in'].shape)
				
				gt = np.expand_dims(batch['train']['label'].argmax(axis=-1),axis=-1).astype(np.int8)
				if paramsTrain.seq_mode=='var' or paramsTrain.seq_mode=='var_label':
					gt = gt[:, label_date_id] # N to 1 label is selected
				#print("Debugging len(input_), input_, input_[0].shape, input_[1].shape",
				#		len(input_), input_, input_[0].shape, input_[1].shape)
				self.metrics['train']['loss'] += self.graph.train_on_batch(
					input_, 
					gt)		# Accumulated epoch
				if self.time_measure==True:
					batch_time=time.time()-start_time
					print(batch_time)
					sys.exit('Batch time:')
				if paramsTrain.seq_mode=='var' or paramsTrain.seq_mode=='var_label':
					if label_date_id < -1: # if -12 to -2, increase 1
						label_date_id = label_date_id + 1
					else: # if -1,
						label_date_id = -data.labeled_dates
			# Average epoch loss
			self.metrics['train']['loss'] /= self.batch['train']['n']

			self.train_predict=True
			#pdb.set_trace()
			#if self.train_predict:

            
			#================== VAL LOOP=====================#
			if self.val_set:
				deb.prints(data.patches['val']['label'].shape)
#				if paramsTrain.seq_mode == 'fixed':
#					data.patches['val']['prediction']=np.zeros_like(data.patches['val']['label'][...,:-1],dtype=prediction_dtype)
				if paramsTrain.seq_mode == 'fixed_label_len':
					data.patches['val']['prediction']=np.zeros_like(data.patches['val']['label'],dtype=prediction_dtype)
				elif paramsTrain.seq_mode == 'var' or paramsTrain.seq_mode =='var_label' or paramsTrain.seq_mode == 'fixed':
					data.patches['val']['prediction']=np.zeros_like(data.patches['val']['label'][...,:-1],dtype=prediction_dtype)

				self.batch_test_stats=False

				for batch_id in range(0, self.batch['val']['n']):
					idx0 = batch_id*self.batch['val']['size']
					idx1 = (batch_id+1)*self.batch['val']['size']

					batch['val']['in'] = data.patches['val']['in'][idx0:idx1]
					batch['val']['label'] = data.patches['val']['label'][idx0:idx1]

#					input_ = self.mim.batchMetricSplitPreprocess(batch['val'], data)

					if self.batch_test_stats:
						
						self.metrics['val']['loss'] += self.graph.test_on_batch(
							input_,
							np.expand_dims(batch['val']['label'].argmax(axis=-1),axis=-1).astype(np.int8))		# Accumulated epoch
					if paramsTrain.seq_mode == 'fixed' or paramsTrain.seq_mode == 'fixed_label_len':
						input_ = self.mim.batchTrainPreprocess(batch['val'], data.ds,  
									label_date_id = -1) # tstep is -12 to -1

						data.patches['val']['prediction'][idx0:idx1]=(self.graph.predict(
							input_,
							batch_size=self.batch['val']['size'])).astype(prediction_dtype) #*13
					
					elif paramsTrain.seq_mode == 'var' or paramsTrain.seq_mode =='var_label':
						for t_step in range(data.labeled_dates): # 0 to 11
							batch_val_label = batch['val']['label'][:, t_step]
							#data.patches['test']['label'] = data.patches['test']['label'][:, label_id]
							##deb.prints(batch_val_label.shape)
							##deb.prints(t_step-data.labeled_dates)
							input_ = self.mim.batchTrainPreprocess(batch['val'], data.ds,  
										label_date_id = t_step-data.labeled_dates) # tstep is -12 to -1

							#deb.prints(data.patches['test']['label'].shape)

							data.patches['val']['prediction'][idx0:idx1, t_step]=(self.graph.predict(
								input_,
								batch_size=self.batch['val']['size'])).astype(prediction_dtype) #*13
						
				metrics_val=data.metrics_get(data.patches['val']['prediction'],data.patches['val']['label'],debug=2)





				self.metrics['val']['loss'] /= self.batch['val']['n']

				self.early_stop_check(metrics_val,epoch,most_important='f1_score')

				metrics_val['per_class_acc'].setflags(write=1)
				metrics_val['per_class_acc'][np.isnan(metrics_val['per_class_acc'])]=-1
				print(metrics_val['f1_score_noavg'])
				
			#print("Val predictions unique",np.unique(data.patches['val']['prediction'],return_counts=True))
			#pdb.set_trace()
			
			#==========================TEST LOOP================================================#
			if self.early_stop['signal']==True:
				print(" ============= EARLY STOP ACHIEVED ===============")
				print("============= LOADING EARLY STOP BEST WEIGHTS ===============")
				self.graph.load_weights('weights_best.h5')
			test_loop_each_epoch=False
			print('Stop epoch is {}. Current epoch is {}/{}'.format(self.stop_epoch,epoch,self.stop_epoch))
			deb.prints(self.stop_epoch==epoch)
			
			if test_loop_each_epoch==True or self.early_stop['signal']==True or (self.stop_epoch>=0 and self.stop_epoch==epoch):
				
				print("======== BEGINNING TEST PREDICT... ============")
				data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'][...,:-1],dtype=prediction_dtype)#.astype(prediction_dtype)
				deb.prints(data.patches['test']['prediction'].shape)
				self.batch_test_stats=False

				for batch_id in range(0, self.batch['test']['n']):
					idx0 = batch_id*self.batch['test']['size']
					idx1 = (batch_id+1)*self.batch['test']['size']

					batch['test']['in'] = data.patches['test']['in'][idx0:idx1]
					batch['test']['label'] = data.patches['test']['label'][idx0:idx1]

					if self.batch_test_stats:
						input_ = batch['test']['in'][:,-paramsTrain.seq_len:].astype(np.float16)
						if paramsTrain.seq_mode == 'var_label':
							input_ = self.addDoty(input_, bounds=[-paramsTrain.seq_len, None])
						else:
							input_ = self.addDoty(input_)
						self.metrics['test']['loss'] += self.graph.test_on_batch(
							input_,
							np.expand_dims(batch['test']['label'].argmax(axis=-1),axis=-1).astype(np.int16))		# Accumulated epoch


					if paramsTrain.seq_mode == 'fixed' or paramsTrain.seq_mode=='fixed_label_len':
						input_ = self.mim.batchTrainPreprocess(batch['test'], data.ds,  
									label_date_id = -1)
						data.patches['test']['prediction'][idx0:idx1]=(self.graph.predict(
							input_,
							batch_size=self.batch['test']['size'])).astype(prediction_dtype) #*13
					
					elif paramsTrain.seq_mode == 'var' or paramsTrain.seq_mode =='var_label':
						for t_step in range(data.labeled_dates):
							batch_val_label = batch['test']['label'][:, t_step]
							#data.patches['test']['label'] = data.patches['test']['label'][:, label_id]
							deb.prints(batch_val_label.shape)
							
							input_ = self.mim.batchTrainPreprocess(batch['test'], data.ds,  
										label_date_id = t_step-data.labeled_dates)

							#deb.prints(data.patches['test']['label'].shape)

							data.patches['test']['prediction'][idx0:idx1, t_step]=(self.graph.predict(
								input_,
								batch_size=self.batch['test']['size'])).astype(prediction_dtype) #*13
						


				if self.batch_test_stats==True:
					# Average epoch loss
					self.metrics['test']['loss'] /= self.batch['test']['n']
				# Get test metrics
				metrics=data.metrics_get(data.patches['test']['prediction'],data.patches['test']['label'],debug=1)
			#====================METRICS GET================================================#
			deb.prints(data.patches['test']['label'].shape)	
			deb.prints(data.patches['test']['prediction'].dtype)
			deb.prints(data.patches['val']['prediction'].dtype)

			#pdb.set_trace()

			deb.prints(idx1)
			print("Epoch={}".format(epoch))	
			
			if test_loop_each_epoch==True:
				print("Test metrics are printed each epoch. Metrics:",epoch,metrics)
			
			if self.early_stop['best_updated']==True:
				if test_loop_each_epoch==True:
					self.early_stop['best_predictions']=data.patches['test']['prediction']
				self.graph.save_weights('weights_best.h5')
				if self.model_save==True:
					self.graph.save('model_best.h5')

			print(self.early_stop['signal'])
			if self.early_stop["signal"]==True or (self.stop_epoch>=0 and self.stop_epoch==epoch):
				self.final_accuracy_report(data,epoch,metrics,init_time)
				break
				
			# Check early stop and store results if they are the best
			#if epoch % 5 == 0:
			#	print("Writing to file...")
				#for i in range(len(txt['test']['metrics'])):

				#	data.metrics_write_to_txt(txt['test']['metrics'][i],np.squeeze(txt['test']['loss'][i]),
				#		txt['test']['epoch'][i],path=self.report['best']['text_history_path'])
			#	txt['test']['metrics']=[]
			#	txt['test']['loss']=[]
			#	txt['test']['epoch']=[]
				#self.graph.save('my_model.h5')


			#data.metrics_write_to_txt(metrics,np.squeeze(self.metrics['test']['loss']),
			#	epoch,path=self.report['best']['text_history_path'])
			#self.test_metrics_evaluate(data.patches['test'],metrics,epoch)
			#if self.early_stop['signal']==True:
			#	break


			##deb.prints(metrics['confusion_matrix'])
			#metrics['average_acc'],metrics['per_class_acc']=self.average_acc(data['prediction_h'],data['label_h'])
			##deb.prints(metrics['per_class_acc'])
			if self.val_set:
				deb.prints(metrics_val['f1_score_noavg'])
			
			#print('oa={}, aa={}, f1={}, f1_wght={}'.format(metrics['overall_acc'],
			#	metrics['average_acc'],metrics['f1_score'],metrics['f1_score_weighted']))
			if self.val_set:
				print('val oa={}, aa={}, f1={}, f1_wght={}'.format(metrics_val['overall_acc'],
					metrics_val['average_acc'],metrics_val['f1_score'],metrics_val['f1_score_weighted']))
			if self.batch_test_stats==True:
				if self.val_set:
				
					print("Loss. Train={}, Val={}, Test={}".format(self.metrics['train']['loss'],
						self.metrics['val']['loss'],self.metrics['test']['loss']))
				else:
					print("Loss. Train={}, Test={}".format(self.metrics['train']['loss'],self.metrics['test']['loss']))
			else:
				print("Train loss",self.metrics['train']['loss'])
			#====================END METRICS GET===========================================#
	def final_accuracy_report(self,data,epoch,metrics,init_time): 
		#self.early_stop['best_predictions']=data.patches['test']['prediction']
		print("EARLY STOP EPOCH",epoch,metrics)
		training_time=round(time.time()-init_time,2)
		print("Training time",training_time)
		metadata = "Timestamp:"+ str(round(time.time(),2))+". Model: "+self.model_type+". Training time: "+str(training_time)
		metadata = metadata + " paramsTrain.id " + paramsTrain.id 
		metadata = metadata + " F1: " + str(metrics['f1_score']) 
		metadata = metadata + " OA: " + str(metrics['overall_acc'])
		metadata = metadata + " epoch: " + str(epoch)
		metadata = metadata + "\n"
		
		deb.prints(metadata)

		txt_append("metadata.txt",metadata)
		np.save("prediction.npy",data.patches['test']['prediction'])
		np.save("labels.npy",data.patches['test']['label'])

class ModelFit(NetModel):
	def train(self, data):
		#========= VAL INIT



		ic(data.patches['train']['coords'].shape)

		self.batch['train']['size'] = 16

		# padding
		ic(data.t_len)
		ic(data.full_ims_train.shape)
		ic(self.model_t_len)
		#pdb.set_trace()
		# change magic number
		data.full_ims_train = data.addPaddingToInput(
			self.model_t_len, data.full_ims_train)

		ic(data.full_ims_train.shape)
		#pdb.set_trace()
		history = self.applyFitMethod(data)

		def PlotHistory(_model, feature, path_file = None):
			val = "val_" + feature
			
			plt.xlabel('Epoch Number')
			plt.ylabel(feature)
			plt.plot(_model.history[feature])
			plt.plot(_model.history[val])
			plt.legend(["train_"+feature, val])    
			if path_file:
				plt.savefig(path_file)
		PlotHistory(history, 'loss', path_file='loss_fig.png')

		self.graph.save('model_best_fit2.h5')		

	def applyFitMethod(self, data):
		'''
		first_batch_in = data.patches['train']['in'][0:16]
		ic(np.min(first_batch_in), np.average(first_batch_in), np.max(first_batch_in))
		first_batch_label = data.patches['train']['label'][0:16]
		ic(first_batch_label.shape)
		ic(np.unique(first_batch_label, return_counts=True))
		pdb.set_trace()
		'''
		history = self.graph.fit(data.patches['train']['in'], data.patches['train']['label'],
			batch_size = self.batch['train']['size'], 
			epochs = 70, 
			validation_data=(data.patches['val']['in'], data.patches['val']['label']),
#			callbacks = [es])
			callbacks = [MonitorNPY(
				validation=(data.patches['val']['in'], data.patches['val']['label']),
				patience=10, classes=self.class_n)],
			shuffle = False
			)
		return history

	def evaluate(self, data):	

		data.patches['test']['in'] = data.addPaddingToInputPatches(
			self.model_t_len, data.patches['test']['in'])
	
		data.patches['test']['prediction'] = self.graph.predict(data.patches['test']['in'])
		metrics_test=data.metrics_get(data.patches['test']['prediction'],
			data.patches['test']['label'],debug=2)
		deb.prints(metrics_test)
		

class ModelLoadGenerator(ModelFit):
	def applyFitMethod(self,data):
		params_train = {
			'dim': (self.t_len,self.patch_len,self.patch_len),
			'label_dim': (self.t_len,self.patch_len,self.patch_len,1),
			'batch_size': self.batch['train']['size'],
#			'n_classes': self.class_n,
			'n_classes': self.class_n + 1, # is it 6 or 5

			'n_channels': 2,
			'shuffle': False,
			'augm': False}

		params_validation = params_train.copy()
		params_validation['augm'] = False

		training_generator = DataGenerator(data.patches['train']['in'], data.patches['train']['label'], **params_train)
		validation_generator = DataGenerator(data.patches['val']['in'], data.patches['val']['label'], **params_validation)

		history = self.graph.fit_generator(generator = training_generator,
#			batch_size = self.batch['train']['size'], 
			epochs = 70, 
#			validation_data=(data.patches['val']['in'], data.patches['val']['label']),
			validation_data=validation_generator,
#			callbacks = [es])
#			callbacks = [MonitorNPY(
			callbacks = [MonitorGenerator(
				validation=validation_generator,
				patience=10, classes=self.class_n)],
			shuffle = False
			)
		return history
class ModelLoadGeneratorDebug(ModelFit):
	def applyFitMethod(self,data):
		params_train = {
			'dim': (self.t_len,self.patch_len,self.patch_len),
			'label_dim': (self.t_len,self.patch_len,self.patch_len,1),
			'batch_size': self.batch['train']['size'],
#			'n_classes': self.class_n,
			'n_classes': self.class_n + 1, # is it 6 or 5

			'n_channels': 2,
			'shuffle': True}

		training_generator = DataGenerator(data.patches['train']['in'], data.patches['train']['label'], **params_train)

		history = self.graph.fit(data.patches['train']['in'], data.patches['train']['label'],
			batch_size = self.batch['train']['size'], 
			epochs = 3, 
			validation_data=(data.patches['val']['in'], data.patches['val']['label']),
#			callbacks = [es])
			callbacks = [MonitorNPY(
				validation=(data.patches['val']['in'], data.patches['val']['label']),
				patience=10, classes=self.class_n)]
			)
		
		history = self.graph.fit_generator(generator = training_generator,
#			batch_size = self.batch['train']['size'], 
			epochs = 3, 
			validation_data=(data.patches['val']['in'], data.patches['val']['label']),
#			callbacks = [es])
			callbacks = [MonitorNPY(
				validation=(data.patches['val']['in'], data.patches['val']['label']),
				patience=10, classes=self.class_n)]
			)
		pdb.set_trace()
		return history
class ModelLoadGeneratorWithCoords(ModelFit):

	def applyFitMethod(self,data):
		ic(self.class_n)
		#pdb.set_trace()
		params_train = {
			'dim': (self.model_t_len,self.patch_len,self.patch_len),
			'label_dim': (self.patch_len,self.patch_len),
			'batch_size': self.batch['train']['size'],
#			'n_classes': self.class_n,
			'n_classes': self.class_n + 1, # it was 6. Now it is 13 + 1 = 14

			'n_channels': 2,
			'shuffle': False,
#			'printCoords': False,
			'augm': True}

		params_validation = params_train.copy()
		params_validation['augm'] = False
		params_validation['shuffle'] = False


		ic(data.patches['train']['coords'].shape)
		ic(data.patches['train']['coords'][0:16])
		ic(data.patches['val']['coords'][0:16])
		generator_type="coords"
		if generator_type=="coords":
			training_generator = DataGeneratorWithCoords(data.full_ims_train, data.full_label_train, 
				data.patches['train']['coords'], **params_train)
			validation_generator = DataGeneratorWithCoords(data.full_ims_train, data.full_label_train, 
				data.patches['val']['coords'], **params_validation)

		ic(data.patches['val']['coords'].shape)
		ic(data.patches['val']['coords'])
#		pdb.set_trace()

#		pdb.set_trace()
		history = self.graph.fit_generator(generator = training_generator,
#			batch_size = self.batch['train']['size'], 
			epochs = 70, 
			validation_data=validation_generator,
#			validation_data=(data.patches['val']['in'], data.patches['val']['label']),
#			callbacks = [es])
#			callbacks = [MonitorNPY(
			callbacks = [MonitorGenerator(
#			callbacks = [MonitorNPYAndGenerator(
#				validation=((data.patches['val']['in'], data.patches['val']['label']),validation_generator),
#				validation=(data.patches['val']['in'], data.patches['val']['label']),				
				validation=validation_generator,
				patience=10, classes=self.class_n)], # it was 5
			shuffle = False
			)

		return history
	def evaluate(self, data):	
		params_test = {
			'dim': (self.model_t_len,self.patch_len,self.patch_len),
			'label_dim': (self.patch_len,self.patch_len),
			'batch_size': 1,
#			'n_classes': self.class_n,
			'n_classes': self.class_n + 1, # it was 6. Now it is 13 + 1 = 14

			'n_channels': 2,
			'shuffle': False,
#			'printCoords': False,
			'augm': False}	

		data.full_ims_test = data.addPaddingToInput(
			self.model_t_len, data.full_ims_test)

		data.getPatchesFromCoords(data.full_label_test, data.patches['test']['coords'])
		test_generator = DataGeneratorWithCoords(data.full_ims_test, data.full_label_test, 
				data.patches['test']['coords'], **params_test)
		data.patches['test']['prediction'] = self.graph.predict_generator(test_generator)
		data.patches['test']['label'] = data.getPatchesFromCoords(
			data.full_label_test, data.patches['test']['coords'])
		metrics_test=data.metrics_get(data.patches['test']['prediction'],
			data.patches['test']['label'],debug=2)
		deb.prints(metrics_test)
class ModelLoadEachBatch(NetModel):
	def train(self,data):

		params_train = {
			'dim': (self.t_len,self.patch_len,self.patch_len),
			'label_dim': (self.t_len,self.patch_len,self.patch_len,1),
			'batch_size': self.batch['train']['size'],
			'n_classes': self.class_n,
			'n_channels': 2,
			'shuffle': True}

		params_validation = {
			'dim': (self.t_len,self.patch_len,self.patch_len),
			'label_dim': (self.t_len,self.patch_len,self.patch_len,1),
			'batch_size': self.batch['val']['size'],
			'n_classes': self.class_n,
			'n_channels': 2,
			'shuffle': False}

		training_generator = DataGenerator(data.partition['train']['in'], data.partition['train']['label'], **params_train)
		validation_generator = DataGenerator(data.partition['val']['in'], data.partition['val']['label'], **params_validation)

		# the model was already compiled using model.compile in __main__ 
		#model.compile(optimizer=Adam(lr=0.01, decay=0.00016667),
		#				loss='binary_crossentropy',
		#				metrics=['accuracy'], options = run_opts)
		es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
		# Train model on dataset
		self.graph.fit_generator(generator=training_generator,
							validation_data=validation_generator,
							use_multiprocessing=True,
							workers=9, 
							callbacks=[es])