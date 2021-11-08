
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
from loss import weighted_categorical_crossentropy, sparse_accuracy_ignoring_last_label, weighted_categorical_crossentropy_ignoring_last_label, categorical_focal_ignoring_last_label, weighted_categorical_focal_ignoring_last_label
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

#from datagenerator import DataGenerator
from generator import DataGenerator, DataGeneratorWithCoords, DataGeneratorWithCoordsRandom

import matplotlib.pyplot as plt
sys.path.append('../../../dataset/dataset/patches_extract_script/')
from dataSource import DataSource, SARSource, Dataset, LEM, LEM2, CampoVerde
from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq
from parameters.params_train import ParamsTrain

from icecream import ic
from monitor import Monitor, MonitorNPY, MonitorGenerator, MonitorNPYAndGenerator

from mosaic import seq_add_padding, add_padding, Mosaic, MosaicHighRAM, MosaicHighRAMPostProcessing
from metrics import Metrics, MetricsTranslated
from postprocessing import PostProcessingMosaic


def load_obj(name ):
	with open('obj/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class ModelCropRecognition(object):
	def __init__(self, paramsTrain, ds, 
		*args, **kwargs):

		print("Initializing object...")
		print(paramsTrain.t_len, paramsTrain.channel_n)
		self.paramsTrain = paramsTrain
		self.patch_len = paramsTrain.patch_len
		self.path = {"v": paramsTrain.path, 'train': {}, 'test': {}}
		self.image = {'train': {}, 'test': {}}
		self.patches = {'train': {}, 'test': {}}

		self.patches['train']['step']=paramsTrain.patch_step_train
		self.patches['test']['step']=paramsTrain.patch_step_test 
	  
		self.path['train']['in'] = paramsTrain.path / 'train_test/train/ims/'
		self.path['test']['in'] = paramsTrain.path / 'train_test/test/ims/'
		self.path['train']['label'] = paramsTrain.path / 'train_test/train/labels/'
		self.path['test']['label'] = paramsTrain.path / 'train_test/test/labels/'

		# in these paths, the augmented train set and validation set are stored
		# they can be loaded after (flag decides whether estimating these values and storing,
		# or loading the precomputed ones)
		self.path_patches_bckndfixed = paramsTrain.path / 'patches_bckndfixed/' 
		self.path['train_bckndfixed']=self.path_patches_bckndfixed / 'train/'
		self.path['val_bckndfixed']=self.path_patches_bckndfixed / 'val/'
		self.path['test_bckndfixed']=self.path_patches_bckndfixed / 'test/'
		self.path['test_loco'] = self.path_patches_bckndfixed / 'test_loco/'

		self.channel_n = paramsTrain.channel_n
		deb.prints(self.channel_n)
		self.debug = paramsTrain.debug
		self.class_n = paramsTrain.class_n
		self.report={'best':{}, 'val':{}}
		self.report['exp_id']=paramsTrain.exp_id
		self.report['best']['text_name']='result_'+paramsTrain.exp_id+'.txt'
		self.report['best']['text_path']='results/'+self.report['best']['text_name']
		self.report['best']['text_history_path']='results/'+'history.txt'
		self.report['val']['history_path']='results/'+'history_val.txt'
		
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


	def build(self, modelArchitecture, class_n):
		modelArchitecture.class_n = class_n
		modelArchitecture.build()

		self.graph = modelArchitecture.graph 

		with open('model_summary.txt','w') as fh:
			self.graph.summary(line_length=125,print_fn=lambda x: fh.write(x+'\n'))

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

		ic(self.name)
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

		self.graph.save('model_best_fit.h5')	
		ic(self.name)
		self.graph.save(self.name)	
			

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

	def evaluate(self, data, ds):	

		data.patches['test']['in'] = data.addPaddingToInputPatches(
			self.model_t_len, data.patches['test']['in'])
	
		data.patches['test']['prediction'] = self.graph.predict(data.patches['test']['in'])
		metrics_test=data.metrics_get(data.patches['test']['prediction'],
			data.patches['test']['label'],debug=2)
		deb.prints(metrics_test)
		
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
			if self.paramsTrain.trainGeneratorRandom == True:
				training_generator = DataGeneratorWithCoordsRandom(data.full_ims_train, data.full_label_train, 
					data.patches['train']['coords'], samples_per_epoch = 9100, **params_train)
			else:
				training_generator = DataGeneratorWithCoords(data.full_ims_train, data.full_label_train, 
					data.patches['train']['coords'], **params_train)

			validation_generator = DataGeneratorWithCoords(data.full_ims_train, data.full_label_train, 
				data.patches['val']['coords'], **params_validation)

		ic(data.patches['val']['coords'].shape)
		ic(data.patches['val']['coords'])
#		pdb.set_trace()

#		pdb.set_trace()
		history = self.graph.fit(training_generator,
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

	
	def evaluate(self, data, ds):	
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

		metrics = Metrics(self.paramsTrain)

		metrics_test=metrics.get(data.patches['test']['prediction'],
			data.patches['test']['label'],debug=2)
		deb.prints(metrics_test)




	def evaluate(self, data, ds, paramsMosaic):	
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

		_, h,w,channel_n = data.full_ims_test.shape

		data.reloadLabel()

		self.paramsMosaic = paramsMosaic

		if self.paramsTrain.openSetMethod == None:
			mosaic = MosaicHighRAM(self.paramsTrain, self.paramsMosaic)
		else:
			mosaic = MosaicHighRAMPostProcessing(self.paramsTrain, self.paramsMosaic)

		self.postProcessing = PostProcessingMosaic(self.paramsTrain, h, w)

		mosaic.create(self.paramsTrain, self, data, ds, self.postProcessing)

		metrics = MetricsTranslated(self.paramsTrain)
		metrics_test = metrics.get(mosaic.prediction_mosaic, mosaic.label_mosaic)

		deb.prints(metrics_test)

#		label_flat = mosaic.getFlatLabel()
#		scores_flat = mosaic.getFlatScores()
		if self.paramsTrain.openSetMethod != None:
			metrics.plotROCCurve(mosaic.getFlatLabel(), mosaic.getFlatScores(), 
				modelId = mosaic.name_id, nameId = mosaic.name_id, unknown_class_id = 20)

	def load_decoder_features(self, in_, prediction_dtype = np.float16, debug  = 1):
	#print(model.summary())

#		layer_names = ['conv_lst_m2d_1', 'activation_6', 'activation_8', 'activation_10']
		layer_names = ['conv_lst_m2d', 'activation_5', 'activation_7', 'activation_9']

		upsample_ratios = [8, 4, 2, 1]

		out1 = UpSampling2D(size=(upsample_ratios[0], upsample_ratios[0]))(self.graph.get_layer(layer_names[0]).output)
		out2 = UpSampling2D(size=(upsample_ratios[1], upsample_ratios[1]))(self.graph.get_layer(layer_names[1]).output)
		out3 = UpSampling2D(size=(upsample_ratios[2], upsample_ratios[2]))(self.graph.get_layer(layer_names[2]).output)
		out4 = UpSampling2D(size=(upsample_ratios[3], upsample_ratios[3]))(self.graph.get_layer(layer_names[3]).output)

		intermediate_layer_model = Model(inputs=self.graph.input, outputs=[out1, #4x4
															out2, #8x8
															out3, #16x16
															out4]) #32x32

		intermediate_features=intermediate_layer_model.predict(in_) 

		if debug > 0:
			deb.prints(intermediate_features[0].shape)
	#		intermediate_features = np.concatenate(intermediate_features, axis = 0)
	#		ic(intermediate_features.shape)
	#		pdb.set_trace()
		intermediate_features = [x.reshape(x.shape[0], -1, x.shape[-1]) for x in intermediate_features]
		if debug > 0:
			[deb.prints(intermediate_features[x].shape) for x in [0,1,2,3]]

		intermediate_features = np.squeeze(np.concatenate(intermediate_features, axis=-1))# .astype(prediction_dtype)

	#		open_features_flat = []
	#		for feature in intermediate_features:
	#			feature = feature.flatten()
	#			deb.prints(feature.shape)




		if debug > 0:
			deb.prints(intermediate_features.shape)
			print("intermediate_features stats", np.min(intermediate_features), np.average(intermediate_features), np.max(intermediate_features))
		return intermediate_features
'''
class ModelDropout(ModelCropRecognition):
	def evaluate(self, data, ds, paramsMosaic):
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

		_, h,w,channel_n = data.full_ims_test.shape

		data.reloadLabel()

		self.paramsMosaic = paramsMosaic
		times = 10

		self.prediction_logits_mosaic_group = np.ones((times, h,w, class_n)).astype(np.float16)
		for t in range(times):
			mosaic = MosaicHighRAM(self.paramsTrain, self.paramsMosaic)

			self.postProcessing = PostProcessingMosaic(self.paramsTrain, h, w)

			mosaic.create(self.paramsTrain, self, data, ds, self.postProcessing)

			mosaic.deleteAllButLogits()

			self.prediction_logits_mosaic_group[t] = mosaic.prediction_logits_mosaic.copy()
		
		self.prediction_logits_mosaic_mean = np.mean(self.prediction_logits_mosaic_group, axis = 0)
		self.prediction_logits_mosaic_std = np.std(self.prediction_logits_mosaic_group, axis = 0)
		
		ic(self.prediction_logits_mosaic_mean.shape, self.prediction_logits_mosaic_std.shape)
		pdb.set_trace()





		metrics = MetricsTranslated(self.paramsTrain)
		metrics_test = metrics.get(mosaic.prediction_mosaic, mosaic.label_mosaic)

		deb.prints(metrics_test)

#		label_flat = mosaic.getFlatLabel()
#		scores_flat = mosaic.getFlatScores()
		if self.paramsTrain.openSetMethod != None:
			metrics.plotROCCurve(mosaic.getFlatLabel(), mosaic.getFlatScores(), 
				modelId = mosaic.name_id, nameId = mosaic.name_id, unknown_class_id = 20)
'''