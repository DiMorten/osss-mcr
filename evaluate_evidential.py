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
from src.densnet import DenseNetFCN
from src.densnet_timedistributed import DenseNetFCNTimeDistributed

#from metrics import fmeasure,categorical_accuracy
import deb
from src.loss import weighted_categorical_crossentropy, sparse_accuracy_ignoring_last_label, weighted_categorical_crossentropy_ignoring_last_label, categorical_focal_ignoring_last_label, weighted_categorical_focal_ignoring_last_label, evidential_categorical_focal_ignoring_last_label
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
from src.generator import DataGenerator, DataGeneratorWithCoords

import matplotlib.pyplot as plt
# sys.path.append('../../../dataset/dataset/patches_extract_script/')
from src.dataSource import DataSource, SARSource, Dataset, LEM, LEM2, CampoVerde
from src.model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq
from parameters.params_train import ParamsTrain
from parameters.params_mosaic import ParamsReconstruct

from icecream import ic
from src.monitor import Monitor, MonitorNPY, MonitorGenerator, MonitorNPYAndGenerator
from src.model import ModelCropRecognition
from src.dataset import Dataset, DatasetWithCoords

from src.patch_extractor import PatchExtractor

from src.mosaic import seq_add_padding, add_padding, Mosaic, MosaicHighRAM, MosaicHighRAMPostProcessing
from src.postprocessing import PostProcessingMosaic

from src.metrics import Metrics, MetricsTranslated

ic.configureOutput(includeContext=True)
np.random.seed(2021)
tf.random.set_seed(2021)
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

def model_summary_print(s):
	with open('model_summary.txt','w+') as f:
		print(s, file=f)


def txt_append(filename, append_text):
	with open(filename, "a") as myfile:
		myfile.write(append_text)

def sizeof_fmt(num, suffix='B'):
	''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
	for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
		if abs(num) < 1024.0:
			return "%3.1f %s%s" % (num, unit, suffix)
		num /= 1024.0
	return "%.1f %s%s" % (num, 'Yi', suffix)
# ========== NetModel object implements model graph definition, train/testing, early stopping ================ #

flag = {"data_create": 2, "label_one_hot": True}

class TrainTest():

	def __init__(self, paramsTrain):
		self.paramsTrain = paramsTrain

		
		if self.paramsTrain.dataset=='lm':
			self.ds=LEM(self.paramsTrain.seq_mode, self.paramsTrain.seq_date, self.paramsTrain.seq_len)
		elif self.paramsTrain.dataset=='l2':
			self.ds=LEM2(self.paramsTrain.seq_mode, self.paramsTrain.seq_date, self.paramsTrain.seq_len)
		elif self.paramsTrain.dataset=='cv':
			self.ds=CampoVerde(self.paramsTrain.seq_mode, self.paramsTrain.seq_date, self.paramsTrain.seq_len)

		deb.prints(self.ds)
		
		self.ds.addDataSource(paramsTrain.dataSource)

		dotys, dotys_sin_cos = self.ds.getDayOfTheYear()

		self.paramsTrain.t_len = self.ds.t_len # modified?
		self.paramsTrain.dotys_sin_cos = dotys_sin_cos
		self.dotys_sin_cos = dotys_sin_cos

	def setData(self):
		self.data = DatasetWithCoords(paramsTrain = self.paramsTrain, ds = self.ds,
			dotys_sin_cos = self.dotys_sin_cos)
		ic(self.data.class_n)

		self.data.create_load()
		self.data.loadMask()

#		pdb.set_trace()
	def setModel(self, model_name_id):

		#self.model_name = model_name_id
		#if self.paramsTrain.dropoutInference == False:
		modelObj = ModelCropRecognition
		#else:
	#		modelObj = ModelDropout
		
		self.model = modelObj(paramsTrain = self.paramsTrain, ds = self.ds) 
		self.model.class_n=self.data.class_n-1 # Model is designed without background class

		self.model.name = model_name_id
		ic(self.model.name)

		ic(self.model.name)	

		ic(self.model.class_n)
		ic(self.data.class_n)
		deb.prints(self.data.class_n)

		self.model.build(self.paramsTrain.model_type, self.data.class_n - 1) # no bcknd


	def preprocess(self):
		

		
		self.paramsTrain.class_n = self.data.class_n
		ic(self.paramsTrain.class_n)

		


		print("=== SELECT VALIDATION SET FROM TRAIN SET")
			
		#val_set = False # fix this
		ic(self.paramsTrain.val_set)
		if self.paramsTrain.val_set==True:
			deb.prints(self.paramsTrain.val_set_mode)
			self.data.val_set_get(self.paramsTrain.val_set_mode,0.15)
			ic(self.data.patches['val']['coords'].shape)
#			pdb.set_trace()
		else:
			self.data.patches['val']={}

			self.data.patches['val']['label']=np.zeros((1,1))
			self.data.patches['val']['in']=np.zeros((1,1))
			
			deb.prints(self.data.patches['val']['label'].shape)
			
		#balancing=False
		
		if self.paramsTrain.balancing==True:
			print("=== AUGMENTING TRAINING DATA")

			#ic(self.data.class_n)
			#self.paramsTrain.samples_per_class = int(self.data.patches['train']['coords'].shape[0] / self.data.class_n)
			#ic(self.paramsTrain.samples_per_class)
#			pdb.set_trace()

			if self.paramsTrain.seq_mode=='fixed' or self.paramsTrain.seq_mode=='fixed_label_len':
				label_type = 'Nto1'
			elif self.paramsTrain.seq_mode=='var' or self.paramsTrain.seq_mode=='var_label':	
				label_type = 'NtoN'
			deb.prints(label_type)
			print("Before balancing:")

			self.data.semantic_balance(self.paramsTrain.samples_per_class,label_type = label_type) #More for known classes few. Compare with 500 later
						
		##self.model.class_n-=1

		# Label background from 0 to last. 
		deb.prints(self.data.patches['train']['coords'].shape)

		#=========== End of moving bcknd label from 0 to last value

	def train(self):

		metrics=['accuracy']

		optim = Adam(lr=self.paramsTrain.learning_rate, beta_1=0.9)
		
		#loss=weighted_categorical_crossentropy_ignoring_last_label(self.model.loss_weights_ones)
		if self.paramsTrain.evidentialDL == False:
			loss=categorical_focal_ignoring_last_label(alpha=0.25,gamma=2)
		else:
			current_epoch = K.variable(0.0)
			loss=evidential_categorical_focal_ignoring_last_label(alpha=0.25,gamma=2)
		
		#loss=weighted_categorical_focal_ignoring_last_label(self.model.loss_weights,alpha=0.25,gamma=2)

		self.model.graph.compile(loss=loss,
					optimizer=optim, metrics=metrics)

		self.model.train(self.data)



	def setPostProcessing(self):
		_, h,w,channel_n = self.data.full_ims_test.shape
		self.postProcessing = PostProcessingMosaic(self.paramsTrain, h, w)

		known_classes = [x + 1 for x in self.paramsTrain.known_classes]
		self.postProcessing.openSetActivate(self.paramsTrain.openSetMethod, known_classes)


	def mosaicCreate(self, paramsMosaic):

		self.data.full_ims_test = self.data.addPaddingToInput(
			self.model.model_t_len, self.data.full_ims_test)


		self.data.reloadLabel()

		if self.paramsTrain.openSetMethod == None:
			self.mosaic = MosaicHighRAM(self.paramsTrain, paramsMosaic)
			self.postProcessing = None
		else:
			self.mosaic = MosaicHighRAMPostProcessing(self.paramsTrain, paramsMosaic)


		self.mosaic.create(self.paramsTrain, self.model, self.data, self.ds, self.postProcessing)

		np.save('prediction_logits_mosaic.npy', self.mosaic.prediction_logits_mosaic)

	def evaluate(self):

		metrics = MetricsTranslated(self.paramsTrain)
		metrics_test = metrics.get(self.mosaic.prediction_mosaic, self.mosaic.label_mosaic)

		deb.prints(metrics_test)

		if self.paramsTrain.openSetMethod != None:
			metrics.plotROCCurve(self.mosaic.getFlatLabel(), self.mosaic.getFlatScores(), 
				modelId = self.mosaic.name_id, nameId = self.mosaic.name_id, unknown_class_id = 20)

	def fitPostProcessing(self):
		if self.paramsTrain.openSetLoadModel == True:
			self.postProcessing.openSetMosaic.loadFittedModel()
		else:
			if self.paramsTrain.openSetMethod == 'OpenPCS' or self.paramsTrain.openSetMethod == 'OpenPCS++':
				prediction_dtype = np.float16

				# first, translate self.train_label

				ic(self.data.full_label_train.shape)

				ic(np.unique(self.data.full_label_train, return_counts=True))

				label_with_unknown_train = self.data.getTrainLabelWithUnknown()

				ic(self.data.full_label_train.shape)
				ic(np.unique(self.data.full_label_train, return_counts=True))

				self.data.full_ims_train = self.data.addPaddingToInput(
					self.model.model_t_len, self.data.full_ims_train)

				self.data.patches_in = self.data.getSequencePatchesFromCoords(
					self.data.full_ims_train, self.data.patches['train']['coords']).astype(prediction_dtype) # test coords is called self.coords, make custom init in this class. self.full_ims is also set independent
				self.data.patches_label = self.data.getPatchesFromCoords(
					label_with_unknown_train, self.data.patches['train']['coords'])
	#       	self.coords = self.data.patches['train']['coords'] # not needed. use train coords directly
				

				self.data.predictions=(self.model.graph.predict(self.data.patches_in)).argmax(axis=-1).astype(np.uint8) 

				self.data.setDateList(self.paramsTrain)

				ic(np.unique(self.data.predictions, return_counts=True))

				self.data.predictions = self.data.newLabel2labelTranslate(self.data.predictions, 
						'results/label_translations/new_labels2labels_'+self.paramsTrain.dataset+'_'+self.data.dataset_date+'_S1.pkl')

				if self.paramsTrain.openSetMethod =='OpenPCS' or self.paramsTrain.openSetMethod =='OpenPCS++':
					self.data.intermediate_features = self.model.load_decoder_features(self.data.patches_in).astype(prediction_dtype)
				else:
					self.data.intermediate_features = self.data.predictions.copy() # to-do: avoid copy
				ic(self.data.patches_in.shape, self.data.patches_label.shape)
				ic(self.data.predictions.shape)
				ic(self.data.intermediate_features.shape)

				self.data.patches_label = self.data.patches_label.flatten()
				self.data.predictions = self.data.predictions.flatten()
				self.data.intermediate_features = np.reshape(self.data.intermediate_features,(-1, self.data.intermediate_features.shape[-1]))

				self.data.intermediate_features = self.data.intermediate_features[self.data.patches_label!=0]
				self.data.predictions = self.data.predictions[self.data.patches_label!=0]
				self.data.patches_label = self.data.patches_label[self.data.patches_label!=0]

				self.data.predictions = self.data.predictions - 1
				self.data.patches_label = self.data.patches_label - 1

				ic(np.unique(self.data.patches_label, return_counts=True))
				ic(np.unique(self.data.predictions, return_counts=True))

				ic(self.data.patches_in.shape, self.data.patches_label.shape)
				ic(self.data.predictions.shape)
				ic(self.data.intermediate_features.shape)
				
	#			pdb.set_trace()
				self.postProcessing.fit(self.data)	

	def main(self):				

		patchExtractor = PatchExtractor(self.paramsTrain, self.ds)	
		if self.paramsTrain.getFullIms == True:
			patchExtractor.getFullIms()	
		else:
			patchExtractor.fullImsLoad()

		if self.paramsTrain.coordsExtract == True:
			patchExtractor.extract()
		del patchExtractor
		
		self.setData() 

		self.preprocess() # validation set, and data augmentation

		self.setModel(self.paramsTrain.model_name_id)

		if self.paramsTrain.train == True:
			self.train()
		else:
			self.model.graph = load_model(self.paramsTrain.model_name_id, compile=False)

		paramsMosaic = ParamsReconstruct(self.paramsTrain)

		if self.paramsTrain.openSetMethod != None:
			self.setPostProcessing()
			self.fitPostProcessing()

		self.mosaicCreate(paramsMosaic)
		self.evaluate()

class TrainTestDropout(TrainTest):
	def mosaicCreate(self, paramsMosaic):

		self.data.full_ims_test = self.data.addPaddingToInput(
			self.model.model_t_len, self.data.full_ims_test)

		_, h,w,channel_n = self.data.full_ims_test.shape

		self.data.reloadLabel()

		times = 30
		ic(times, h,w, self.model.class_n)
		self.prediction_logits_mosaic_group = np.ones((times, h,w, self.model.class_n), dtype = np.float16)
		for t in range(times):
			self.mosaic = MosaicHighRAM(self.paramsTrain, paramsMosaic)
			self.postProcessing = None

			self.mosaic.create(self.paramsTrain, self.model, self.data, self.ds, self.postProcessing)

			#self.mosaic.deleteAllButLogits()

			self.prediction_logits_mosaic_group[t] = self.mosaic.prediction_logits_mosaic.copy()


		self.prediction_logits_mosaic_mean = np.mean(self.prediction_logits_mosaic_group, axis = 0)
		self.prediction_logits_mosaic_std = np.std(self.prediction_logits_mosaic_group, axis = 0)

		np.save('prediction_logits_mosaic_group.npy', self.prediction_logits_mosaic_group)
		
		ic(self.prediction_logits_mosaic_mean.shape, self.prediction_logits_mosaic_std.shape)
		pdb.set_trace()

if __name__ == '__main__':


	paramsTrainCustom = {
		'getFullIms': False,
		'coordsExtract': False,
		'train': False,
		'openSetMethod': None, # Options: None, OpenPCS, OpenPCS++
#		'openSetLoadModel': True,
		'selectMainClasses': True,
		'evidentialDL': True,
		'dataset': 'lm', # lm: L Eduardo Magalhaes.
		'seq_date': 'mar'
	}

	paramsTrain = ParamsTrain('parameters/', **paramsTrainCustom)

	paramsTrain.dataSource = SARSource()

	trainTest = TrainTest(paramsTrain)

	trainTest.main()
