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


#from datagenerator import DataGenerator
from generator import DataGenerator, DataGeneratorWithCoords

import matplotlib.pyplot as plt
sys.path.append('../../../dataset/dataset/patches_extract_script/')
from dataSource import DataSource, SARSource, OpticalSource, Dataset, LEM, LEM2, CampoVerde, OpticalSourceWithClouds, Humidity
from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq
from parameters.parameters_reader import ParamsTrain

from icecream import ic
from monitor import Monitor, MonitorNPY, MonitorGenerator, MonitorNPYAndGenerator
from model import ModelLoadGeneratorWithCoords
from dataset import Dataset, DatasetWithCoords

from patch_extractor import PatchExtractor
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
		if self.paramsTrain.sliceFromCoords == False:
			datasetClass = Dataset
		else:
			datasetClass = DatasetWithCoords
		self.data = datasetClass(paramsTrain = self.paramsTrain, ds = self.ds,
			dotys_sin_cos = self.dotys_sin_cos)
		ic(self.data.class_n)
#		pdb.set_trace()
	def setModel(self):
		if self.paramsTrain.sliceFromCoords == False:
			modelClass = ModelLoadGenerator
		else:
			modelClass = ModelLoadGeneratorWithCoords

		self.model = modelClass(paramsTrain = self.paramsTrain, ds = self.ds, 
						) # , self.data = self.data
		self.model.class_n=self.data.class_n-1 # Model is designed without background class
		ic(self.model.class_n)
		ic(self.data.class_n)
		deb.prints(self.data.class_n)

		self.model.build(self.paramsTrain.model_type)

		self.model.class_n+=1 # This is used in loss_weights_estimate, val_set_get, semantic_balance (To-do: Eliminate bcknd class)


	def preprocess(self, model_name_id):
		
		self.model_name = model_name_id
		ic(self.model_name)

		self.data.create_load()
		
		self.paramsTrain.class_n = self.data.class_n
		ic(self.paramsTrain.class_n)

		
		#model = ModelLoadEachBatch(epochs=self.paramsTrain.epochs, patch_len=self.paramsTrain.patch_len,
	##	modelClass = NetModel
	##	modelClass = ModelFit
	##	modelClass = ModelLoadGenerator
		self.model.name = self.model_name		

		print("=== SELECT VALIDATION SET FROM TRAIN SET")
			
		#val_set = False # fix this
		if self.paramsTrain.val_set==True:
			deb.prints(self.paramsTrain.val_set_mode)
			self.data.val_set_get(self.paramsTrain.val_set_mode,0.15)
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
						
		self.model.class_n-=1

		# Label background from 0 to last. 
		deb.prints(self.data.patches['train']['coords'].shape)

		#=========== End of moving bcknd label from 0 to last value

	def train(self):

		metrics=['accuracy']

		optim = Adam(lr=self.paramsTrain.learning_rate, beta_1=0.9)

		#loss=weighted_categorical_crossentropy_ignoring_last_label(self.model.loss_weights_ones)
		loss=categorical_focal_ignoring_last_label(alpha=0.25,gamma=2)
		#loss=weighted_categorical_focal_ignoring_last_label(self.model.loss_weights,alpha=0.25,gamma=2)

		self.model.graph.compile(loss=loss,
					optimizer=optim, metrics=metrics)

		self.model.train(self.data)

	def modelLoad(self, model_name_id):

		self.model_name = 'model_best_fit2.h5'
		self.model_name = 'model_lm_mar_nomask_good.h5'
		self.model_name = 'model_lm_jun_maize_nomask_good.h5'
		self.model_name = 'model_lm_jun_maize_nomask_good.h5'
		self.model_name = 'model_best_UUnet4ConvLSTM_jun.h5'
		self.model_name = 'model_cv_may_3classes_nomask.h5'
		self.model_name = 'model_best_fit2.h5'
#		self.model_name = 'model_lm_mar_nomask_good.h5'
#			self.model_name = 'model_best_UUnet4ConvLSTM_jun_cv_criteria_0_92.h5'
		self.model.graph=load_model(self.model_name, compile=False)		

#		self.model.evaluate(self.data)

	def evaluate(self):
		self.data.loadMask()
		self.model.evaluate(self.data, self.ds)

if __name__ == '__main__':

	paramsTrain = ParamsTrain('parameters/')

	dataset = paramsTrain.dataset

	paramsTrain.dataSource = SARSource()

	trainTest = TrainTest(paramsTrain)

	patchExtractor = PatchExtractor(paramsTrain, trainTest.ds)	
	if paramsTrain.getFullIms == True:
		patchExtractor.getFullIms()	
	else:
		patchExtractor.fullImsLoad()

	if paramsTrain.coordsExtract == True:
		patchExtractor.extract()

	trainTest.setData()
	trainTest.setModel()

	assert isinstance(str(paramsTrain.model_type), str)
	model_name_id = 'model_best_' + str(paramsTrain.model_type) + '_' + \
			paramsTrain.seq_date + '_' + paramsTrain.dataset + '_' + \
			paramsTrain.model_name + '.h5'

	trainTest.preprocess(model_name_id) # move into if
	if paramsTrain.train == True:
		trainTest.train()
	else:
		trainTest.modelLoad(model_name_id)

	trainTest.evaluate()



