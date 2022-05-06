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
from src.modelManager import ModelManagerCropRecognition
from src.dataset import Dataset, DatasetWithCoords

from src.patch_extractor import PatchExtractor

from src.mosaic import seq_add_padding, add_padding, Mosaic, MosaicHighRAM, MosaicHighRAMPostProcessing
from src.postprocessing import OpenSetManager

from src.metrics import Metrics, MetricsTranslated

from train_and_evaluate import TrainTest

from src.modelArchitecture import UUnetConvLSTM, UnetSelfAttention, UUnetConvLSTMDropout, UUnetConvLSTMEvidential

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


class TrainTestEvidential(TrainTest):
	def predict(self, evidence):
		# evidence = np.squeeze(model.predict(np.expand_dims(test_img_input, axis=0)))
		class_n = evidence.shape[-1]
		ic(class_n)
		alpha = evidence + 1
		u = np.squeeze(class_n / np.sum(alpha, axis= -1, keepdims=True))

		print("alpha", alpha.shape)
		print("u", u.shape)
		predictions = alpha / np.sum(alpha, axis = -1, keepdims=True)  # prob
		return predictions.argmax(axis=-1), u	
	def infer(self, paramsMosaic):
		super().infer(paramsMosaic)
		np.save('prediction_logits_mosaic_evidential.npy', self.mosaic.prediction_logits_mosaic)
		ic("Old prediction mosaic", np.unique(self.mosaic.prediction_mosaic, return_counts = True))
		ic(np.unique(self.mosaic.prediction_logits_mosaic.argmax(axis=-1), return_counts = True))
		self.mosaic.prediction_mosaic, _ = self.predict(self.mosaic.prediction_logits_mosaic)
		self.mosaic.prediction_mosaic[self.data.mask != 2] = 255
		ic(np.unique(self.mosaic.prediction_mosaic, return_counts = True))
		self.mosaic.prediction_mosaic = self.mosaic.prediction_mosaic.astype(np.uint8)
		ic(np.unique(self.mosaic.prediction_mosaic, return_counts = True))
		#pdb.set_trace()
		self.mosaic.prediction_mosaic = self.data.newLabel2labelTranslate(self.mosaic.prediction_mosaic, 
					'results/label_translations/new_labels2labels_'+self.paramsTrain.dataset+'_'+self.data.dataset_date+'_S1.pkl')

		ic(np.unique(self.mosaic.prediction_mosaic, return_counts = True))
		self.mosaic.prediction_mosaic = self.mosaic.prediction_mosaic - 1
		ic(np.unique(self.mosaic.prediction_mosaic, return_counts = True))

		#self.data.mask
		#pdb.set_trace()

if __name__ == '__main__':


	paramsTrainCustom = {
		'getFullIms': False,
		'coordsExtract': False,
		'train': True,
		'openSetMethod': None, # Options: None, OpenPCS, OpenPCS++
#		'openSetLoadModel': True,
		'selectMainClasses': True,
		'evidentialDL': True,
		'dataset': 'lm', # lm: L Eduardo Magalhaes.
		'seq_date': 'jun',
		'id': 'evidential_700samples3',
		'model_type': UUnetConvLSTMEvidential
	}

	paramsTrain = ParamsTrain('parameters/', **paramsTrainCustom)

	paramsTrain.dataSource = SARSource()

	trainTest = TrainTestEvidential(paramsTrain)

	trainTest.main()



	paramsTrainCustom = {
		'getFullIms': False,
		'coordsExtract': False,
		'train': True,
		'openSetMethod': None, # Options: None, OpenPCS, OpenPCS++
#		'openSetLoadModel': True,
		'selectMainClasses': True,
		'evidentialDL': True,
		'dataset': 'lm', # lm: L Eduardo Magalhaes.
		'seq_date': 'jun',
		'id': 'evidential_700samples',
		'model_type': UUnetConvLSTMEvidential

	}

	paramsTrain = ParamsTrain('parameters/', **paramsTrainCustom)

	paramsTrain.dataSource = SARSource()

	trainTest = TrainTestEvidential(paramsTrain)

	trainTest.main()



	paramsTrainCustom = {
		'getFullIms': False,
		'coordsExtract': False,
		'train': True,
		'openSetMethod': None, # Options: None, OpenPCS, OpenPCS++
#		'openSetLoadModel': True,
		'selectMainClasses': True,
		'evidentialDL': True,
		'dataset': 'lm', # lm: L Eduardo Magalhaes.
		'seq_date': 'mar',
		'id': 'evidential_700samples',
		'model_type': UUnetConvLSTMEvidential

	}

	paramsTrain = ParamsTrain('parameters/', **paramsTrainCustom)

	paramsTrain.dataSource = SARSource()

	trainTest = TrainTestEvidential(paramsTrain)

	trainTest.main()
