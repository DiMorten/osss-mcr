from colorama import init
init()

import numpy as np
from sklearn.utils import shuffle
import cv2
import argparse


import scipy
import sys
import glob

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report
# Local
from src.densnet import DenseNetFCN
from src.densnet_timedistributed import DenseNetFCNTimeDistributed

#from metrics import fmeasure,categorical_accuracy
import deb
from src.loss import weighted_categorical_crossentropy, sparse_accuracy_ignoring_last_label, weighted_categorical_crossentropy_ignoring_last_label, categorical_focal_ignoring_last_label, weighted_categorical_focal_ignoring_last_label
import time
import pickle
#from keras_self_attention import SeqSelfAttention
import pdb
import pathlib
from pathlib import Path, PureWindowsPath
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
from parameters.params_train import ParamsTrain
from sklearn.metrics import f1_score
from stellargraph.calibration import plot_reliability_diagram, expected_calibration_error
from stellargraph.calibration import TemperatureCalibration
from sklearn.calibration import calibration_curve
from sklearn import model_selection
ic.configureOutput(includeContext=True)
np.random.seed(2021)
import tensorflow as tf
tf.random.set_seed(2021)
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.experimental.output_all_intermediates(True)
from train_and_evaluate import TrainTest, TrainTestDropout

paramsTrainCustom = {
		'getFullIms': True,
		'coordsExtract': True,
		'train': True,
		'openSetMethod': None, # Options: None, OpenPCS, OpenPCS++
#		'openSetLoadModel': True,
		'selectMainClasses': True,
		'dataset': 'lm', # lm: L Eduardo Magalhaes.
		'seq_date': 'mar'
	}

mode = 'dropout' # dropout, evidential, closed_set
#mode = 'closed_set'
#mode = 'evidential'

name_id = ""
paramsTrain = ParamsTrain('parameters/', **paramsTrainCustom)

mask = cv2.imread(str(paramsTrain.path / 'TrainTestMask.tif'),-1)
mask_flat = mask.flatten()

label_test = np.load(paramsTrain.path / 'full_ims' / 'full_label_test.npy').astype(np.uint8)[-1]
#ic(label_test.shape)

label_test = label_test.flatten()
#ic(label_test.shape)

label_test = label_test[mask_flat==2]

filename = 'prediction_logits_mosaic.npy'
filename = 'prediction_logits_mosaic.npy'


pred_prob = np.load(filename)
ic(pred_prob.shape)


def pred_prob_flatten_split(pred_prob, split="test"):
	if split=="test":
		split_id = 2
	elif split=="train":
		split_id = 1
	split_count = mask_flat[mask_flat == 2].shape[0]
	pred_prob_split = np.zeros((split_count, pred_prob.shape[-1]))
	for c in range(pred_prob.shape[-1]):
		pred_prob_split[:,c] = pred_prob[...,c].flatten()[mask_flat == 2]
	return pred_prob_split


def label_unknown_classes_as_id(label_test, known_classes = [0, 1, 10, 12],
	unknown_id = 20):
	unique = np.unique(label_test)
	for unique_value in unique:
		if unique_value not in known_classes:
			label_test[label_test == unique_value] = unknown_id
	ic(np.unique(label_test, return_counts = True))
	return label_test

def delete_unknown_samples(softmax, label_test):
	label_test_tmp = label_test[label_test!=unknown_id]
	ic(np.unique(label_test_tmp, return_counts = True))

	softmax_tmp = np.zeros((label_test_tmp.shape[0], softmax.shape[-1]))

	for chan in range(softmax_tmp.shape[-1]):
		softmax_tmp[..., chan] = softmax[..., chan][label_test!=unknown_id]
	label_test = label_test_tmp
	softmax = softmax_tmp
	return softmax, label_test


def vector_to_one_hot(a):
	def idx_to_incremental(a):
		unique = np.unique(a)
		for idx, value in enumerate(unique):
			a[a==value] = idx
		return a
	a = idx_to_incremental(a)
	b = np.zeros((a.size, a.max()+1))
	b[np.arange(a.size),a] = 1
	return b


#pdb.set_trace()
def get_calibration_data(softmax, label_test):
	calibration_data = []
	for i in range(softmax.shape[1]):  # iterate over classes
		calibration_data.append(
			calibration_curve(
				y_prob=softmax[:, i], y_true=label_test[:, i], n_bins=10, normalize=True
			)
		)
	return calibration_data

def get_ece(softmax, calibration_data):
	ece = []
	for i in range(softmax.shape[1]):
		fraction_of_positives, mean_predicted_value = calibration_data[i]
		ece.append(
			expected_calibration_error(
				prediction_probabilities=softmax[:, i],
				accuracy=fraction_of_positives,
				confidence=mean_predicted_value,
			)
		)
	return ece

pred_prob_test = pred_prob_flatten_split(pred_prob, split="test")

# apply calibration
# T = 1.2016096
T = 0.5
T = 0.1
T = 200
T = 2
T = 0.3
T = 0.05
T = 1
T = 1
T = 8.612723
# T = 17.2
pred_prob_test = pred_prob_test / T
softmax = scipy.special.softmax(pred_prob_test, axis=-1)
ic(softmax.shape, label_test.shape)

label_test = label_test - 1

unknown_id = 20

label_test = label_unknown_classes_as_id(label_test, unknown_id = unknown_id)
ic(softmax.shape, label_test.shape)

softmax, label_test = delete_unknown_samples(softmax, label_test)
ic(np.unique(label_test, return_counts = True))

label_test = vector_to_one_hot(label_test)    
ic(softmax.shape, label_test.shape)

calibration_data = get_calibration_data(softmax, label_test)
ece = get_ece(softmax, calibration_data)
ic(ece)
plot_reliability_diagram(calibration_data, softmax, ece=ece)
plt.show()

trainTemperature = False

if trainTemperature == True:

	def loadVal(paramsTrain):
		paramsTrain.dataSource = SARSource()

		trainTest = TrainTest(paramsTrain)

		trainTest.setData()
		trainTest.preprocess()

		trainTest.data.getValSamplesFromCoords()

		ic(trainTest.data.patches['val']['label'].shape)
		ic(trainTest.data.patches['val']['in'].shape)
		return paramsTrain, trainTest
	def predictVal(trainTest):
		predictionsVal = trainTest.modelManager.model.predict(trainTest.data.patches['val']['in'])
		ic(predictionsVal.shape)

		predictionsValFlat = np.reshape(predictionsVal, (-1, predictionsVal.shape[-1]))
		return predictionsValFlat

	def temperatureScale(predictionsValFlat, labelsValFlat, epochs=1000):
		x_cal_train, x_cal_val, y_cal_train, y_cal_val = model_selection.train_test_split(
			predictionsValFlat, labelsValFlat)
		
		ic(x_cal_train.shape, x_cal_val.shape, y_cal_train.shape, y_cal_val.shape)

		calibration_model_temperature = TemperatureCalibration(epochs=epochs)
		ic(calibration_model_temperature)

		calibration_model_temperature.fit(
				x_train=x_cal_train, y_train=y_cal_train, x_val=x_cal_val, y_val=y_cal_val
			)	
		ic(calibration_model_temperature)
		pdb.set_trace()

		calibration_model_temperature.plot_training_history()
		plt.show()

		return calibration_model_temperature
		
	def removeBckndClass(predictionsValFlat, labelsValFlat):
		bcknd_id = np.unique(labelsValFlat)[-1]
		predictionsValFlat = predictionsValFlat[labelsValFlat != bcknd_id]
		labelsValFlat = labelsValFlat[labelsValFlat != bcknd_id]
		return predictionsValFlat, labelsValFlat
	paramsTrain = ParamsTrain('parameters/', **paramsTrainCustom)

	paramsTrain, trainTest = loadVal(paramsTrain)
	trainTest.setModelManager(paramsTrain.model_name_id)

	predictionsValFlat = predictVal(trainTest)

	labelsValFlat = trainTest.data.patches['val']['label'].flatten()

	ic(predictionsValFlat.shape, labelsValFlat.shape)
	ic(np.unique(labelsValFlat)) # bcknd is last

	predictionsValFlat, labelsValFlat = removeBckndClass(predictionsValFlat, labelsValFlat)
	labelsValFlat = vector_to_one_hot(labelsValFlat)
	ic(predictionsValFlat.shape, labelsValFlat.shape)
	calibration_model_temperature = temperatureScale(predictionsValFlat, labelsValFlat, epochs=100000)
	ic(calibration_model_temperature.temperature)
	
	plt.clf()
	plt.plot(calibration_model_temperature.history[:, 2])
	plt.title('Temperature')
	plt.ylabel('Temperature')
	plt.xlabel('Epoch')
	plt.show()
	plt.savefig('temperature_history.png', dpi=200)










	pdb.set_trace()
	
	


'''
# === get coords test
def get_coords_test(mask):
	coords_test = []
	for h in range(mask.shape[0]):
		for w in range(mask.shape[1]):
			if mask[h,w] == 2:
				coords_test.append([h,w])
	return coords_test

coords_test = get_coords_test(mask)

def im_from_coords_samples(x, mask, coords):
	im = np.zeros_like(mask, dtype = np.float32)
	im.shape
	for count, coord in enumerate(coords):
		h, w = coord
		im[h,w] = x[count]

	return im

def im_save(im, nameId):
	plt.figure()
	plt.imshow(im.astype(np.float32))
	plt.axis('off')
	plt.savefig(nameId + '.png', dpi = 500)	


#ic(label_test.shape)
label_test = label_test - 1


known_classes = [0, 1, 10, 12]
unknown_id = 20
unique = np.unique(label_test)
for unique_value in unique:
	if unique_value not in known_classes:
		label_test[label_test == unique_value] = unknown_id
ic(np.unique(label_test, return_counts = True))

pred_probs_test = np.load('prediction_logits_mosaic_group_test.npy')

pred_probs = scipy.special.softmax(pred_probs_test, axis=-1)

ic(pred_probs.shape) # N, classes expected
# poner temp calibration
'''