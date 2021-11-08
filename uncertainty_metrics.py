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


import scipy
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
#from tensorflow.keras import metrics
import sys
import glob

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report
# Local
from src.densnet import DenseNetFCN
from src.densnet_timedistributed import DenseNetFCNTimeDistributed

#from metrics import fmeasure,categorical_accuracy
import deb
from src.loss import weighted_categorical_crossentropy, sparse_accuracy_ignoring_last_label, weighted_categorical_crossentropy_ignoring_last_label, categorical_focal_ignoring_last_label, weighted_categorical_focal_ignoring_last_label
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
from parameters.params_train import ParamsTrain

ic.configureOutput(includeContext=True)
np.random.seed(2021)
tf.random.set_seed(2021)
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


def get_mean(pred_probs):
    return np.mean(pred_probs, axis=0)

def predictive_variance(pred_probs):
    pred_var = np.var(pred_probs, axis = 0)
    pred_var = np.average(pred_var, axis = -1)
    return pred_var

epsilon = 1e-10


def predictive_entropy(pred_probs):
    pred_mean = get_mean(pred_probs) # shape (patch_len, patch_len, class_n)
    pred_entropy = np.zeros((pred_mean.shape[0:2]))

    K = pred_mean.shape[-1]
    for k in range(K):
        pred_entropy = pred_entropy + pred_mean[..., k] * np.log(pred_mean[..., k] + epsilon) 
    pred_entropy = - pred_entropy / K
    return pred_entropy


def single_experiment_entropy(pred_prob):
    pred_entropy = np.zeros(pred_prob.shape[0:2])
    ic(pred_entropy.shape)
    
    K = pred_prob.shape[-1]
    for k in range(K):
        pred_entropy = pred_entropy + pred_prob[..., k] * np.log(pred_prob[..., k] + epsilon) 
    pred_entropy = - pred_entropy / K
    return pred_entropy

def mutual_information(pred_probs):
    H = predictive_entropy(pred_probs)
    sum_entropy = 0

    n = pred_probs.shape[0]
    K = pred_probs.shape[-1]
    ic(n, K)

    for i in range(n):
        for k in range(K):
            sum_entropy = sum_entropy + pred_probs[i, ..., k] * np.log(pred_probs[i, ..., k] + epsilon)

    sum_entropy = - sum_entropy / (n * K)

    MI = H - sum_entropy
    return MI




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
dropout_flag = False
name_id = "t30"

paramsTrain = ParamsTrain('parameters/', **paramsTrainCustom)

mask = cv2.imread(str(paramsTrain.path / 'TrainTestMask.tif'),-1)
mask_flat = mask.flatten()

label_test = np.load(paramsTrain.path / 'full_ims' / 'full_label_test.npy').astype(np.uint8)[-1]
#ic(label_test.shape)

label_test = label_test.flatten()
#ic(label_test.shape)

label_test = label_test[mask_flat==2]

#ic(label_test.shape)
label_test = label_test - 1


known_classes = [0, 1, 10, 12]
unknown_id = 20
unique = np.unique(label_test)
for unique_value in unique:
	if unique_value not in known_classes:
		label_test[label_test == unique_value] = unknown_id
ic(np.unique(label_test, return_counts = True))

		
#pdb.set_trace()
'''
plt.figure()
plt.imshow(mask)
plt.axis('off')
plt.savefig('mask.png', dpi = 500)
'''

if dropout_flag == True:
	filename = 'prediction_logits_mosaic_group.npy'
	pred_probs = np.load(filename)

	for t in range(pred_probs.shape[0]):
		for c in range(pred_probs.shape[-1]):
			pred_probs[t,...,c][mask != 2] = 0
	ic(pred_probs.shape, pred_probs.dtype)

else:
	filename = 'prediction_logits_mosaic.npy'
	evidence = np.load(filename)

	def predict(test_img_input):
		# evidence = np.squeeze(model.predict(np.expand_dims(test_img_input, axis=0)))
		class_n = evidence.shape[-1]
		ic(class_n)
		alpha = evidence + 1
		u = np.squeeze(class_n / np.sum(alpha, axis= -1, keepdims=True))

		print("alpha", alpha.shape)
		print("u", u.shape)
		predictions = alpha / np.sum(alpha, axis = -1, keepdims=True)  # prob
		return predictions, u
	_, u = predict(evidence)
	ic(u.shape)
#	for c in range(pred_probs.shape[-1]):
	u[mask != 2] = 0

#pdb.set_trace()

if dropout_flag == True:
	dropout_repetitions = 30
	if dropout_repetition == 30:
		for idx in range(3):
			pred_probs[idx*10:(idx+1)*10] = scipy.special.softmax(pred_probs[idx*10:(idx+1)*10], axis=-1)
	elif dropout_repetition == 10:
			pred_probs = scipy.special.softmax(pred_probs, axis=-1)


	ic(pred_probs.shape)
#pred_probs = pred_probs[:, 1800:2090, 3910:4210]
#ic(pred_probs.shape)
metrics = MetricsTranslated(None)
def plotROC(label_test, scores, scores_flat, modelId, nameId):
	metrics.plotROCCurve(label_test, scores_flat, 
					modelId = modelId, nameId = nameId, unknown_class_id = 20)
			
	plt.figure()
	plt.imshow(scores.astype(np.float32))
	plt.axis('off')
	plt.savefig(nameId + '.png', dpi = 500)

if dropout_flag == False:
	ic(u.shape)
	u_flat = u.flatten()
	u_flat = u_flat[mask_flat == 2]
	ic(u_flat.shape, mask_flat.shape)
	print("Evidential uncertainty")
	plotROC(label_test, u, u_flat, "UUnetConvLSTM", "EvidentialDL")


elif dropout_flag == True:
	pred_entropy = predictive_entropy(pred_probs)
	np.save('pred_entropy_30.npy', pred_entropy)
	pred_entropy_flat = pred_entropy.flatten()
	pred_entropy_flat = pred_entropy_flat[mask_flat==2]
	ic(label_test.shape, pred_entropy.shape)
	print("Predictive entropy")
	plotROC(label_test, pred_entropy, pred_entropy_flat, "UUnetConvLSTM", "DropoutPredEntropy" + name_id)

	metrics.plotROCCurve(label_test, pred_entropy_flat, 
					modelId = "UUnetConvLSTM", nameId = "DropoutPredEntropy" + name_id, unknown_class_id = 20)
			
	plt.figure()
	plt.imshow(pred_entropy.astype(np.float32))
	plt.axis('off')
	plt.savefig('pred_entropy' + name_id + '.png', dpi = 500)

	MI = mutual_information(pred_probs)
	MI_flat = MI.flatten()
	MI_flat = MI_flat[mask_flat==2]
	MI_flat = np.nan_to_num(MI_flat, nan = 0.)
	print("MI")

	metrics.plotROCCurve(label_test, MI_flat, 
					modelId = "UUnetConvLSTM", nameId = "DropoutMI" + name_id, unknown_class_id = 20)

	plt.figure()
	plt.imshow(MI.astype(np.float32))
	plt.axis('off')
	plt.savefig('MI' + name_id + '.png', dpi = 500)
	#plt.show()

	pred_var = predictive_variance(pred_probs)
	pred_var_flat = pred_var.flatten()
	pred_var_flat = pred_var_flat[mask_flat==2]
	print("Predictive variance")

	metrics.plotROCCurve(label_test, pred_var_flat, 
					modelId = "UUnetConvLSTM", nameId = "DropoutPredVar" + name_id, unknown_class_id = 20)

	plt.figure()
	plt.imshow(pred_var.astype(np.float32))
	plt.axis('off')
	plt.savefig('pred_var' + name_id + '.png', dpi = 500)

	pred_entropy_single = single_experiment_entropy(pred_probs[0])
	pred_entropy_single_flat = pred_entropy_single.flatten()
	pred_entropy_single_flat = pred_entropy_single_flat[mask_flat==2]
	print("Predictive entropy single experiment")

	metrics.plotROCCurve(label_test, pred_entropy_single_flat, 
					modelId = "UUnetConvLSTM", nameId = "DropoutPredEntropySingle" + name_id, unknown_class_id = 20)

	plt.figure()
	plt.imshow(pred_entropy_single.astype(np.float32))
	plt.axis('off')
	plt.savefig('pred_entropy_single' + name_id + '.png', dpi = 500)


'''
mean_prediction = get_mean(pred_probs)

				
##np.save('pred_var.npy', pred_var)
'''

'''



'''
'''


plt.figure()
plt.imshow(pred_entropy_single.astype(np.float32))
plt.axis('off')
plt.savefig('pred_entropy_single.png', dpi = 500)




#plt.show()

plt.figure()
plt.imshow(mean_prediction.argmax(axis=-1))
plt.axis('off')
plt.savefig('mean_prediction.png', dpi = 500)
#plt.show()
'''





#pdb.set_trace()