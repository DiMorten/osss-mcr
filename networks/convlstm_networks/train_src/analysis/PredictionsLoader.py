from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose
# from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.optimizers import Adam,Adagrad 
from keras.models import Model
from keras import backend as K
import keras

import numpy as np
from sklearn.utils import shuffle
import cv2
#from skimage.util import view_as_windows
import argparse
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import metrics
import sys
import glob
import pdb
import pickle

sys.path.append('../../../../dataset/dataset/patches_extract_script/')
from dataSource import DataSource, SARSource, OpticalSource, Dataset, LEM, LEM2, CampoVerde, OpticalSourceWithClouds, Humidity

sys.path.append('../')
from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixed_PaddedSeq
import deb
class PredictionsLoader():
	def __init__(self):
		pass


class PredictionsLoaderNPY(PredictionsLoader):
	def __init__(self):
		pass
	def loadPredictions(self,path_predictions, path_labels):
		return np.load(path_predictions, allow_pickle=True), np.load(path_labels, allow_pickle=True)

class PredictionsLoaderModel(PredictionsLoader):
	def __init__(self, path_test):
		self.path_test=path_test
	def loadPredictions(self,path_model):
		print("============== loading model =============")
		model=load_model(path_model, compile=False)
		print("Model", model)
		print("Loading in data: ",self.path_test+'patches_in.npy')
		test_in=np.load(self.path_test+'patches_in.npy',mmap_mode='r')
		test_label=np.load(self.path_test+'patches_label.npy')

		test_predictions = model.predict(test_in)
		print(test_in.shape, test_label.shape, test_predictions.shape)
		print("Test predictions dtype",test_predictions.dtype)
		del test_in
		return test_predictions, test_label
	def loadModel(self,path_model):
		model=load_model(path_model, compile=False)
		return model


class PredictionsLoaderModelNto1(PredictionsLoaderModel):
	def __init__(self, path_test, dataset):
		self.path_test=path_test
		self.dataset=dataset
	def loadPredictions(self,path_model):
		print("============== loading model =============")
		model=load_model(path_model, compile=False)
		print("Model", model)
		print("Loading in data: ",self.path_test+'patches_in.npy')
		test_in=np.load(self.path_test+'patches_in.npy',mmap_mode='r')
#		test_label=np.load(self.path_test+'patches_label.npy')
		test_label=np.load(self.path_test+'patches_label.npy')[:,-1] # may18
		

		# add doty

		#	if dataset=='lm':
		ds=LEM()
		dataSource = SARSource()
		ds.addDataSource(dataSource)
		dotys, dotys_sin_cos = ds.getDayOfTheYear()

		def addDoty(input_):
			
			input_ = [input_, dotys_sin_cos]
			return input_

		dotys_sin_cos = np.expand_dims(dotys_sin_cos,axis=0) # add batch dimension
		dotys_sin_cos = np.repeat(dotys_sin_cos,test_in.shape[0],axis=0)

		#test_in = addDoty(test_in)
		# Here do N to 1 prediction for last timestep at first...
		test_predictions = model.predict(test_in)
		print(test_in[0].shape, test_label.shape, test_predictions.shape)
		print("Test predictions dtype",test_predictions.dtype)
		#pdb.set_trace()
		del test_in
		return test_predictions, test_label


class PredictionsLoaderModelNto1FixedSeqFixedLabel(PredictionsLoaderModelNto1):
	def newLabel2labelTranslate(self, label, filename, bcknd_flag=True):
		print("Entering newLabel2labelTranslate")
		label = label.astype(np.uint8)
		# bcknd to 0
		deb.prints(np.unique(label,return_counts=True))
		deb.prints(np.unique(label)[-1])
		if bcknd_flag == True:
			label[label==np.unique(label)[-1]] = 255 # this -1 will be different for each dataset
		deb.prints(np.unique(label,return_counts=True))
		label = label + 1
		
		deb.prints(np.unique(label,return_counts=True))

		# translate 
		f = open(filename, "rb")
		new_labels2labels = pickle.load(f)
		deb.prints(new_labels2labels)

		classes = np.unique(label)
		deb.prints(classes)
		translated_label = label.copy()
		for j in range(len(classes)):
			print(classes[j])
			print("Translated",new_labels2labels[classes[j]])
			translated_label[label == classes[j]] = new_labels2labels[classes[j]]

		# bcknd to last
		##label = label - 1 # bcknd is 255
		##label[label==255] = np.unique(label)[-2]
		return translated_label 
	def npyLoadPredictions(self, seq_date):
		batch = {}
		dated_patches_name =True
		if dated_patches_name==False:

			batch['in']=np.load(self.path_test+'patches_in.npy',mmap_mode='r') # len is 21
	#		test_label=np.load(self.path_test+'patches_label.npy')
			batch['label']=np.load(self.path_test+'patches_label.npy') # may18
		else:
			batch['in']=np.load(self.path_test+'patches_in_fixed_'+seq_date+'.npy',mmap_mode='r') # len is 21
	#		test_label=np.load(self.path_test+'patches_label.npy')
			deb.prints(self.path_test+'patches_label_fixed_'+seq_date+'.npy')
			batch['label']=np.load(self.path_test+'patches_label_fixed_'+seq_date+'.npy') # may18			


		deb.prints(batch['in'].shape)
		deb.prints(batch['label'].shape)
		return batch
	def loadPredictions(self,path_model,seq_date=None, model_dataset=None):
		print("============== loading model =============")
		model=load_model(path_model, compile=False)
		print("Model", model)
		print("Loading in data: ",self.path_test+'patches_in.npy')

		batch = self.npyLoadPredictions(seq_date)

		self.labeled_dates = 12

		one_hot_label=True
		if one_hot_label==True:
			batch['label'] = batch['label'].argmax(axis=-1)

		#pdb.set_trace()
		
#		self.mim = MIMVarLabel_PaddedSeq()
#		self.mim = MIMFixed()
		self.mim = MIMFixed_PaddedSeq()

		data = {'labeled_dates': 12}
		data['labeled_dates'] = 12

		
		#batch = {'in': test_in, 'label': test_label}

		# add doty

		if self.dataset=='lm':
			ds=LEM('fixed', seq_date)
		elif self.dataset=='l2':
			ds=LEM2('fixed', seq_date)
		dataSource = SARSource()
		ds.addDataSource(dataSource)
	
		time_delta = ds.getTimeDelta(delta=True,format='days')
		ds.setDotyFlag(False)
		dotys, dotys_sin_cos = ds.getDayOfTheYear()
		ds.dotyReplicateSamples(sample_n = batch['label'].shape[0])

		prediction_dtype = np.float16
		
		model_t_len = 12
		batch['shape'] = (batch['in'].shape[0], model_t_len) + batch['in'].shape[2:]

		# model dataset is to get the correct last date from the model dataset

		if model_dataset=='lm':
			train_ds=LEM('fixed',seq_date)
		elif model_dataset=='l2':
			train_ds=LEM2('fixed', seq_date)
		train_ds.addDataSource(SARSource())
		# get model class n
		model_shape = model.layers[-1].output_shape
		model_class_n = model_shape[-1]
		deb.prints(model_shape)
		deb.prints(model_class_n)
#		test_predictions = np.zeros_like(batch['label'][...,:-1	], dtype = prediction_dtype)
		test_predictions = np.zeros((batch['label'].shape[0],)+model_shape[1:], 
			dtype = prediction_dtype)
		deb.prints(test_predictions.shape)

		#pdb.set_trace()

		input_ = self.mim.batchTrainPreprocess(batch, ds,  
					label_date_id = -1) # tstep is -12 to -1
		deb.prints(input_[1].shape)

		#pdb.set_trace()
		test_predictions=(model.predict(input_)).astype(prediction_dtype) 
		#pdb.set_trace()
		print(" shapes", test_predictions.shape, batch['label'].shape)
		
		print("batch['in'][0].shape, batch['label'].shape, test_predictions.shape",batch['in'][0].shape, batch['label'].shape, test_predictions.shape)
		print("Test predictions dtype",test_predictions.dtype)

		deb.prints(np.unique(test_predictions.argmax(axis=-1), return_counts=True))
		deb.prints(np.unique(batch['label'], return_counts=True))

		print(" shapes", test_predictions.shape, batch['label'].shape)
		self.test_pred_proba = test_predictions.copy()

		test_predictions = test_predictions.argmax(axis=-1)
		#batch['label'] = batch['label'].argmax(axis=-1)
		print(" shapes", test_predictions.shape, batch['label'].shape)
		print( "uniques",np.unique(test_predictions, return_counts=True),np.unique(batch['label'], return_counts=True))
		
		translate_mode=True
		deb.prints(translate_mode)

		if translate_mode==True:
			translate_label_path = '../'
			test_predictions = self.newLabel2labelTranslate(test_predictions, 
					#translate_label_path + 'new_labels2labels_lm_20171209_S1.pkl',
					translate_label_path + 'new_labels2labels_'+model_dataset+'_'+train_ds.im_list[-1]+'.pkl',
					bcknd_flag=False)
						
			batch['label'] = self.newLabel2labelTranslate(batch['label'], 
					translate_label_path + 'new_labels2labels_'+self.dataset+'_'+ds.im_list[-1]+'.pkl',
					bcknd_flag=True)
		print("End shapes", test_predictions.shape, batch['label'].shape)
		print(" shapes", test_predictions.shape, batch['label'].shape)
		print( "uniques",np.unique(test_predictions, return_counts=True),np.unique(batch['label'], return_counts=True))
		#pdb.set_trace()

		
		del batch['in']
		return test_predictions, batch['label']

class PredictionsLoaderModelNto1FixedSeqFixedLabelOpenSet(PredictionsLoaderModelNto1FixedSeqFixedLabel):
	def npyLoadPredictions(self, seq_date):
		batch = {}
		dated_patches_name =True
		if dated_patches_name==False:

			batch['in']=np.load(self.path_test+'patches_in.npy',mmap_mode='r') # len is 21
	#		test_label=np.load(self.path_test+'patches_label.npy')
			batch['label']=np.load(self.path_test+'patches_label.npy') # may18
		else:
			batch['in']=np.load(self.path_test+'patches_in_fixed_'+seq_date+'.npy',mmap_mode='r') # len is 21
	#		test_label=np.load(self.path_test+'patches_label.npy')
			deb.prints(self.path_test+'patches_label_fixed_'+seq_date+'.npy')
			batch['label']=np.load(self.path_test+'patches_label_fixed_'+seq_date+'.npy') # may18

		self.loco_class = 8			
		batch['label_with_loco_class']=np.load(self.path_test+'patches_label_fixed_'+seq_date+'_loco'+str(self.loco_class)+'.npy') # may18			
		# test label with loco class. 
		# If loco_class=8, batch['label_with_loco_class'] contains the loco class as 8+1=9 because 0 is the background ID
		deb.prints(np.unique(batch['label_with_loco_class'], return_counts=True))

		batch['label_with_loco_class'][batch['label_with_loco_class']!=self.loco_class+1]=0 
		deb.prints(np.unique(batch['label_with_loco_class'], return_counts=True))

		self.label_with_loco_class = batch['label_with_loco_class'].copy() 
		
		deb.prints(batch['in'].shape)
		deb.prints(batch['label'].shape)
		return batch
	def addLocoClass(self, test_label):
		print('*'*20, 'addLocoClass')
		deb.prints(np.unique(test_label,return_counts=True))
		deb.prints(np.unique(self.label_with_loco_class, return_counts=True))
		test_label[self.label_with_loco_class == self.loco_class + 1] = self.loco_class + 1
		deb.prints(np.unique(test_label,return_counts=True))
		print('*'*20, 'end addLocoClass')
		return test_label

	def loadPredictions(self,path_model,seq_date=None, model_dataset=None):
		print(1)
		test_predictions, test_label = super().loadPredictions(path_model, seq_date, model_dataset)
		print(2)

		test_label = self.addLocoClass(test_label)
		print(3)

		return test_predictions, test_label



class PredictionsLoaderModelNto1FixedSeqFixedLabelAdditionalTestClsses(PredictionsLoaderModelNto1FixedSeqFixedLabel):
	def newLabel2labelTranslate(self, label, filename, bcknd_flag=True):
		print("Entering newLabel2labelTranslate")
		label = label.astype(np.uint8)
		# bcknd to 0
		deb.prints(np.unique(label,return_counts=True))
		deb.prints(np.unique(label)[-1])
		if bcknd_flag == True:
			label[label==np.unique(label)[-2]] = 255 # this -1 will be different for each dataset
		deb.prints(np.unique(label,return_counts=True))
		label = label + 1
		
		deb.prints(np.unique(label,return_counts=True))

		# translate 
		f = open(filename, "rb")
		new_labels2labels = pickle.load(f)
		deb.prints(new_labels2labels)

		classes = np.unique(label)
		deb.prints(classes)
		translated_label = label.copy()
		for j in range(len(classes)):
			print(classes[j])
			
			if classes[j]<20:
				print("Translated",new_labels2labels[classes[j]])
				translated_label[label == classes[j]] = new_labels2labels[classes[j]]
			else:
				print("class is outside dict") 

		# bcknd to last
		##label = label - 1 # bcknd is 255
		##label[label==255] = np.unique(label)[-2]
		return translated_label 
	def loadPredictions(self,path_model,seq_date=None, model_dataset=None):
		print("============== loading model =============")
		model=load_model(path_model, compile=False)
		print("Model", model)
		print("Loading in data: ",self.path_test+'patches_in.npy')
		batch = {}
		dated_patches_name =True
		if dated_patches_name==False:

			batch['in']=np.load(self.path_test+'patches_in.npy',mmap_mode='r') # len is 21
	#		test_label=np.load(self.path_test+'patches_label.npy')
			batch['label']=np.load(self.path_test+'patches_label.npy') # may18
		else:
			batch['in']=np.load(self.path_test+'patches_in_fixed_'+seq_date+'.npy',mmap_mode='r') # len is 21
	#		test_label=np.load(self.path_test+'patches_label.npy')
			deb.prints(self.path_test+'patches_label_fixed_'+seq_date+'.npy')
			batch['label']=np.load(self.path_test+'patches_label_fixed_'+seq_date+'.npy') # may18			
		self.labeled_dates = 12


		deb.prints(batch['in'].shape)
		deb.prints(batch['label'].shape)

		one_hot_label=False
		if one_hot_label==True:
			batch['label'] = batch['label'].argmax(axis=-1)

		#pdb.set_trace()
		
#		self.mim = MIMVarLabel_PaddedSeq()
		self.mim = MIMFixed()

		data = {'labeled_dates': 12}
		data['labeled_dates'] = 12

		
		#batch = {'in': test_in, 'label': test_label}

		# add doty

		if self.dataset=='lm':
			ds=LEM('fixed', seq_date)
		elif self.dataset=='l2':
			ds=LEM2('fixed', seq_date)
		dataSource = SARSource()
		ds.addDataSource(dataSource)
	
		time_delta = ds.getTimeDelta(delta=True,format='days')
		ds.setDotyFlag(True)
		dotys, dotys_sin_cos = ds.getDayOfTheYear()
		ds.dotyReplicateSamples(sample_n = batch['label'].shape[0])

		prediction_dtype = np.float16
		
		model_t_len = 12
		batch['shape'] = (batch['in'].shape[0], model_t_len) + batch['in'].shape[2:]

		# model dataset is to get the correct last date from the model dataset

		if model_dataset=='lm':
			train_ds=LEM('fixed',seq_date)
		elif model_dataset=='l2':
			train_ds=LEM2('fixed', seq_date)
		train_ds.addDataSource(SARSource())
		# get model class n
		model_shape = model.layers[-1].output_shape
		model_class_n = model_shape[-1]
		deb.prints(model_shape)
		deb.prints(model_class_n)
#		test_predictions = np.zeros_like(batch['label'][...,:-1	], dtype = prediction_dtype)
		test_predictions = np.zeros((batch['label'].shape[0],)+model_shape[1:], 
			dtype = prediction_dtype)
		deb.prints(test_predictions.shape)

		#pdb.set_trace()

		input_ = self.mim.batchTrainPreprocess(batch, ds,  
					label_date_id = -1) # tstep is -12 to -1
		deb.prints(input_[1].shape)

		#pdb.set_trace()
		test_predictions=(model.predict(input_)).astype(prediction_dtype) 
		#pdb.set_trace()
		print(" shapes", test_predictions.shape, batch['label'].shape)
		
		print("batch['in'][0].shape, batch['label'].shape, test_predictions.shape",batch['in'][0].shape, batch['label'].shape, test_predictions.shape)
		print("Test predictions dtype",test_predictions.dtype)
		deb.prints(np.unique(test_predictions.argmax(axis=-1), return_counts=True))
		deb.prints(np.unique(batch['label'], return_counts=True))

		print(" shapes", test_predictions.shape, batch['label'].shape)

		test_predictions = test_predictions.argmax(axis=-1)
		#batch['label'] = batch['label'].argmax(axis=-1)
		print(" shapes", test_predictions.shape, batch['label'].shape)
		print( "uniques",np.unique(test_predictions, return_counts=True),np.unique(batch['label'], return_counts=True))
		
		translate_mode=True
		deb.prints(translate_mode)

		if translate_mode==True:
			translate_label_path = '../'
			test_predictions = self.newLabel2labelTranslate(test_predictions, 
					#translate_label_path + 'new_labels2labels_lm_20171209_S1.pkl',
					translate_label_path + 'new_labels2labels_'+model_dataset+'_'+train_ds.im_list[-1]+'.pkl',
					
					bcknd_flag=False)
						
			batch['label'] = self.newLabel2labelTranslate(batch['label'], 
					translate_label_path + 'new_labels2labels_l2_'+ds.im_list[-1]+'.pkl',
					bcknd_flag=True)
		print("End shapes", test_predictions.shape, batch['label'].shape)
		print(" shapes", test_predictions.shape, batch['label'].shape)
		print( "uniques",np.unique(test_predictions, return_counts=True),np.unique(batch['label'], return_counts=True))
		#pdb.set_trace()
		del batch['in']
		return test_predictions, batch['label']
class PredictionsLoaderModelNto1FixedSeqVarLabel(PredictionsLoaderModelNto1):
	def newLabel2labelTranslate(self, label, filename):
		label = label.astype(np.uint8)
		# bcknd to 0
		label[np.unique(label)[-1]] = 255 # this -1 will be different for each dataset
		label = label + 1

		# translate 
		f = open(filename, "rb")
		new_labels2labels = pickle.load(f)

		classes = np.unique(label)
		translated_label = label.copy()
		for j in range(len(classes)):
			translated_label[label == classes[j]] = new_labels2labels[classes[j]]

		# bcknd to last
		##label = label - 1 # bcknd is 255
		##label[label==255] = np.unique(label)[-2]
		return translated_label 
	def loadPredictions(self,path_model):
		print("============== loading model =============")
		model=load_model(path_model, compile=False)
		print("Model", model)
		print("Loading in data: ",self.path_test+'patches_in.npy')
		batch = {}
		batch['in']=np.load(self.path_test+'patches_in.npy',mmap_mode='r') # len is 21
#		test_label=np.load(self.path_test+'patches_label.npy')
		self.labeled_dates = 12
		batch['label']=np.load(self.path_test+'patches_label.npy')
		deb.prints(batch['label'].shape)
		#pdb.set_trace()
		batch['label']=batch['label'][:,-self.labeled_dates:] # may18
		deb.prints(batch['in'].shape)
		deb.prints(batch['label'].shape)
		#pdb.set_trace()
		
		self.mim = MIMVarLabel_PaddedSeq()
#		self.mim = MIMFixed()

		data = {'labeled_dates': 12}
		data['labeled_dates'] = 12

		
		#batch = {'in': test_in, 'label': test_label}

		# add doty

		if self.dataset=='lm':
			ds=LEM()
		elif self.dataset=='l2':
			ds=LEM2()
		dataSource = SARSource()
		ds.addDataSource(dataSource)
	
		time_delta = ds.getTimeDelta(delta=True,format='days')
		ds.setDotyFlag(False)
		dotys, dotys_sin_cos = ds.getDayOfTheYear()
		ds.dotyReplicateSamples(sample_n = batch['label'].shape[0])

		prediction_dtype = np.float16
		
		model_t_len = 12
		batch['shape'] = (batch['in'].shape[0], model_t_len) + batch['in'].shape[2:]

		# get model class n
		model_shape = model.layers[-1].output_shape
		model_class_n = model_shape[-1]
		deb.prints(model_shape)
		deb.prints(model_class_n)
		test_predictions = np.zeros_like(batch['label'][...,:-1	], dtype = prediction_dtype)
#		test_predictions = np.zeros((batch['label'].shape[:-1])+(model_class_n,), dtype = prediction_dtype)
		deb.prints(test_predictions.shape)

		#pdb.set_trace()
		for t_step in range(data['labeled_dates']): # 0 to 11
			###batch_val_label = batch['label'][:, t_step]
			#data.patches['test']['label'] = data.patches['test']['label'][:, label_id]
			##deb.prints(batch_val_label.shape)
			##deb.prints(t_step-data['labeled_dates'])
			#pdb.set_trace()
			deb.prints(ds.doty_flag)
			input_ = self.mim.batchTrainPreprocess(batch, ds,  
						label_date_id = t_step-data['labeled_dates']) # tstep is -12 to -1
			deb.prints(input_[0].shape)
			#pdb.set_trace()
			#deb.prints(data.patches['test']['label'].shape)
			
			
			test_predictions[:, t_step]=(model.predict(
				input_)).astype(prediction_dtype) 
			#pdb.set_trace()
		print(" shapes", test_predictions, batch['label'])
		
		print("batch['in'][0].shape, batch['label'].shape, test_predictions.shape",batch['in'][0].shape, batch['label'].shape, test_predictions.shape)
		print("Test predictions dtype",test_predictions.dtype)
		deb.prints(np.unique(test_predictions.argmax(axis=-1), return_counts=True))
		deb.prints(np.unique(batch['label'].argmax(axis=-1), return_counts=True))

		
		#test_predictions = test_predictions.argmax(axis=-1)
		#batch['label'] = batch['label'].argmax(axis=-1)
		print( "uniques",np.unique(test_predictions.argmax(axis=-1), return_counts=True),np.unique(batch['label'].argmax(axis=-1), return_counts=True))

		#test_predictions = self.newLabel2labelTranslate(test_predictions, 'new_labels2labels_lm_20171209_S1.pkl')
		#batch['label'] = self.newLabel2labelTranslate(batch['label'], 'new_labels2labels_l2_20191223_S1.pkl')
		print( "uniques",np.unique(test_predictions.argmax(axis=-1), return_counts=True),np.unique(batch['label'].argmax(axis=-1), return_counts=True))
		#pdb.set_trace()
		del batch['in']
		return test_predictions, batch['label']
