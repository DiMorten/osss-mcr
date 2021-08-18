from colorama import init
init()
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose
# from tensorflow.keras.callbacks import ModelCheckpoint , EarlyStopping
from tensorflow.keras.optimizers import Adam,Adagrad 
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow.keras as keras
import os
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

from generator import DataGenerator, DataGeneratorWithCoords, DataGeneratorWithCoordsRandom

import matplotlib.pyplot as plt
sys.path.append('../../../dataset/dataset/patches_extract_script/')
from dataSource import DataSource, SARSource, OpticalSource, Dataset, LEM, LEM2, CampoVerde, OpticalSourceWithClouds, Humidity
from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq
from parameters.parameters_reader import ParamsTrain

from icecream import ic
from monitor import Monitor, MonitorNPY, MonitorGenerator, MonitorNPYAndGenerator
import natsort


# ================= Dataset class implements data loading, patch extraction, metric calculation and image reconstruction =======#
class Dataset(object):

	def __init__(self, paramsTrain, ds, *args, **kwargs):

		print("Initializing object...")
		print(paramsTrain.t_len, paramsTrain.channel_n)
		self.paramsTrain = paramsTrain
		
		self.patch_len = self.paramsTrain.patch_len
		self.path = {"v": self.paramsTrain.path, 'train': {}, 'test': {}}
		self.image = {'train': {}, 'test': {}}
		self.patches = {'train': {}, 'test': {}}

		self.patches['train']['step']=self.paramsTrain.patch_step_train
		self.patches['test']['step']=self.paramsTrain.patch_step_test 
	  
		self.path['train']['in'] = self.paramsTrain.path / 'train_test/train/ims/'
		self.path['test']['in'] = self.paramsTrain.path / 'train_test/test/ims/'
		self.path['train']['label'] = self.paramsTrain.path / 'train_test/train/labels/'
		self.path['test']['label'] = self.paramsTrain.path / 'train_test/test/labels/'

		# in these paths, the augmented train set and validation set are stored
		# they can be loaded after (flag decides whether estimating these values and storing,
		# or loading the precomputed ones)
		self.path_patches_bckndfixed = self.paramsTrain.path / 'patches_bckndfixed/' 
		self.path['train_bckndfixed']=self.path_patches_bckndfixed/'train/'
		self.path['val_bckndfixed']=self.path_patches_bckndfixed/'val/'
		self.path['test_bckndfixed']=self.path_patches_bckndfixed/'test/'
		self.path['test_loco'] = self.path_patches_bckndfixed/'test_loco/'

		self.channel_n = self.paramsTrain.channel_n
		deb.prints(self.channel_n)
		self.debug = self.paramsTrain.debug
		self.class_n = self.paramsTrain.class_n
		self.reloadLabel(train=True)
		self.class_n = len(np.unique(self.full_label_train))
		ic(self.class_n)
		self.report={'best':{}, 'val':{}}
		self.report['exp_id']=self.paramsTrain.exp_id
		self.report['best']['text_name']='result_'+self.paramsTrain.exp_id+'.txt'
		self.report['best']['text_path']='../results/'+self.report['best']['text_name']
		self.report['best']['text_history_path']='../results/'+'history.txt'
		self.report['val']['history_path']='../results/'+'history_val.txt'
		
		self.t_len=self.paramsTrain.t_len
		deb.prints(self.t_len)
		self.dotys_sin_cos = self.paramsTrain.dotys_sin_cos
		self.dotys_sin_cos = np.expand_dims(self.dotys_sin_cos,axis=0) # add batch dimension
		self.dotys_sin_cos = np.repeat(self.dotys_sin_cos,16,axis=0)
		self.ds = ds

#		super().__init__(*args, **kwargs)
		self.im_gray_idx_to_rgb_table=[[0,[0,0,255],29],
									[1,[0,255,0],150],
									[2,[0,255,255],179],
									[3,[255,255,0],226],
									[4,[255,255,255],255]]
		if self.debug >= 1:
			print("Initializing Dataset instance")
		
		#should be in another object
		self.padded_dates = []


	def batch_label_to_one_hot(self,im):
		im_one_hot=np.zeros(im.shape+(self.class_n,))
		print(im_one_hot.shape)
		print(im.shape)
		for clss in range(0,self.class_n):
			im_one_hot[..., clss][im == clss] = 1
		return im_one_hot

	def folder_load(self,folder_path): #move to patches_handler
		paths=glob.glob(folder_path+'*.npy')
		paths = natsort.natsorted(paths)
		#ic(paths)
		#pdb.set_trace()
		files=[]
		deb.prints(len(paths))
		for path in paths:
			#print(path)
			files.append(np.load(path))
		return np.asarray(files),paths




	def addDoty(self, input_, bounds=None):
		if self.doty_flag==True:
			if bounds!=None:
				dotys_sin_cos = self.dotys_sin_cos[:,bounds[0]:bounds[1] if bounds[1]!=0 else None]
			else:
				dotys_sin_cos = self.dotys_sin_cos

			input_ = [input_, dotys_sin_cos]
		return input_	
	def addDotyPadded(self, input_, bounds=None, seq_len=12, sample_n = 16):
		if self.doty_flag==True:
			if bounds!=None:
#				deb.prints(bounds)
				dotys_sin_cos = self.dotys_sin_cos[:,bounds[0]:bounds[1] if bounds[1]!=0 else None]
#				deb.prints(self.dotys_sin_cos.shape)
#				deb.prints(dotys_sin_cos.shape)
			else:
				dotys_sin_cos = self.dotys_sin_cos.copy()
			dotys_sin_cos_padded = np.zeros((16, seq_len, 2))
			dotys_sin_cos_padded[:, -dotys_sin_cos.shape[1]:] = dotys_sin_cos
			input_ = [input_, dotys_sin_cos_padded]
		return input_	


#=============== METRICS CALCULATION ====================#
	def reloadLabel(self, train=False):
		self.full_label_test = np.load(self.path['v'] / 'full_ims' / 'full_label_test.npy').astype(np.uint8)
		self.full_label_test = self.full_label_test[-1]

		if train == True:
			self.full_label_train = np.load(self.path['v'] / 'full_ims' / 'full_label_train.npy').astype(np.uint8)
			self.full_label_train = self.full_label_train[-1]

class DatasetWithCoords(Dataset):

	def create_load(self):
#		super().create_load()
		ic(os.path.dirname(os.path.abspath(__file__)))
		ic(os.getcwd())
		##os.chdir(os.path.dirname(os.path.abspath(__file__)))
		self.patches['train']['coords'] = np.load(self.path['v'] / 'coords_train.npy').astype(np.int)
		self.patches['test']['coords'] = np.load(self.path['v'] / 'coords_test.npy').astype(np.int)
		ic(self.patches['train']['coords'].shape)
		#pdb.set_trace()
		self.full_ims_train = np.load(self.path['v'] / 'full_ims' / 'full_ims_train.npy')
		self.full_ims_test = np.load(self.path['v'] / 'full_ims' / 'full_ims_test.npy')
		
		self.full_label_train = np.load(self.path['v'] / 'full_ims' / 'full_label_train.npy').astype(np.uint8)
		self.full_label_test = np.load(self.path['v'] / 'full_ims' / 'full_label_test.npy').astype(np.uint8)


		self.labelPreprocess()

		self.patches['train']['n'] = self.patches['train']['coords'].shape[0]
		self.patches['train']['idx']=range(self.patches['train']['n'])

		self.patches['test']['label'] = self.getPatchesFromCoords(
			self.full_label_test, self.patches['test']['coords'])
		self.patches['train']['label'] = self.getPatchesFromCoords(
			self.full_label_train, self.patches['train']['coords'])

		self.patches['test']['in'] = self.getSequencePatchesFromCoords(
			self.full_ims_test, self.patches['test']['coords'])
		self.patches['train']['in'] = self.getSequencePatchesFromCoords(
			self.full_ims_train, self.patches['train']['coords'])

		ic(self.patches['train']['label'].shape)
		ic(np.unique(self.patches['train']['label'], return_counts = True))
		ic(self.patches['test']['label'].shape)
		ic(np.unique(self.patches['test']['label'], return_counts = True))
		

		unique,count=np.unique(self.full_label_train,return_counts=True)
		self.class_n=unique.shape[0]
		ic(np.unique(self.full_label_train,return_counts=True))
		ic(self.class_n)
		'''
		self.patches['train']['label'] = self.patches['train']['label']-1
		self.patches['test']['label'] = self.patches['test']['label']-1

		deb.prints(np.unique(self.patches['train']['label'], return_counts=True))
		
##            labels_val = labels_val-1
		class_n_no_bkcnd = len(self.classes)-1
		self.patches['train']['label'][self.patches['train']['label']==255] = class_n_no_bkcnd
		'''

	def knownClassesGet(self):
		# known classes criteria
		unique, count = np.unique(self.full_label_train, return_counts=True)
		ic(unique, count)
		
		unique = unique[1:] - 1
		count = count[1:]
		ic(unique, count)

		total_count = np.sum(count)
		ic(total_count)
		count_percentage = count / total_count
		ic(count_percentage)
		ic(sorted(zip(count_percentage, unique)))

		unique_sorted = sorted(zip(count_percentage, unique), reverse=True)
		unique_sorted = np.asarray(unique_sorted)
		ic(unique_sorted)
		if self.paramsTrain.openMode != 'NoMode':
			cum_percentage = 0.
			self.paramsTrain.known_classes = []
			ic(unique.shape[0])
			ic(self.paramsTrain.known_classes_percentage)
			for idx in range(unique.shape[0]):
				cum_percentage += unique_sorted[idx, 0]
				ic(idx, unique_sorted[idx], cum_percentage)
				if cum_percentage<self.paramsTrain.known_classes_percentage:
					self.paramsTrain.known_classes.append(int(unique_sorted[idx, 1]))
				else:
					break
	#				print("Unknown class")
	#			else:
	#				break
			self.paramsTrain.known_classes = sorted(self.paramsTrain.known_classes)
			ic(self.paramsTrain.known_classes)
		#pdb.set_trace()

#		self.known_classes = #
	def labelPreprocess(self, saveDicts = True):
		self.full_label_train = self.full_label_train[-1]
		self.full_label_test = self.full_label_test[-1]

		
		

		unique,count=np.unique(self.full_label_train,return_counts=True)
		self.class_n=unique.shape[0] # plus background

		ic(np.unique(self.full_label_train, return_counts=True))
		ic(np.unique(self.full_label_test, return_counts=True))

		if self.paramsTrain.openMode == 'NoMode':
			self.paramsTrain.known_classes = np.unique(self.full_label_train)[1:] - 1
			ic(self.paramsTrain.known_classes)
#		pdb.set_trace()
		
		#ic(self.unknown_classes)
		
		# save label with unknown class
		'''
		np.save(self.path['v'] + 
			'full_ims/label_with_unknown/full_label_train_with_unknown_' + 
			str(self.paramsTrain.seq_date) + '.npy', 
			self.full_label_train)
		np.save(self.path['v'] + 
			'full_ims/label_with_unknown/full_label_test_with_unknown_' + 
			str(self.paramsTrain.seq_date) + '.npy', 
			self.full_label_test)
		'''
		if self.paramsTrain.open_set==True:

			self.knownClassesGet()
			ic(self.paramsTrain.known_classes)
			if self.paramsTrain.select_kept_classes_flag==False:	
				self.unknown_classes = self.paramsTrain.unknown_classes
			else:
				all_classes = np.unique(self.full_label_train) # with background
				all_classes = all_classes[1:] - 1 # no bcknd
				deb.prints(all_classes)
				deb.prints(self.paramsTrain.known_classes)
				self.unknown_classes = np.setdiff1d(all_classes, self.paramsTrain.known_classes)
				deb.prints(self.unknown_classes)

			for clss in self.unknown_classes:
				self.full_label_train[self.full_label_train==int(clss) + 1] = 0
				self.full_label_test[self.full_label_test==int(clss) + 1] = 0

		elif self.paramsTrain.group_bcknd_classes == True:
			all_classes = np.unique(self.full_label_train) # with background
			all_classes = all_classes[1:] - 1 # no bcknd
			deb.prints(all_classes)
			deb.prints(self.paramsTrain.known_classes)
			self.unknown_classes = np.setdiff1d(all_classes, self.paramsTrain.known_classes)
			deb.prints(self.unknown_classes)
			for clss in self.unknown_classes:
				self.full_label_train[self.full_label_train==int(clss) + 1] = 20	
				self.full_label_test[self.full_label_test==int(clss) + 1] = 20	
		ic(np.unique(self.full_label_train, return_counts=True))
		self.classes = np.unique(self.full_label_train)

		tmp_tr = self.full_label_train.copy()
		tmp_tst = self.full_label_test.copy()

		ic(self.classes)
		deb.prints(np.unique(self.full_label_train, return_counts=True))
		self.labels2new_labels = dict((c, i) for i, c in enumerate(self.classes))
		self.new_labels2labels = dict((i, c) for i, c in enumerate(self.classes))
		ic(self.labels2new_labels, self.new_labels2labels)
		print("Transforming labels2new_labels...")
		for j in range(len(self.classes)):
			#ic(j, self.classes[j], self.labels2new_labels[self.classes[j]])
			self.full_label_train[tmp_tr == self.classes[j]] = self.labels2new_labels[self.classes[j]]
			self.full_label_test[tmp_tst == self.classes[j]] = self.labels2new_labels[self.classes[j]]
	
		print("Transformed labels2new_labels. Moving bcknd to last...")

		# save dicts
		if saveDicts == True:
			dict_filename = "new_labels2labels_"+self.ds.name+"_"+self.ds.im_list[-1]+".pkl" 
			deb.prints(dict_filename)
			f = open(dict_filename, "wb")
			pickle.dump(self.new_labels2labels, f)
			f.close()
		deb.prints(self.new_labels2labels)

		# bcknd to last class
		ic(np.unique(self.full_label_train, return_counts=True))

		unique = np.unique(self.full_label_train)
		self.full_label_train = self.full_label_train - 1
		self.full_label_test = self.full_label_test - 1
		self.full_label_train[self.full_label_train == 255] = unique[-1]
		self.full_label_test[self.full_label_test == 255] = unique[-1]

		print("Moved bcknd to last")
		ic(np.unique(self.full_label_train, return_counts=True))
		ic(np.unique(self.full_label_test, return_counts=True))
		unique,count=np.unique(self.full_label_train,return_counts=True)
		self.class_n=unique.shape[0] # plus background
		ic(self.class_n)

	def addPaddingToInput(self, model_t_len, im):
		if self.t_len < model_t_len: 
			seq_pad_len = model_t_len - self.t_len
			im = np.concatenate(
					(np.zeros((seq_pad_len, *im.shape[1:])),
					im),
					axis = 0)
		ic(im.shape)
		return im

	def addPaddingToInputPatches(self, patches, model_t_len):
		if self.t_len < model_t_len: 
			seq_pad_len = model_t_len - self.t_len
			patch_n = patches.shape[0]
			patches = np.concatenate(
					(np.zeros((patch_n, seq_pad_len, *patches.shape[2:])),
					patches),
					axis = 1)
		ic(patches.shape)
		return patches
	def val_set_get(self,mode='random',validation_split=0.2, idxs=None):

		self.patches['val']={'n':int(self.patches['train']['n']*validation_split)}
		ic(self.patches['train']['n'], self.patches['val']['n'])
		ic(self.patches['train']['coords'].shape)

		#===== CHOOSE VAL IDX
		if mode=='random':
			self.patches['val']['idx']=np.random.choice(self.patches['train']['idx'],self.patches['val']['n'],replace=False)
			self.patches['val']['coords'] =  self.patches['train']['coords'][self.patches['val']['idx']]
		
		self.patches['train']['coords']=np.delete(self.patches['train']['coords'],self.patches['val']['idx'],axis=0)

		ic(self.patches['train']['coords'].shape)
		ic(self.patches['val']['coords'].shape)
		
	def semantic_balance(self,samples_per_class=500,label_type='Nto1'): # samples mean sequence of patches. Keep
		print("data.semantic_balance")
		
		# Count test
		patch_count=np.zeros(self.class_n)
		if label_type == 'NtoN':
			patch_count_axis = (1,2,3)
			rotation_axis = (2,3)
		elif label_type == 'Nto1':	
			patch_count_axis = (1,2)
			rotation_axis = (1,2)

		
		# Count train
		patch_count=np.zeros(self.class_n)


		
		# Start balancing
		balance={}
		balance["out_n"]=(self.class_n-1)*samples_per_class


		balance["coords"] = np.zeros((balance["out_n"], *self.patches["train"]["coords"].shape[1:])).astype(np.int)
		ic(balance["coords"].shape) 
		k=0

		# get patch count from coords only
		ic(np.unique(self.full_label_train, return_counts = True))
		
#		pdb.set_trace()

		# get patch class
		coords_classes = np.zeros((self.patches['train']['coords'].shape[0], self.class_n))
		ic(coords_classes.shape)
		unique_train = np.unique(self.full_label_train)
		ic(unique_train)
		bcknd_idx = unique_train[-1]
		ic(bcknd_idx)
		psize = self.paramsTrain.patch_len # 32
		ic(psize)


		for idx in range(self.patches['train']['coords'].shape[0]):
			coord = self.patches['train']['coords'][idx]
			label_patch = self.full_label_train[coord[0]:coord[0]+psize,
				coord[1]:coord[1]+psize]
			patchClassCount = Counter(label_patch[label_patch<bcknd_idx]) # es mayor a 0? o el bcknd esta al final?
			for key in patchClassCount:
				patch_count[key] = patch_count[key] + 1
				coords_classes[idx, key] = 1
		
#		getClassCountFromCoords()
		ic(patch_count)
		for clss in range(0,self.class_n-1):
#			ic(clss)
#			ic(patch_count[clss])
#			patch_count[clss] = np.count_nonzero(np.where(classes == clss))

			ic(patch_count[clss])
			#pdb.set_trace()
			if patch_count[clss]==0:
				continue
			ic(clss)

			idxs = coords_classes[:, clss] == 1
			ic(idxs.shape,idxs.dtype)
			ic(np.unique(idxs, return_counts = True))
			#pdb.set_trace()

			balance["class_coords"]=self.patches['train']['coords'][idxs]

			ic(balance["class_coords"].shape)
			ic(samples_per_class)
			deb.prints(clss)
			if balance["class_coords"].shape[0]>samples_per_class:
				replace=False
				index=range(balance["class_coords"].shape[0])
				index = np.random.choice(index, samples_per_class, replace=replace)

				balance["coords"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["class_coords"][index]

			else:

				augmented_manipulations=False
				if augmented_manipulations==True:

					augmented_coords = balance["class_coords"]

					for i in range(int(samples_per_class/balance["label"].shape[0] - 1)):                
						augmented_coords = np.vstack((augmented_coords, balance['class_coords']))
						
		#            augmented_labels_temp = np.tile(clss_labels,samples_per_class/num_samples )

#					index = range(augmented_data.shape[0])
					index = range(augmented_coords.shape[0])

					ic(augmented_coords.shape)
					index = np.random.choice(index, samples_per_class, replace=True)
					ic(index.shape)
					balance["coords"][k*samples_per_class:k*samples_per_class + samples_per_class] = augmented_coords[index]

				else:
					replace=True
#					index = range(balance["label"].shape[0])
					index = range(balance["class_coords"].shape[0])

					index = np.random.choice(index, samples_per_class, replace=replace)
					balance["coords"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["class_coords"][index]

			k+=1
		
		idx = np.random.permutation(balance["coords"].shape[0])

		self.patches['train']['coords'] = balance["coords"][idx]
		print("Balanced train unique (coords):")
		deb.prints(self.patches['train']['coords'].shape)
##		deb.prints(np.unique(self.patches['train']['label'].argmax(axis=-1),return_counts=True))
	def getPatchesFromCoords(self, im, coords):
		# coords is n, row, col
		patch_size = self.patch_len
		# patch_size = 32
		patch_dim = (patch_size, patch_size)
		Y = np.empty((coords.shape[0], *patch_dim), dtype=int)
		for idx in range(coords.shape[0]):
			patch = im[coords[idx][0]:coords[idx][0]+patch_size,
						coords[idx][1]:coords[idx][1]+patch_size]
			Y[idx] = patch
		return Y
	def getSequencePatchesFromCoords(self, ims, coords):
		patch_size = self.patch_len #32
		# t_len = 12
		# channel_n = 2
		patch_dim = (ims.shape[0], patch_size, patch_size, ims.shape[-1])
		Y = np.empty((coords.shape[0], *patch_dim), dtype=np.float16)
		for idx in range(coords.shape[0]):
			patch = ims[:, coords[idx][0]:coords[idx][0]+patch_size,
						coords[idx][1]:coords[idx][1]+patch_size]
			Y[idx] = patch
		return Y
	def comparePatchesCoords(self):
		Y = self.getPatchesFromCoords(self.full_label_train, self.patches['train']['coords'][0:16])
		ic(np.unique(self.patches['train']['label'].argmax(axis=-1)[0:16], return_counts=True))
		ic(np.unique(Y, return_counts=True))
		pdb.set_trace()

	def loadMask(self):
		ic(str(self.paramsTrain.path))
		self.mask=cv2.imread(str(self.paramsTrain.path / 'TrainTestMask.tif'),-1)
		ic(self.mask.shape)
		#return mask


	def setDateList(self, paramsTrain):
		lm_labeled_dates = ['20170612', '20170706', '20170811', '20170916', '20171010', '20171115', 
							'20171209', '20180114', '20180219', '20180315', '20180420', '20180514']
		l2_labeled_dates = ['20191012','20191117','20191223','20200116','20200221','20200316',
							'20200421','20200515','20200620','20200714','20200819','20200912']
		cv_labeled_dates = ['20151029', '20151110', '20151122', '20151204', '20151216', '20160121', 
							'20160214', '20160309', '20160321', '20160508', '20160520', '20160613', 
							'20160707', '20160731']
		if paramsTrain.dataset == 'lm':
			if paramsTrain.seq_date=='jan':
				self.dataset_date = lm_labeled_dates[7]
				

			elif paramsTrain.seq_date=='feb':
				self.dataset_date = lm_labeled_dates[8]
				

			elif paramsTrain.seq_date=='mar':
				self.dataset_date = lm_labeled_dates[9]
				

			elif paramsTrain.seq_date=='apr':
				self.dataset_date = lm_labeled_dates[10]
				

			elif paramsTrain.seq_date=='may':
				self.dataset_date = lm_labeled_dates[11]
				

			elif paramsTrain.seq_date=='jun':
				self.dataset_date = lm_labeled_dates[0]
				

			elif paramsTrain.seq_date=='jul':
				self.dataset_date = lm_labeled_dates[1]
				

			elif paramsTrain.seq_date=='aug':
				self.dataset_date = lm_labeled_dates[2]
				

			elif paramsTrain.seq_date=='sep':
				self.dataset_date = lm_labeled_dates[3]
				

			elif paramsTrain.seq_date=='oct':
				self.dataset_date = lm_labeled_dates[4]
				

			elif paramsTrain.seq_date=='nov':
				self.dataset_date = lm_labeled_dates[5]
				

			if paramsTrain.seq_date=='dec':
			#dec
				self.dataset_date = lm_labeled_dates[6]
				l2_date = l2_labeled_dates[2]
		elif paramsTrain.dataset == 'cv':
			if paramsTrain.seq_date=='jun':
				self.dataset_date = cv_labeled_dates[11]


	def newLabel2labelTranslate(self, label, filename, bcknd_flag=False, debug = 1):

		if debug == 1:
			print("Entering newLabel2labelTranslate")
		label = label.astype(np.uint8)
		# bcknd to 0
		if debug == 1:
			deb.prints(np.unique(label,return_counts=True))
			deb.prints(np.unique(label)[-1])
		if bcknd_flag == True:
			label[label==np.unique(label)[-1]] = 255 # this -1 will be different for each dataset
		if debug == 1:
			deb.prints(np.unique(label,return_counts=True))
		label = label + 1
		if debug == 1:
			deb.prints(np.unique(label,return_counts=True))

		# translate 
		f = open(filename, "rb")
		new_labels2labels = pickle.load(f)
		if debug == 1:
			print("new_labels2labels filename",f)
			deb.prints(new_labels2labels)

		classes = np.unique(label)
		if debug == 1:
			deb.prints(classes)
		translated_label = label.copy()
		for j in range(len(classes)):
			if debug == 1:
				print(classes[j])
			
			try:
				if debug == 1:
					print("Translated",new_labels2labels[classes[j]])
				translated_label[label == classes[j]] = new_labels2labels[classes[j]]
			except:
				if debug == 1:
					print("Translation of class {} failed. Not in dictionary. Continuing anyway...", classes[j])

		# bcknd to last
		##label = label - 1 # bcknd is 255
		##label[label==255] = np.unique(label)[-2]
		return translated_label 

	def small_classes_ignore(self, label, predictions, important_classes_idx):
		class_n = 15
		important_classes_idx = list(important_classes_idx)
		ic(important_classes_idx)
		important_classes_idx.append(255) # bcknd
		for idx in range(class_n):
			if idx not in important_classes_idx:
				predictions[predictions==idx]=20
				label[label==idx]=20	
		important_classes_idx = important_classes_idx[:-1]
		deb.prints(important_classes_idx)

		deb.prints(np.unique(label,return_counts=True))
		deb.prints(np.unique(predictions,return_counts=True))

		return label, predictions, important_classes_idx
