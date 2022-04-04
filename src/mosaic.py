 
import numpy as np
import cv2
import glob
import argparse
import pdb
import sys, os
#sys.path.append('../../../../../train_src/analysis/')
import pathlib
import pdb

from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq
#sys.path.append('../../../../../dataset/dataset/patches_extract_script/')
#from dataSource import DataSource, SARSource, Dataset, LEM, LEM2, CampoVerde
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report,recall_score,precision_score
import colorama
colorama.init()
import pickle
import deb
import time
#sys.path.append('../../../train_src/analysis')
#print(sys.path)
##sys.path.append('../results/convlstm_results/reconstruct')
##from PredictionsLoader import PredictionsLoaderNPY, PredictionsLoaderModel, PredictionsLoaderModelNto1FixedSeqFixedLabelOpenSet
from icecream import ic 
from parameters.params_train import ParamsTrain, ParamsAnalysis
from open_set import SoftmaxThresholding, OpenPCS
from parameters.params_mosaic import ParamsReconstruct
from parameters.params_batchprocessing import ParamsBatchProcessing
from postprocessing import OpenSetManager
ic.configureOutput(includeContext=False, prefix='[@debug] ')

from model_input_mode import MIMVarLabel_PaddedSeq, MIMFixed_PaddedSeq


class Mosaic():
	def __init__(self, paramsTrain, paramsMosaic):

		self.paramsTrain = paramsTrain
		self.paramsMosaic = paramsMosaic
#		self.paramsMosaic = ParamsReconstruct(paramsTrain)
		self.pb = ParamsBatchProcessing(paramsTrain, self.paramsMosaic)


		ic(paramsTrain.seq_date)
		dataset=paramsTrain.dataset


		deb.prints(dataset)
		deb.prints(paramsTrain.model_type)
#		deb.prints(direct_execution)

		self.debug = -2


	def loopOverImage(self, paramsTrain):

		for m in range(paramsTrain.patch_len//2,self.h-paramsTrain.patch_len//2,self.stride): 
			for n in range(paramsTrain.patch_len//2,self.w-paramsTrain.patch_len//2,self.stride):

				patch_mask = self.mask_pad[m-paramsTrain.patch_len//2:m+paramsTrain.patch_len//2 + paramsTrain.patch_len%2,
						n-paramsTrain.patch_len//2:n+paramsTrain.patch_len//2 + paramsTrain.patch_len%2]

				if self.paramsMosaic.conditionType == 'test':
					condition = np.any(patch_mask==2)
				else:
					condition = True	
				if condition:		

					patch = self.data.full_ims_test[:, 
								m-paramsTrain.patch_len//2:m+paramsTrain.patch_len//2 + paramsTrain.patch_len%2,
								n-paramsTrain.patch_len//2:n+paramsTrain.patch_len//2 + paramsTrain.patch_len%2]
					patch = np.expand_dims(patch, axis = 0)
					pred_logits = self.modelManager.model.predict(patch)
					pred = pred_logits.argmax(axis=-1).astype(np.uint8)
					_, x, y, c = pred_logits.shape

					#ic(pred_logits.shape, pred.shape)
					#ic(self.prediction_logits_mosaic.shape, self.prediction_mosaic.shape)
					
						
					self.prediction_logits_mosaic[m-self.stride//2:m+self.stride//2,n-self.stride//2:n+self.stride//2] = pred_logits[0,self.overlap//2:x-self.overlap//2,self.overlap//2:y-self.overlap//2]
					self.prediction_mosaic[m-self.stride//2:m+self.stride//2,n-self.stride//2:n+self.stride//2] = pred[0,self.overlap//2:x-self.overlap//2,self.overlap//2:y-self.overlap//2]

		
		if self.paramsMosaic.open_set_mode == False:
			self.prediction_mosaic = self.prediction_logits_mosaic.argmax(axis=-1).astype(np.uint8)

		if self.paramsMosaic.add_padding_flag==True:
			self.prediction_mosaic=self.prediction_mosaic[self.overlap//2:-step_row,self.overlap//2:-step_col]
			self.scores_mosaic=self.scores_mosaic[self.overlap//2:-step_row,self.overlap//2:-step_col]
		ic(self.prediction_mosaic.shape, self.mask_pad.shape, self.data.full_label_test.shape)
		ic(np.unique(self.data.full_label_test, return_counts=True))
		ic(np.unique(self.prediction_mosaic, return_counts=True))
		self.paramsMosaic.spatial_results_path.mkdir(parents=True, exist_ok=True)
		np.save(self.paramsMosaic.spatial_results_path / 
			('prediction_mosaic_'+self.data.dataset_date+'_'+self.name_id+'_overl'+str(self.paramsMosaic.overlap)+'.npy'),self.prediction_mosaic)
		if self.paramsMosaic.open_set_mode == True:
			np.save(self.paramsMosaic.spatial_results_path / 
				('scores_mosaic_'+self.data.dataset_date+'_'+self.name_id+'_overl'+str(self.paramsMosaic.overlap)+'.npy'),self.scores_mosaic)

	def loadMosaic(self):
		self.prediction_mosaic = np.load(self.paramsMosaic.spatial_results_path / 
			('prediction_mosaic_'+self.data.dataset_date+'_'+self.name_id+'_overl'+str(self.paramsMosaic.overlap)+'.npy'))
		if self.paramsMosaic.open_set_mode == True:
			self.openSetManager.scores_mosaic = np.load(self.paramsMosaic.spatial_results_path / 
				('scores_mosaic_'+self.data.dataset_date+'_'+self.name_id+'_overl'+str(self.paramsMosaic.overlap)+'.npy'))

	def defineMosaicVars(self, h, w, class_n):
		self.prediction_mosaic=np.ones((h,w)).astype(np.uint8)*255
		self.scores_mosaic=np.zeros((h,w)).astype(np.float16)
		self.prediction_logits_mosaic=np.ones((h,w, class_n)).astype(np.float16)
		self.h = h
		self.w = w
	def infer(self, paramsTrain, modelManager, data, ds, postProcessing = None):
		self.modelManager = modelManager
		self.data = data
		self.ds = ds
		self.mim = MIMFixed_PaddedSeq()
		self.ds.setDotyFlag(False)

		if self.paramsMosaic.add_padding_flag==True:
			self.data.full_ims_test, self.stride, step_row, step_col, self.overlap = seq_add_padding(
				self.data.full_ims_test, paramsTrain.patch_len, self.paramsMosaic.overlap)
			self.mask_pad, _, _, _, _ = add_padding(self.data.mask,paramsTrain.patch_len,self.paramsMosaic.overlap)
		else:
			self.mask_pad=self.data.mask.copy()
			self.stride=paramsTrain.patch_len
			self.overlap=0
			

		t_len, h, w, _ = self.data.full_ims_test.shape
		ic(self.data.class_n)
		ic(np.unique(self.data.full_label_test), len(np.unique(self.data.full_label_test)))
		ic(np.unique(self.data.full_label_train), len(np.unique(self.data.full_label_train)))
		class_n = len(np.unique(self.data.full_label_train)) - 1
		ic(class_n)
#		pdb.set_trace()
		self.defineMosaicVars(h, w, class_n)
		
		t0 = time.time()

		self.data.setDateList(paramsTrain)
		self.name_id = 'closed_set'

		if self.paramsMosaic.mosaic_flag == True:
			self.loopOverImage(paramsTrain)
		else:
			self.loadMosaic()
		self.postProcess(paramsTrain)
			
		ic(np.unique(self.prediction_mosaic, return_counts=True))

		ic(time.time()-t0)


		ic(self.data.full_label_test.shape)

		
		cv2.imwrite('sample.png', self.prediction_logits_mosaic.argmax(axis=-1).astype(np.uint8)*10)
		cv2.imwrite('label_sample.png', self.data.full_label_test.astype(np.uint8)*10)

		ic(np.unique(self.prediction_mosaic, return_counts=True))




		cv2.imwrite('sample_translate.png', self.prediction_mosaic.astype(np.uint8)*10)

		ic(self.data.full_label_test.shape)
		ic(np.unique(self.data.full_label_test, return_counts=True))
		self.label_mosaic = self.data.full_label_test
		ic(np.unique(self.label_mosaic, return_counts=True))
		ic(np.unique(self.prediction_mosaic, return_counts=True))

		# bcknd to 255
		self.label_mosaic = self.label_mosaic - 1
		self.prediction_mosaic = self.prediction_mosaic - 1

		ic(np.unique(self.label_mosaic, return_counts=True))
		ic(np.unique(self.prediction_mosaic, return_counts=True))
#		pdb.set_trace()
		important_classes_idx = paramsTrain.known_classes


		self.label_mosaic, self.prediction_mosaic, important_classes_idx = self.data.small_classes_ignore(
					self.label_mosaic, self.prediction_mosaic,important_classes_idx)

		self.prediction_mosaic[self.prediction_mosaic==39] = 20
		self.label_mosaic[self.label_mosaic==39] = 20


		deb.prints(np.unique(self.label_mosaic,return_counts=True))
		deb.prints(np.unique(self.prediction_mosaic,return_counts=True))
		deb.prints(self.label_mosaic.shape)
		deb.prints(self.prediction_mosaic.shape)
		deb.prints(important_classes_idx)

		ic(self.data.mask.shape, self.mask_pad.shape, self.label_mosaic.shape, self.prediction_mosaic.shape)
		self.save_prediction_label_mosaic_Nto1(self.data.mask, 
				self.paramsMosaic.custom_colormap, self.paramsMosaic.spatial_results_path, paramsTrain, 
				small_classes_ignore=True)


	def postProcess(self, paramsTrain):
		self.prediction_mosaic = self.data.newLabel2labelTranslate(self.prediction_mosaic, 
					'results/label_translations/new_labels2labels_'+paramsTrain.dataset+'_'+self.data.dataset_date+'_S1.pkl')

	def getPostProcessingScores(self):
		pass

	def save_prediction_label_mosaic_Nto1(self, mask, 
			custom_colormap, path, paramsTrain, small_classes_ignore=True):
	#	for t_step in range(t_len):

		ic(np.unique(mask, return_counts=True))

		self.label_mosaic[mask!=2]=255
		
		if self.paramsMosaic.prediction_mask == True:
			self.prediction_mosaic[mask!=2]=255	
		deb.prints(np.unique(self.label_mosaic,return_counts=True))
		deb.prints(np.unique(self.prediction_mosaic,return_counts=True))


		print("everything outside mask is 255")
		ic(np.unique(self.label_mosaic,return_counts=True))
		ic(np.unique(self.prediction_mosaic,return_counts=True))

		# Paint it!

		print(custom_colormap.shape)
		#class_n=custom_colormap.shape[0]
		#=== change to rgb
		print("Gray",self.prediction_mosaic.dtype)
		prediction_rgb=np.zeros((self.prediction_mosaic.shape+(3,))).astype(np.uint8)
		label_rgb=np.zeros_like(prediction_rgb)
		print("Adding color...")


		prediction_rgb=cv2.cvtColor(self.prediction_mosaic,cv2.COLOR_GRAY2RGB)
		label_rgb=cv2.cvtColor(self.label_mosaic,cv2.COLOR_GRAY2RGB)

		print("RGB",prediction_rgb.dtype,prediction_rgb.shape)

	#	for chan in [0,1,2]:
	#		prediction_rgb[...,chan][prediction_rgb[...,chan]==255]=custom_colormap[idx,chan]
	#		label_rgb[...,chan][label_rgb[...,chan]==255]=custom_colormap[idx,chan]


		deb.prints(custom_colormap)
		prediction_rgb_tmp = prediction_rgb.copy()
		label_rgb_tmp = label_rgb.copy()

		print("Assigning color...")

		for idx in range(custom_colormap.shape[0]):

			for chan in [0,1,2]:
				#deb.prints(np.unique(label_rgb[...,chan],return_counts=True))

				prediction_rgb[...,chan][prediction_rgb_tmp[...,chan]==idx]=custom_colormap[idx,chan]
				label_rgb[...,chan][label_rgb_tmp[...,chan]==idx]=custom_colormap[idx,chan]

		# color the unknown
		red_rgb = [255, 0, 0]
		for chan in [0,1,2]:
			prediction_rgb[...,chan][prediction_rgb[...,chan]==20]=red_rgb[chan]
			label_rgb[...,chan][label_rgb[...,chan]==20]=red_rgb[chan]

		print("RGB",prediction_rgb.dtype,prediction_rgb.shape)

		print("Saving the resulting images...")

		label_rgb=cv2.cvtColor(label_rgb,cv2.COLOR_BGR2RGB)
		prediction_rgb=cv2.cvtColor(prediction_rgb,cv2.COLOR_BGR2RGB)
		save_folder=path / paramsTrain.dataset / str(paramsTrain.model_type) / paramsTrain.seq_date
		save_folder.mkdir(parents=True, exist_ok=True)
		deb.prints(save_folder)

		if self.paramsMosaic.open_set_mode == True:
			tpr_threshold_names = ['0_1', '0_3', '0_5', '0_7', '0_9']
			threshIdxName = "_TPR" + tpr_threshold_names[self.paramsMosaic.threshold_idx]
			prediction_savename = save_folder / ("prediction_t_" + paramsTrain.seq_date + "_" + str(paramsTrain.model_type) +
				"_" + self.name_id+threshIdxName + "_overl" + str(self.paramsMosaic.overlap) + "_" + self.paramsMosaic.conditionType + ".png")

		else:
			prediction_savename = save_folder / ("prediction_t_" + paramsTrain.seq_date + "_" + str(paramsTrain.model_type) +
				"_closedset_" + self.name_id + "_overl" + str(self.paramsMosaic.overlap) + "_" + self.paramsMosaic.conditionType + ".png")

		ic(prediction_savename)
		print("saving...")
		try:

			os.remove(prediction_savename)
		except:
			print("no file to remove")
		ret = cv2.imwrite(str(prediction_savename), prediction_rgb)

		deb.prints(ret)
		ic(save_folder / ("label_t_"+paramsTrain.seq_date+"_"+str(paramsTrain.model_type)+"_"+self.name_id+".png"))
		ret = cv2.imwrite(str(save_folder / ("label_t_"+paramsTrain.seq_date+"_"+str(paramsTrain.model_type)+"_"+self.name_id+".png")),label_rgb)
		deb.prints(ret)
		ret = cv2.imwrite(str(save_folder / "mask.png"),mask*200)
		deb.prints(ret)

class MosaicHighRAM(Mosaic):



	def loopOverImage(self, paramsTrain):
		class_n = len(paramsTrain.known_classes)

		print("stride", self.stride)
		print(len(range(paramsTrain.patch_len//2,self.h-paramsTrain.patch_len//2,self.stride)))
		print(len(range(paramsTrain.patch_len//2,self.w-paramsTrain.patch_len//2,self.stride)))

		self.count = 0
		self.count_mask = 0

		self.loopToCountValidPatches()
		self.getPatchesPerBatch()
		
		self.count_mask = 0
		self.count_mask_overall = 0
		self.count_mask_batch = 0

		for self.batch in range(self.pb.batch_processing_n):
			print("================== starting self.batch ... ==================")

			self.count_mask_overall += self.count_mask_batch
			ic(self.count_mask_overall)
			ic(self.batch)
			ic(self.batch * self.patches_per_batch)
			ic((self.batch + 1) * self.patches_per_batch)
			self.count_mask_batch = 0
			self.count_mask = 0

			patches_in = self.loopToGetInputPatchesInBatch()

			ic(np.average(self.data.full_ims_test), np.min(self.data.full_ims_test), np.max(self.data.full_ims_test))
			ic(np.average(patches_in), np.min(patches_in), np.max(patches_in))
			ic(patches_in.shape)

			ic(self.count_mask_batch)
			ic(self.count_mask)
			#pdb.set_trace()


			self.pred_logits_patches = self.modelManager.model.predict(patches_in).astype(self.paramsMosaic.prediction_dtype)

			# T = 8.6
			# self.pred_logits_patches = self.pred_logits_patches / T
			
			ic(self.pred_logits_patches.dtype)
			ic(self.pred_logits_patches.shape)	

			ic(np.unique(self.pred_logits_patches.argmax(axis=-1), return_counts = True))
			##pdb.set_trace()
	#		self.openSetManager = PostProcessing()
	#		self.openSetManager.openSetActivate()

	#		pred_proba_patches = self.openSetManager.predict(self.modelManager, patches_in)

			#self.openSetManager.load_decoder_features(self.modelManager, patches_in, )
			if self.paramsMosaic.open_set_mode == True:
				self.pred_proba_patches = self.openSetManager.load_intermediate_features(
					self.modelManager, patches_in, self.pred_logits_patches, debug = 0)	
			
				self.pred_proba_patches = self.pred_proba_patches.astype(self.paramsMosaic.prediction_dtype)

				ic(self.pred_proba_patches.dtype, self.pred_proba_patches.shape)
			ic(self.count_mask)
			self.count_mask = 0
			self.count_mask_batch = 0
			
			self.loopToMosaic()

			ic(np.unique(self.prediction_mosaic, return_counts=True))
			##pdb.set_trace()
			if self.paramsMosaic.open_set_mode == False:
				self.prediction_mosaic = self.prediction_logits_mosaic.argmax(axis=-1).astype(np.uint8)
				self.prediction_mosaic[self.mask_pad != 2] = 255
			ic(np.unique(self.prediction_mosaic, return_counts=True))

			if self.paramsMosaic.add_padding_flag==True:
				ic(prediction_mosaic.shape)
				ic(self.overlap)
				ic(step_row)
				self.prediction_mosaic=self.prediction_mosaic[self.overlap//2:-step_row,self.overlap//2:-step_col]
				self.openSetManager.scores_mosaic=self.openSetManager.scores_mosaic[self.overlap//2:-step_row,self.overlap//2:-step_col]
			pathlib.Path('results/spatial_results').mkdir(parents=True, exist_ok=True)

			np.save('results/spatial_results/prediction_mosaic_'+self.data.dataset_date+'_'+self.name_id+'_overl'+str(self.paramsMosaic.overlap)+'.npy',self.prediction_mosaic)
			if self.paramsMosaic.open_set_mode == True:
				np.save('results/spatial_results/scores_mosaic_'+self.data.dataset_date+'_'+self.name_id+'_overl'+str(self.paramsMosaic.overlap)+'.npy',self.openSetManager.scores_mosaic)


	def loopToGetInputPatchesInBatch(self):
		patches_in = []
			
		for m in range(self.paramsTrain.patch_len//2,self.h-self.paramsTrain.patch_len//2,self.stride): 
			for n in range(self.paramsTrain.patch_len//2,self.w-self.paramsTrain.patch_len//2,self.stride):
				
				patch_mask = self.mask_pad[m-self.paramsTrain.patch_len//2:m+self.paramsTrain.patch_len//2 + self.paramsTrain.patch_len%2,
							n-self.paramsTrain.patch_len//2:n+self.paramsTrain.patch_len//2 + self.paramsTrain.patch_len%2]

				if self.paramsMosaic.conditionType == 'test':
					condition_masking = np.any(patch_mask==2)
				else:
					condition_masking = True
						
				condition = condition_masking and self.count_mask >= self.batch * self.patches_per_batch and self.count_mask < (self.batch + 1) * self.patches_per_batch
				if condition:

					patch = {}			
					patch['in'] = self.data.full_ims_test[:,m-self.paramsTrain.patch_len//2:m+self.paramsTrain.patch_len//2 + self.paramsTrain.patch_len%2,
								n-self.paramsTrain.patch_len//2:n+self.paramsTrain.patch_len//2 + self.paramsTrain.patch_len%2]

					patch['in'] = np.expand_dims(patch['in'], axis = 0)

					patch['shape'] = (patch['in'].shape[0], self.paramsTrain.seq_len) + patch['in'].shape[2:]


					input_ = self.mim.batchTrainPreprocess(patch, self.ds,  
								label_date_id = -1) # tstep is -12 to -1

					patches_in.append(input_)
					self.count_mask_batch += 1
				if condition_masking:
					self.count_mask += 1
		patches_in = np.concatenate(patches_in, axis=0)
		return patches_in

	def loopToCountValidPatches(self):

		for m in range(self.paramsTrain.patch_len//2,self.h-self.paramsTrain.patch_len//2,self.stride): 
			for n in range(self.paramsTrain.patch_len//2,self.w-self.paramsTrain.patch_len//2,self.stride):
				patch_mask = self.mask_pad[m-self.paramsTrain.patch_len//2:m+self.paramsTrain.patch_len//2 + self.paramsTrain.patch_len%2,
							n-self.paramsTrain.patch_len//2:n+self.paramsTrain.patch_len//2 + self.paramsTrain.patch_len%2]
				if self.paramsMosaic.conditionType == 'test':
					condition = np.any(patch_mask==2)
				else:
					condition = True			
				if condition:
					self.count_mask += 1

	def loopToMosaic(self):
		
		for m in range(self.paramsTrain.patch_len//2,self.h-self.paramsTrain.patch_len//2,self.stride): 
			for n in range(self.paramsTrain.patch_len//2,self.w-self.paramsTrain.patch_len//2,self.stride):

				patch_mask = self.mask_pad[m-self.paramsTrain.patch_len//2:m+self.paramsTrain.patch_len//2 + self.paramsTrain.patch_len%2,
						n-self.paramsTrain.patch_len//2:n+self.paramsTrain.patch_len//2 + self.paramsTrain.patch_len%2]

				if self.paramsMosaic.conditionType == 'test':
					condition_masking = np.any(patch_mask==2)
				else:
					condition_masking = True	
				condition = condition_masking and self.count_mask >= self.batch * self.patches_per_batch and self.count_mask < (self.batch + 1) * self.patches_per_batch
				if condition:

	##				t0=time.time()
					#pred_logits = np.squeeze(model.predict(input_))
					pred_logits = np.squeeze(self.pred_logits_patches[self.count_mask_batch])
					pred_cl = pred_logits.argmax(axis=-1)

					if self.paramsMosaic.open_set_mode == True: # do in postProcessing
						self.test_pred_proba = np.squeeze(self.pred_proba_patches[self.count_mask_batch])


					x, y = pred_cl.shape
					prediction_shape = pred_cl.shape

					if self.paramsMosaic.open_set_mode == True:
						if self.debug>-1: # do in postProcessing
							
							print('*'*20, "Starting openModel predict")
							ic(pred_cl.shape)
							ic(self.test_pred_proba.shape)

							ic(np.min(self.test_pred_proba), np.average(self.test_pred_proba), np.median(self.test_pred_proba), np.max(self.test_pred_proba))

						# translate the preddictions.

						pred_cl = self.data.newLabel2labelTranslate(pred_cl, 
								'results/label_translations/new_labels2labels_'+self.paramsTrain.dataset+'_'+self.data.dataset_date+'_S1.pkl',
								debug = self.debug)

						if self.debug>0:
							ic(pred_cl.shape)

						self.openSetManager.predictPatch(pred_cl, self.test_pred_proba, m, n, self.stride, self.overlap, debug = self.debug)
						

					if self.paramsMosaic.overlap_mode == 'replace':
						self.prediction_mosaic[m-self.stride//2:m+self.stride//2,n-self.stride//2:n+self.stride//2] = pred_cl[self.overlap//2:x-self.overlap//2,self.overlap//2:y-self.overlap//2]
						self.prediction_logits_mosaic[m-self.stride//2:m+self.stride//2,n-self.stride//2:n+self.stride//2] = pred_logits[self.overlap//2:x-self.overlap//2,self.overlap//2:y-self.overlap//2]						
					self.count_mask_batch += 1
				if condition_masking:				
					self.count_mask += 1

			self.count = self.count + 1
			if self.count % 50000 == 0:
				print(self.count)

		ic(self.count_mask_batch)
		ic(self.count_mask)
	

		
	def getPatchesPerBatch(self):
		ic(self.count_mask)
		self.patches_per_batch = self.count_mask // self.pb.batch_processing_n
		ic(self.patches_per_batch)
		ic(self.patches_per_batch * self.pb.batch_processing_n)
		ic(self.pb.batch_processing_n)
		assert self.patches_per_batch * self.pb.batch_processing_n == self.count_mask
		assert self.patches_per_batch < 10200

	def deleteAllButLogits(self):
		del self.prediction_mosaic
		# del self.test_pred_proba
		del self.scores_mosaic
class MosaicHighRAMPostProcessing(MosaicHighRAM):
	def infer(self, paramsTrain, model, data, ds, postProcessing):
		self.openSetManager = postProcessing
			#pdb.set_trace()

		super().infer(paramsTrain, model, data, ds)

	def getFlatLabel(self):
		label_flat = self.label_mosaic.flatten()
		mask_flat = self.mask_pad.flatten()
		label_flat = label_flat[mask_flat == 2]


		ic(label_flat.shape, mask_flat)
#		pdb.set_trace()
		return label_flat
	
	def getFlatScores(self):
		scores_flat = self.openSetManager.scores_mosaic.flatten()
		mask_flat = self.mask_pad.flatten()
		scores_flat = scores_flat[mask_flat == 2]

		ic(scores_flat.shape, mask_flat)

		return scores_flat

	def getFlatPrediction(self):
		ic(self.prediction_mosaic.shape)
		# pdb.set_trace()
		prediction_flat = self.prediction_mosaic.flatten()
		mask_flat = self.mask_pad.flatten()
		prediction_flat = prediction_flat[mask_flat == 2]
		
		ic(prediction_flat.shape, mask_flat)
		# ic(np.unique(prediction_flat, return_counts = True))
		# pdb.set_trace()
		return prediction_flat

	def postProcess(self, paramsTrain):
		if self.paramsTrain.applyThreshold == True:
			self.prediction_mosaic = self.openSetManager.applyThreshold(self.prediction_mosaic, 
		 		debug = self.debug)
		ic(self.prediction_mosaic.shape)


'''
class MosaicOpenSet(MosaicHighRAM):
	def __init__(self, paramsTrain):

		self.openSetManager = PostProcessing()
		self.openSetManager.addMethod(OpenSet())
'''



def add_padding(img, psize, overl):
	'''Function to padding image
		input:
			patches_size: psize
			stride: stride
			img: image (row,col,bands)
	'''    

	try:
		row, col, bands = img.shape
	except:
		bands = 0
		row, col = img.shape
		
	# Percent of overlap between consecutive patches.
	# The overlap will be multiple of 2
	overlap = int(round(psize * overl))
	overlap -= overlap % 2
	stride = psize - overlap

	# Add Padding to the image to match with the patch size and the overlap
	row += overlap//2
	col += overlap//2
	step_row = (stride - row % stride) % stride
	step_col = (stride - col % stride) % stride
	
	if bands>0:
		npad_img = ((overlap//2, step_row), (overlap//2, step_col),(0,0))
	else:        
		npad_img = ((overlap//2, step_row), (overlap//2, step_col))  
		
	# padd with symetric (espelhado)    
	pad_img = np.pad(img, npad_img, mode='symmetric')

	# Number of patches: k1xk2
	k1, k2 = (row+step_row)//stride, (col+step_col)//stride
	print('Number of patches: %d' %(k1 * k2))

	return pad_img, stride, step_row, step_col, overlap

def seq_add_padding(img, psize, overl):
	'''Function to padding image
		input:
			patches_size: psize
			stride: stride
			img: image (row,col,bands)
	'''    
	try:
		t_len, row, col, bands = img.shape
	except:
		bands = 0
		t_len, row, col = img.shape
		
	# Percent of overlap between consecutive patches.
	# The overlap will be multiple of 2
	overlap = int(round(psize * overl))
	overlap -= overlap % 2
	stride = psize - overlap

	# Add Padding to the image to match with the patch size and the overlap
	row += overlap//2
	col += overlap//2
	step_row = (stride - row % stride) % stride
	step_col = (stride - col % stride) % stride
	
	if bands>0:
		npad_img = ((0,0),(overlap//2, step_row), (overlap//2, step_col),(0,0))
	else:        
		npad_img = ((0,0),(overlap//2, step_row), (overlap//2, step_col))  
	
	#pad_img = np.zeros((t_len,row, col, bands))
	# padd with symetric (espelhado)    
	#for t_step in t_len:
	#    pad_img[t_step] = np.pad(img[t_step], npad_img, mode='symmetric')
	pad_img = np.pad(img, npad_img, mode='symmetric')
	# Number of patches: k1xk2
	k1, k2 = (row+step_row)//stride, (col+step_col)//stride
	print('Number of patches: %d' %(k1 * k2))

	return pad_img, stride, step_row, step_col, overlap
