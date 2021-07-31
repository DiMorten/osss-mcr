 
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
#from dataSource import DataSource, SARSource, OpticalSource, Dataset, LEM, LEM2, CampoVerde, OpticalSourceWithClouds, Humidity
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
from parameters.parameters_reader import ParamsTrain, ParamsAnalysis
from analysis.open_set import SoftmaxThresholding, OpenPCS
from parameters.params_reconstruct import ParamsReconstruct
from parameters.params_batchprocessing import ParamsBatchProcessing

ic.configureOutput(includeContext=False, prefix='[@debug] ')



class Mosaic():
	def __init__(self, paramsTrain):
		self.pr = ParamsReconstruct(paramsTrain)

		self.paramsTrain = paramsTrain
		self.pr = ParamsReconstruct(paramsTrain)
		self.pb = ParamsBatchProcessing(paramsTrain, self.pr)


		ic(paramsTrain.seq_date)
		dataset=paramsTrain.dataset


		deb.prints(dataset)
		deb.prints(paramsTrain.model_type)
#		deb.prints(direct_execution)


	def evaluate(self):
		pass

	def create(self, paramsTrain, model, data):


		if self.pr.add_padding_flag==True:
			data.full_ims_test, stride, step_row, step_col, overlap = seq_add_padding(
				data.full_ims_test, paramsTrain.patch_len, self.pr.overlap)
			mask_pad, _, _, _, _ = add_padding(data.mask,paramsTrain.patch_len,self.pr.overlap)
		else:
			mask_pad=data.mask.copy()
			stride=paramsTrain.patch_len
			overlap=0
			

		t_len, h, w, _ = data.full_ims_test.shape
		ic(data.class_n)
		ic(np.unique(data.full_label_test), len(np.unique(data.full_label_test)))
		ic(np.unique(data.full_label_train), len(np.unique(data.full_label_train)))
		class_n = len(np.unique(data.full_label_train)) - 1
#		pdb.set_trace()
#		cl_img = np.zeros((h,w,class_n)) # is it class_n + 1?
#		cl_img = cl_img.astype('float16')

		prediction_mosaic=np.ones((h,w)).astype(np.uint8)*255
		scores_mosaic=np.zeros((h,w)).astype(np.float16)
		prediction_logits_mosaic=np.ones((h,w, class_n)).astype(np.float16)
		
		t0 = time.time()

		data.setDateList(paramsTrain)
		name_id = 'closed_set'
		if self.pr.mosaic_flag == True:
		
			for m in range(paramsTrain.patch_len//2,h-paramsTrain.patch_len//2,stride): 
				for n in range(paramsTrain.patch_len//2,w-paramsTrain.patch_len//2,stride):

					patch_mask = mask_pad[m-paramsTrain.patch_len//2:m+paramsTrain.patch_len//2 + paramsTrain.patch_len%2,
							n-paramsTrain.patch_len//2:n+paramsTrain.patch_len//2 + paramsTrain.patch_len%2]

					if self.pr.conditionType == 'test':
						condition = np.any(patch_mask==2)
					else:
						condition = True	
					if condition:		

						patch = data.full_ims_test[:, 
									m-paramsTrain.patch_len//2:m+paramsTrain.patch_len//2 + paramsTrain.patch_len%2,
									n-paramsTrain.patch_len//2:n+paramsTrain.patch_len//2 + paramsTrain.patch_len%2]
						patch = np.expand_dims(patch, axis = 0)
						pred_logits = model.predict(patch)
						pred = pred_logits.argmax(axis=-1).astype(np.uint8)
						_, x, y, c = pred_logits.shape

						#ic(pred_logits.shape, pred.shape)
						#ic(prediction_logits_mosaic.shape, prediction_mosaic.shape)
						
							
						prediction_logits_mosaic[m-stride//2:m+stride//2,n-stride//2:n+stride//2] = pred_logits[0,overlap//2:x-overlap//2,overlap//2:y-overlap//2]
						prediction_mosaic[m-stride//2:m+stride//2,n-stride//2:n+stride//2] = pred[0,overlap//2:x-overlap//2,overlap//2:y-overlap//2]

			
			if self.pr.open_set_mode == False:
				prediction_mosaic = prediction_logits_mosaic.argmax(axis=-1).astype(np.uint8)

			if self.pr.add_padding_flag==True:
				prediction_mosaic=prediction_mosaic[overlap//2:-step_row,overlap//2:-step_col]
				scores_mosaic=scores_mosaic[overlap//2:-step_row,overlap//2:-step_col]
			ic(prediction_mosaic.shape, mask_pad.shape, data.full_label_test.shape)
			ic(np.unique(data.full_label_test, return_counts=True))
			ic(np.unique(prediction_mosaic, return_counts=True))
			np.save(self.pr.spatial_results_path / 
				('prediction_mosaic_'+data.dataset_date+'_'+name_id+'_overl'+str(self.pr.overlap)+'.npy'),prediction_mosaic)
			if self.pr.open_set_mode == True:
				np.save(self.pr.spatial_results_path / 
					('scores_mosaic_'+data.dataset_date+'_'+name_id+'_overl'+str(self.pr.overlap)+'.npy'),scores_mosaic)

		else:

			prediction_mosaic = np.load(self.pr.spatial_results_path / 
				('prediction_mosaic_'+data.dataset_date+'_'+name_id+'_overl'+str(self.pr.overlap)+'.npy'))
			if self.pr.open_set_mode == True:
				scores_mosaic = np.load(self.pr.spatial_results_path / 
					('scores_mosaic_'+data.dataset_date+'_'+name_id+'_overl'+str(self.pr.overlap)+'.npy'))

		ic(time.time()-t0)

#		if self.pr.add_padding_flag==True:
#			data.full_label_test = data.full_label_test[overlap//2:-step_row,overlap//2:-step_col]



		ic(data.full_label_test.shape)

		
		cv2.imwrite('sample.png', prediction_logits_mosaic.argmax(axis=-1).astype(np.uint8)*10)
		cv2.imwrite('label_sample.png', data.full_label_test.astype(np.uint8)*10)

		ic(np.unique(prediction_mosaic, return_counts=True))

		prediction_mosaic = data.newLabel2labelTranslate(prediction_mosaic, 
				'new_labels2labels_'+paramsTrain.dataset+'_'+data.dataset_date+'_S1.pkl')
		ic(np.unique(prediction_mosaic, return_counts=True))
		cv2.imwrite('sample_translate.png', prediction_mosaic.astype(np.uint8)*10)

		ic(data.full_label_test.shape)
		ic(np.unique(data.full_label_test, return_counts=True))
		label_mosaic = data.full_label_test
		ic(np.unique(label_mosaic, return_counts=True))
		ic(np.unique(prediction_mosaic, return_counts=True))

		# bcknd to 255
		label_mosaic = label_mosaic - 1
		prediction_mosaic = prediction_mosaic - 1

		ic(np.unique(label_mosaic, return_counts=True))
		ic(np.unique(prediction_mosaic, return_counts=True))
#		pdb.set_trace()
		important_classes_idx = paramsTrain.known_classes


		label_mosaic, prediction_mosaic, important_classes_idx = data.small_classes_ignore(
					label_mosaic, prediction_mosaic,important_classes_idx)

		prediction_mosaic[prediction_mosaic==39] = 20
		label_mosaic[label_mosaic==39] = 20


		deb.prints(np.unique(label_mosaic,return_counts=True))
		deb.prints(np.unique(prediction_mosaic,return_counts=True))
		deb.prints(label_mosaic.shape)
		deb.prints(prediction_mosaic.shape)
		deb.prints(important_classes_idx)

		ic(data.mask.shape, mask_pad.shape, label_mosaic.shape, prediction_mosaic.shape)
		self.save_prediction_label_mosaic_Nto1(label_mosaic, prediction_mosaic, data.mask, 
				self.pr.custom_colormap, self.pr.spatial_results_path, paramsTrain, 
				small_classes_ignore=True,
				name_id = name_id)
		
		self.prediction_mosaic = prediction_mosaic
		self.label_mosaic = label_mosaic

	def save_prediction_label_mosaic_Nto1(self, label_mosaic, prediction_mosaic, mask, 
			custom_colormap, path, paramsTrain, small_classes_ignore=True, name_id=""):
	#	for t_step in range(t_len):

		ic(np.unique(mask, return_counts=True))

		label_mosaic[mask!=2]=255
		
		if self.pr.prediction_mask == True:
			prediction_mosaic[mask!=2]=255	
		deb.prints(np.unique(label_mosaic,return_counts=True))
		deb.prints(np.unique(prediction_mosaic,return_counts=True))


		print("everything outside mask is 255")
		ic(np.unique(label_mosaic,return_counts=True))
		ic(np.unique(prediction_mosaic,return_counts=True))

		# Paint it!

		print(custom_colormap.shape)
		#class_n=custom_colormap.shape[0]
		#=== change to rgb
		print("Gray",prediction_mosaic.dtype)
		prediction_rgb=np.zeros((prediction_mosaic.shape+(3,))).astype(np.uint8)
		label_rgb=np.zeros_like(prediction_rgb)
		print("Adding color...")


		prediction_rgb=cv2.cvtColor(prediction_mosaic,cv2.COLOR_GRAY2RGB)
		label_rgb=cv2.cvtColor(label_mosaic,cv2.COLOR_GRAY2RGB)

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
		save_folder=path / paramsTrain.dataset / paramsTrain.model_type / paramsTrain.seq_date
		save_folder.mkdir(parents=True, exist_ok=True)
		deb.prints(save_folder)

		if self.pr.open_set_mode == True:
			threshIdxName = "_TPR" + tpr_threshold_names[self.pr.threshold_idx]
			prediction_savename = save_folder / ("prediction_t_" + paramsTrain.seq_date + "_" + paramsTrain.model_type +
				"_" + name_id+threshIdxName + "_overl" + str(self.pr.overlap) + "_" + self.pr.conditionType + ".png")

		else:
			prediction_savename = save_folder / ("prediction_t_" + paramsTrain.seq_date + "_" + paramsTrain.model_type +
				"_closedset_" + name_id + "_overl" + str(self.pr.overlap) + "_" + self.pr.conditionType + ".png")

		ic(prediction_savename)
		print("saving...")
		try:

			os.remove(prediction_savename)
		except:
			print("no file to remove")
		ret = cv2.imwrite(str(prediction_savename), prediction_rgb)

		deb.prints(ret)
		ic(save_folder / ("label_t_"+paramsTrain.seq_date+"_"+paramsTrain.model_type+"_"+name_id+".png"))
		ret = cv2.imwrite(str(save_folder / ("label_t_"+paramsTrain.seq_date+"_"+paramsTrain.model_type+"_"+name_id+".png")),label_rgb)
		deb.prints(ret)
		ret = cv2.imwrite(str(save_folder / "mask.png"),mask*200)
		deb.prints(ret)



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
