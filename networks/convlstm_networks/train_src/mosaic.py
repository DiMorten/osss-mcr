 
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


		if paramsTrain.add_padding_flag==True:
			data.full_ims_test, stride, step_row, step_col, overlap = seq_add_padding(
				data.full_ims_test, paramsTrain.patch_len, paramsTrain.overlap)
			mask_pad, _, _, _, _ = add_padding(data.mask,paramsTrain.patch_len,overlap)
		else:
			#mask_pad=mask.copy()
			stride=paramsTrain.patch_len
			overlap=0

		t_len, h, w, _ = data.full_ims_test.shape
		ic(data.class_n)
		ic(np.unique(data.full_label_test), len(np.unique(data.full_label_test)))
		ic(np.unique(data.full_label_train), len(np.unique(data.full_label_train)))
		class_n = len(np.unique(data.full_label_train)) - 1
#		pdb.set_trace()
		cl_img = np.zeros((h,w,class_n)) # is it class_n + 1?
		cl_img = cl_img.astype('float16')

		prediction_mosaic=np.ones((row,col)).astype(np.uint8)*255
		scores_mosaic=np.zeros((row,col)).astype(np.float16)
		prediction_logits_mosaic=np.ones((row,col, class_n)).astype(np.float16)
		
		t0 = time.time()
		if pr.mosaic_flag == True:
		
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
							
						prediction_logits_mosaic[m-stride//2:m+stride//2,n-stride//2:n+stride//2,:] = pred_logits[0,overlap//2:x-overlap//2,overlap//2:y-overlap//2,:]
						prediction_mosaic[m-stride//2:m+stride//2,n-stride//2:n+stride//2,:] = pred[0,overlap//2:x-overlap//2,overlap//2:y-overlap//2,:]

			
			prediction_logits_mosaic = prediction_logits_mosaic[overlap//2:-step_row,overlap//2:-step_col,:]
			np.save(self.spatial_results_path / 
				('prediction_mosaic_'+dataset_date+'_'+name_id+'_overl'+str(pr.overlap)+'.npy'),prediction_mosaic)
			if pr.open_set_mode == True:
				np.save(self.spatial_results_path / 
					('scores_mosaic_'+dataset_date+'_'+name_id+'_overl'+str(pr.overlap)+'.npy'),scores_mosaic)

		else:

			prediction_mosaic = np.load(self.spatial_results_path / 
				('prediction_mosaic_'+dataset_date+'_'+name_id+'_overl'+str(pr.overlap)+'.npy'))
			if pr.open_set_mode == True:
				scores_mosaic = np.load(self.spatial_results_path / 
					('scores_mosaic_'+dataset_date+'_'+name_id+'_overl'+str(pr.overlap)+'.npy'))

		data.full_label_test = data.full_label_test[overlap//2:-step_row,overlap//2:-step_col]

		data.setDateList(paramsTrain)
		ic(cl_img.shape)
		ic(data.full_label_test.shape)

		
		cv2.imwrite('sample.png', cl_img.argmax(axis=-1).astype(np.uint8)*10)
		cv2.imwrite('label_sample.png', data.full_label_test.astype(np.uint8)*10)
		cl_img_int = cl_img.argmax(axis=-1)
		ic(np.unique(cl_img_int, return_counts=True))

		cl_img_int = data.newLabel2labelTranslate(cl_img_int, 
				'new_labels2labels_'+paramsTrain.dataset+'_'+data.dataset_date+'_S1.pkl')
		ic(np.unique(cl_img_int, return_counts=True))
		cv2.imwrite('sample_translate.png', cl_img_int.astype(np.uint8)*10)


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