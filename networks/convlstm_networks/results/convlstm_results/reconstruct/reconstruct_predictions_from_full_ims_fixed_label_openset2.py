 
import numpy as np
import cv2
import glob
import argparse
import pdb
import sys, os
#sys.path.append('../../../../../train_src/analysis/')
import pathlib
from utils import seq_add_padding, add_padding
import pdb
sys.path.append('../../../train_src/')
from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq
sys.path.append('../../../../../dataset/dataset/patches_extract_script/')
from dataSource import DataSource, SARSource, OpticalSource, Dataset, LEM, LEM2, CampoVerde, OpticalSourceWithClouds, Humidity
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report,recall_score,precision_score
import colorama
colorama.init()
import pickle
import deb
import time
#sys.path.append('../../../train_src/analysis')
#print(sys.path)
from PredictionsLoader import PredictionsLoaderNPY, PredictionsLoaderModel, PredictionsLoaderModelNto1FixedSeqFixedLabelOpenSet
from icecream import ic 
from parameters.parameters_reader import ParamsTrain, ParamsAnalysis
from analysis.open_set import SoftmaxThresholding, OpenPCS
from params_reconstruct import ParamsReconstruct

ic.configureOutput(includeContext=False, prefix='[@debug] ')


paramsTrain = ParamsTrain('../../../train_src/parameters/')
paramsAnalysis = ParamsAnalysis('../../../train_src/analysis/parameters_analysis/')
pr = ParamsReconstruct()


ic(paramsTrain.seq_date)
dataset=paramsTrain.dataset

direct_execution=False
if direct_execution==True:
	dataset='lm'
	paramsTrain.model_type='unet'

if paramsTrain.model_type == 'UUnet4ConvLSTM':
	paramsTrain.model_type = 'unet'
deb.prints(dataset)
deb.prints(paramsTrain.model_type)
deb.prints(direct_execution)

def patch_file_id_order_from_folder(folder_path):
	paths=glob.glob(folder_path+'*.npy')
	print(paths[:10])	
	order=[int(paths[i].partition('patch_')[2].partition('_')[0]) for i in range(len(paths))]
	print(len(order))
	print(order[0:20])
	return order

path='../model/'

data_path='../../../../../dataset/dataset/'
if dataset=='lm':

	path+='lm/'
	if paramsTrain.model_type=='densenet':
		predictions_path=path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
	elif paramsTrain.model_type=='biconvlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy'
	elif paramsTrain.model_type=='convlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_batch16_full.npy'
	elif paramsTrain.model_type=='unet':
		predictions_path=path+'prediction_BUnet4ConvLSTM_repeating1.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating4.npy'
		predictions_path = path+'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+paramsTrain.seq_date+'_700perclass.h5'			
		predictions_path = path+'model_best_UUnet4ConvLSTM_fixed_label_fixed_'+paramsTrain.seq_date+'_loco8_lm_testlm_fewknownclasses.h5'	
		if paramsTrain.seq_date == 'mar':
			predictions_path = path+'model_best_UUnet4ConvLSTM_fixed_label_fixed_'+paramsTrain.seq_date+'_loco8_lm_testlm_fewknownclasses.h5'	
			predictions_path = path+'model_lm_mar_nomask_good.h5'	
			#predictions_path = path+'model_best_UUnet4ConvLSTM_mar_lm_fixed_fewknownclasses_groupclasses_newdataaugmentation_coords.h5'
		elif paramsTrain.seq_date == 'jun':
			predictions_path = path+'model_best_UUnet4ConvLSTM_fixed_label_fixed_jun_lm_fewknownclasses2.h5'	
			predictions_path = path+'model_best_UUnet4ConvLSTM_jun_lm_.h5'	

	elif paramsTrain.model_type=='atrous':
		predictions_path=path+'prediction_BAtrousConvLSTM_2convins5.npy'
	elif paramsTrain.model_type=='atrousgap':
		predictions_path=path+'prediction_BAtrousGAPConvLSTM_raulapproved.npy'
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating3.npy'
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating4.npy'
		


	mask_path=data_path+'lm_data/TrainTestMask.tif'
	location_path=data_path+'lm_data/locations/'
	folder_load_path=data_path+'lm_data/train_test/test/labels/'

	custom_colormap = np.array([[255,146,36],
					[255,255,0],
					[164,164,164],
					[255,62,62],
					[0,0,0],
					[172,89,255],
					[0,166,83],
					[40,255,40],
					[187,122,83],
					[217,64,238],
					[0,113,225],
					[128,0,0],
					[114,114,56],
					[53,255,255]])
elif dataset=='cv':

	path+='cv/'
	if paramsTrain.model_type=='densenet':
		predictions_path=path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
	elif paramsTrain.model_type=='biconvlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy'
	elif paramsTrain.model_type=='convlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_batch16_full.npy'		
	elif paramsTrain.model_type=='unet':
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy'
		predictions_path=path+'model_best_BUnet4ConvLSTM_int16.h5'
		if paramsTrain.seq_date == 'jun':
			predictions_path = path+'model_best_UUnet4ConvLSTM_jun.h5'
			predictions_path = path+'model_best_UUnet4ConvLSTM_jun_cv_criteria_0_92.h5'
		if paramsTrain.seq_date == 'may':
			predictions_path = path+'model_cv_may_3classes_nomask.h5'
	elif paramsTrain.model_type=='atrous':
		predictions_path=path+'prediction_BAtrousConvLSTM_repeating2.npy'			
	elif paramsTrain.model_type=='atrousgap':
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_raulapproved.npy'			
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating4.npy'			
		predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating6.npy'			
	elif paramsTrain.model_type=='unetend':
		predictions_path=path+'prediction_unet_convlstm_temouri2.npy'			
	elif paramsTrain.model_type=='allinputs':
		predictions_path=path+'prediction_bconvlstm_wholeinput.npy'			

	mask_path=data_path+'cv_data/TrainTestMask.tif'
	location_path=data_path+'cv_data/locations/'

	folder_load_path=data_path+'cv_data/train_test/test/labels/'

	custom_colormap = np.array([[255, 146, 36],
				   [255, 255, 0],
				   [164, 164, 164],
				   [255, 62, 62],
				   [0, 0, 0],
				   [172, 89, 255],
				   [0, 166, 83],
				   [40, 255, 40],
				   [187, 122, 83],
				   [217, 64, 238],
				   [45, 150, 255]])
elif dataset=='l2':
	path+='l2/'	
	if paramsTrain.model_type=='unet':
		predictions_path=path+'prediction_BUnet4ConvLSTM_repeating1.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating4.npy'
		predictions_path = path+'model_best_UUnet4ConvLSTM_doty_fixed_label_dec.h5'
		predictions_path = path+'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+paramsTrain.seq_date+'.h5'
		predictions_path = path+'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+paramsTrain.seq_date+'_700perclass.h5'
#		predictions_path = path+'model_best_UUnet4ConvLSTM_doty_fixed_label_dec_good_slvc05.h5'
	mask_path=data_path+'l2_data/TrainTestMask.tif'
	location_path=data_path+'l2_data/locations/'
	folder_load_path=data_path+'l2_data/train_test/test/labels/'

	custom_colormap = np.array([[255,146,36],
					[255,255,0],
					[164,164,164],
					[255,62,62],
					[0,0,0],
					[172,89,255],
					[0,166,83],
					[40,255,40],
					[187,122,83],
					[217,64,238],
					[0,113,225],
					[128,0,0],
					[114,114,56],
					[53,255,255]])
print("Loading patch locations...")
ic(dataset)
ic(paramsTrain.model_type)
ic(predictions_path)
#order_id_load=False
#if order_id_load==False:
#	order_id=patch_file_id_order_from_folder(folder_load_path)
#	np.save('order_id.npy',order_id)
#else:
#	order_id=np.load('order_id.npy')

#cols=np.load(location_path+'locations_col.npy')
#rows=np.load(location_path+'locations_row.npy')

#print(cols.shape, rows.shape)
#cols=cols[order_id]
#rows=rows[order_id]

# ======== load labels and predictions 

#labels=np.load(path+'labels.npy').argmax(axis=4)
#predictions=np.load(predictions_path).argmax(axis=4)

print("Loading labels and predictions...")

pr.prediction_type = 'model'
results_path="../"
#path=results_path+dataset+'/'
#prediction_path=path+predictions_path
path_test='../../../../../dataset/dataset/'+dataset+'_data/patches_bckndfixed/test/'
print('path_test',path_test)

#pr.prediction_type = 'model'
if pr.prediction_type=='npy':
	predictionsLoader = PredictionsLoaderNPY()
	predictions, labels = predictionsLoader.loadPredictions(predictions_path,path+'labels.npy')
elif pr.prediction_type=='model':	
	#model_path=results_path + 'model/'+dataset+'/'+prediction_filename
	print('model_path',predictions_path)

	# predictionsLoader = PredictionsLoaderModel(path_test)
	
	
	#PredictionsLoaderModelNto1FixedSeqFixedLabelOpenSet
	predictionsLoaderTest = PredictionsLoaderModelNto1FixedSeqFixedLabelOpenSet(path_test, dataset=dataset)
#predictions, label_test, test_pred_proba, model = predictionsLoaderTest.loadPredictions(predictions_path, seq_date=paramsTrain.seq_date, 
#		model_dataset=paramsTrain.model_dataset)	
# load the full ims ... then create the known classes and unknown class
# using the paramsTrain values...
# maybe use the mim object (although not sure if needed)
	model = predictionsLoaderTest.loadModel(predictions_path)
#================= load labels and predictions






#class_n=np.max(predictions)+1
#print("class_n",class_n)
#labels[labels==class_n]=255 # background

# Print stuff
#print(cols.shape)
#print(rows.shape)
#print(labels.shape)
#print(predictions.shape)
#print("np.unique(labels,return_counts=True)",
#	np.unique(labels,return_counts=True))
#print("np.unique(predictions,return_counts=True)",
#	np.unique(predictions,return_counts=True))

# Specify variables
#sequence_len=labels.shape[1]
#patch_len=labels.shape[2]

# Load mask
mask=cv2.imread(mask_path,-1)
mask[mask==1]=0 # training as background
print("Mask shape",mask.shape)
#print((sequence_len,)+mask.shape)

# ================= LOAD THE INPUT IMAGE.
full_path = '../../../../../dataset/dataset/'+dataset+'_data/full_ims/' 
full_ims_test = np.load(full_path+'full_ims_test.npy')
full_label_test = np.load(full_path+'full_label_test.npy').astype(np.uint8)

ic(full_ims_test.shape)
ic(full_label_test.shape)
#pdb.set_trace()
# ================ HERE CROP THE IMAGE IF NEEDED

if pr.croppedFlag == True:

#	full_ims_test = full_ims_test[:, 5200:6100,4900:6800]
#	full_label_test = full_label_test[:, 5200:6100,4900:6800]
#	mask = mask[5200:6100,4900:6800]

	full_ims_test = full_ims_test[:, 5100:6100,4900:5900]
	full_label_test = full_label_test[:, 5100:6100,4900:5900]
	mask = mask[5100:6100,4900:5900]

if pr.save_input_im == True:
	im = np.load('../../../../../dataset/dataset/'+dataset+'_data/in_np2/20180315_S1.npy')
	ic(im.shape)
	im = im[5100:6100,4900:5900, 1]
	ic(im.shape)
#	max_val = np.average(im)+np.std(im)*4
	max_val = 0.6
	im = im * 255/max_val
	ic(np.min(im), np.average(im), np.max(im))
	cv2.imwrite('sample_im.png', im.astype(np.uint8))
	#pdb.set_trace()
# convert labels; background is last
#class_n=len(np.unique(full_label_test))-1
#full_label_test=full_label_test-1
#full_label_test[full_label_test==255]=class_n

ic(full_ims_test.shape)
ic(full_label_test.shape)
ic(mask.shape)
#pdb.set_trace()
print("Full label test unique",np.unique(full_label_test,return_counts=True))
#pdb.set_trace()
# add doty
#mim = MIMFixed()
mim = MIMFixed_PaddedSeq()

data = {'labeled_dates': 12}
data['labeled_dates'] = 12

seq_mode='fixed'

if dataset=='lm':
	ds=LEM(seq_mode, paramsTrain.seq_date)
elif dataset=='l2':
	ds=LEM2(seq_mode, paramsTrain.seq_date)
elif dataset=='cv':
	ds=CampoVerde(seq_mode, paramsTrain.seq_date)
deb.prints(ds)
dataSource = SARSource()
ds.addDataSource(dataSource)

time_delta = ds.getTimeDelta(delta=True,format='days')
ds.setDotyFlag(False)
dotys, dotys_sin_cos = ds.getDayOfTheYear()
ds.dotyReplicateSamples(sample_n = 1)

# Reconstruct the image
print("Reconstructing the labes and predictions...")

patch_size=paramsTrain.patch_len
# pr.add_padding_flag=True
if pr.add_padding_flag==True:
	full_ims_test, stride, step_row, step_col, overlap = seq_add_padding(
		full_ims_test, patch_size, pr.overlap)
	#full_label_test, _, _, _, _ = seq_add_padding(full_label_test,32,0)
	mask_pad, _, _, _, _ = add_padding(mask,patch_size,0)
else:
	mask_pad=mask.copy()
	stride=patch_size
	overlap=0
print(full_ims_test.shape)
print(full_label_test.shape)
print("Full label test unique",np.unique(full_label_test,return_counts=True))

sequence_len, row, col, bands = full_ims_test.shape
#pdb.set_trace()

label_rebuilt=full_label_test[-1]
print("full_label_test.shape, label_rebuilt.shape", full_label_test.shape, label_rebuilt.shape)
print("label_rebuilt.shape",label_rebuilt.shape)
print("label_rebuilt.unique",np.unique(label_rebuilt,return_counts=True))

##pdb.set_trace()
lm_labeled_dates = ['20170612', '20170706', '20170811', '20170916', '20171010', '20171115', 
					'20171209', '20180114', '20180219', '20180315', '20180420', '20180514']
l2_labeled_dates = ['20191012','20191117','20191223','20200116','20200221','20200316',
					'20200421','20200515','20200620','20200714','20200819','20200912']
cv_labeled_dates = ['20151029', '20151110', '20151122', '20151204', '20151216', '20160121', 
					'20160214', '20160309', '20160321', '20160508', '20160520', '20160613', 
					'20160707', '20160731']
if paramsTrain.dataset == 'lm':
	if paramsTrain.seq_date=='jan':
		dataset_date = lm_labeled_dates[7]
		l2_date = l2_labeled_dates[3]

	elif paramsTrain.seq_date=='feb':
		dataset_date = lm_labeled_dates[8]
		l2_date = l2_labeled_dates[4]

	elif paramsTrain.seq_date=='mar':
		dataset_date = lm_labeled_dates[9]
		l2_date = l2_labeled_dates[5]

	elif paramsTrain.seq_date=='apr':
		dataset_date = lm_labeled_dates[10]
		l2_date = l2_labeled_dates[6]

	elif paramsTrain.seq_date=='may':
		dataset_date = lm_labeled_dates[11]
		l2_date = l2_labeled_dates[7]

	elif paramsTrain.seq_date=='jun':
		dataset_date = lm_labeled_dates[0]
		l2_date = l2_labeled_dates[8]

	elif paramsTrain.seq_date=='jul':
		dataset_date = lm_labeled_dates[1]
		l2_date = l2_labeled_dates[9]

	elif paramsTrain.seq_date=='aug':
		dataset_date = lm_labeled_dates[2]
		l2_date = l2_labeled_dates[10]

	elif paramsTrain.seq_date=='sep':
		dataset_date = lm_labeled_dates[3]
		l2_date = l2_labeled_dates[11]

	elif paramsTrain.seq_date=='oct':
		dataset_date = lm_labeled_dates[4]
		l2_date = l2_labeled_dates[0]

	elif paramsTrain.seq_date=='nov':
		dataset_date = lm_labeled_dates[5]
		l2_date = l2_labeled_dates[1]

	if paramsTrain.seq_date=='dec':
	#dec
		dataset_date = lm_labeled_dates[6]
		l2_date = l2_labeled_dates[2]
elif paramsTrain.dataset == 'cv':
	if paramsTrain.seq_date=='jun':
		dataset_date = cv_labeled_dates[11]
deb.prints(paramsTrain.seq_date)
deb.prints(dataset_date)

del full_label_test
translate_label_path = '../../../train_src/'
#name_id = "closed_set"
#name_id = "openpca_identitycovmatrix_90pcs_crop"
name_id = paramsAnalysis.openSetMethod

if paramsAnalysis.openSetMethod == "OpenPCS" and paramsAnalysis.makeCovMatrixIdentity == True:
	name_id = name_id + "++" 
name_id = name_id + "_" + paramsTrain.dataset

#name_id = name_id + "_" + paramsTrain.seq_date

if pr.croppedFlag == True:
	name_id = name_id + "_crop"

# pr.open_set_mode = True
# pr.mosaic_flag = False

# --================= open set

tpr_threshold_values = [0.1, 0.3, 0.5, 0.7, 0.9]
tpr_threshold_names = ['0_1', '0_3', '0_5', '0_7', '0_9']
if paramsTrain.dataset == 'lm':
	if paramsAnalysis.openSetMethod == 'SoftmaxThresholding':
		thresholds = [0.956, 0.9395, 0.9194, 0.8896, 0.796 ]
	elif paramsAnalysis.openSetMethod == 'OpenPCS' and paramsAnalysis.makeCovMatrixIdentity == True:
		thresholds = [-109., -116.4, -125.2, -139.3, -177.3]
	elif paramsAnalysis.openSetMethod == 'OpenPCS' and paramsAnalysis.makeCovMatrixIdentity == False:
		thresholds = [102.2, 64.2, 53.7, 36.0, -3.5]
elif paramsTrain.dataset == 'cv':
	if paramsAnalysis.openSetMethod == 'SoftmaxThresholding':
		thresholds = [0.693, 0.647, 0.5845, 0.4814, 0.4177]
	elif paramsAnalysis.openSetMethod == 'OpenPCS' and paramsAnalysis.makeCovMatrixIdentity == True:
		thresholds = [-105.84688768, -112.12532142, -118.91670095, -130.085203,   -162.30341175]
	elif paramsAnalysis.openSetMethod == 'OpenPCS' and paramsAnalysis.makeCovMatrixIdentity == False:
		thresholds = [231.84498577, 213.47336953, 205.30315372, 194.55497196, 162.43701695]


	

deb.prints(thresholds)
#	threshold = -19
#	threshold = 100
#	threshold = -210
#threshold = -175
#	threshold = 0.7
##threshold = 0.7
#	threshold = -1
# pr.threshold_idx = 4
threshold = thresholds[pr.threshold_idx]

ic(paramsAnalysis.openSetMethod)
ic(threshold)
ic(paramsAnalysis.makeCovMatrixIdentity)

known_classes = [x + 1 for x in paramsTrain.known_classes]
deb.prints(known_classes)
if paramsAnalysis.openSetMethod == 'OpenPCS':
	openModel = OpenPCS(known_classes = known_classes,
#			n_components = 16)
		n_components = 90)
	openModel.makeCovMatrixIdentitySet(paramsAnalysis.makeCovMatrixIdentity)
elif paramsAnalysis.openSetMethod == 'SoftmaxThresholding':
	openModel = SoftmaxThresholding()
openModel.setThreshold(threshold)

try:
	openModel.setModelSaveNameID(paramsTrain.seq_date, paramsTrain.dataset)
#	nameID = paramsAnalysis.openSetMethod
#	if paramsAnalysis.makeCovMatrixIdentity == True:
#		nameID = nameID + "_covmatrix"
	openModel.loadFittedModel(path = '../../../train_src/analysis/', nameID = openModel.nameID)
#	openModel.loadFittedModel(path = '../../../train_src/analysis/')

except:
	print("Exception: No fitted model method")

debug = -2

if pr.mosaic_flag == True:
	prediction_rebuilt=np.ones((row,col)).astype(np.uint8)*255
	scores_rebuilt=np.zeros((row,col)).astype(np.float16)



	print("stride", stride)
	print(len(range(patch_size//2,row-patch_size//2,stride)))
	print(len(range(patch_size//2,col-patch_size//2,stride)))


#	debug = 1
	t0 = time.time()
	count = 0
	# score get
	for m in range(patch_size//2,row-patch_size//2,stride): 
		for n in range(patch_size//2,col-patch_size//2,stride):
			patch_mask = mask_pad[m-patch_size//2:m+patch_size//2 + patch_size%2,
						n-patch_size//2:n+patch_size//2 + patch_size%2]
			if np.any(patch_mask==2):
				patch = {}			
				patch['in'] = full_ims_test[:,m-patch_size//2:m+patch_size//2 + patch_size%2,
							n-patch_size//2:n+patch_size//2 + patch_size%2]
							
				patch['in'] = np.expand_dims(patch['in'], axis = 0)
				#patch = patch.reshape((1,patch_size,patch_size,bands))

				# features = predictionsLoaderTest.getFeatures(patch['in'], )
				patch['shape'] = (patch['in'].shape[0], paramsTrain.seq_len) + patch['in'].shape[2:]

				input_ = mim.batchTrainPreprocess(patch, ds,  
							label_date_id = -1) # tstep is -12 to -1

				pred_logits = np.squeeze(model.predict(input_))
				if pr.open_set_mode == True:
					if debug>-1:
						print('*'*20, "Load decoder features")
						ic(paramsAnalysis.openSetMethod)

					if paramsAnalysis.openSetMethod =='OpenPCS':
						test_pred_proba = predictionsLoaderTest.load_decoder_features(model, input_, debug = debug) # , debug = debug
					else:
						test_pred_proba = pred_logits.copy()
						test_pred_proba_shape = test_pred_proba.shape 
						if debug>0:
							ic(test_pred_proba_shape) # h, w, classes
						test_pred_proba = np.reshape(test_pred_proba, (-1, test_pred_proba.shape[-1]))
#				ic(np.average(test_pred_proba))

				#print(input_[0].shape)
				#ic(len(input_))

#				ic(input_.shape)
				pred_cl = pred_logits.argmax(axis=-1)
				#deb.prints(pred_cl.shape)
				x, y = pred_cl.shape
				prediction_shape = pred_cl.shape
				if debug>-1:
					print('*'*20, "Starting openModel predict")
					ic(pred_cl.shape)
					ic(test_pred_proba.shape)

					ic(np.min(test_pred_proba), np.average(test_pred_proba), np.median(test_pred_proba), np.max(test_pred_proba))
				# ========================================== open set
				if pr.open_set_mode == True:
					# translate the preddictions.
					pred_cl = predictionsLoaderTest.newLabel2labelTranslate(pred_cl, 
							translate_label_path + 'new_labels2labels_'+paramsTrain.dataset+'_'+dataset_date+'_S1.pkl',
							bcknd_flag=False, debug = debug)

					if debug>0:
						ic(pred_cl.shape)
					#ic()
					#test_pred_proba = np.reshape(test_pred_proba, test_pred_proba_shape)
					openModel.predictScores(pred_cl.flatten() - 1, test_pred_proba,
								debug = debug)
#					openModel.predictScores(pred_cl.flatten(), test_pred_proba,
#								debug = debug)
#					pdb.set_trace()
					openModel.scores = np.reshape(openModel.scores, (x, y)) # reshape to h, w
					if debug>-2:
						ic(np.min(test_pred_proba), np.average(test_pred_proba), 
							np.median(test_pred_proba), np.max(test_pred_proba))
						ic(np.min(openModel.scores), np.average(openModel.scores), 
							np.median(openModel.scores), np.max(openModel.scores))
						ic(openModel.scores.shape)
						ic(test_pred_proba.shape)

						idx = 1020
						ic(np.min(test_pred_proba[idx]), np.average(test_pred_proba[idx]), 
							np.median(test_pred_proba[idx]), np.max(test_pred_proba[idx]))
						ic(openModel.scores.flatten()[idx].shape)
						ic(openModel.scores.flatten()[idx])
						ic(pred_cl.flatten()[idx])

#						pdb.set_trace()

					#pdb.set_trace()
					# load the pca model / covariance matrix 
					#ic(pred_cl.shape)

				##deb.prints(np.unique(predictions_openmodel, return_counts=True))
				#deb.prints(predictions_openmodel.shape)
				##deb.prints(np.unique(prediction_rebuilt, return_counts=True))
				if debug>1:
					ic(openModel.scores.shape)
					ic(overlap)
					ic(openModel.scores[overlap//2:x-overlap//2,overlap//2:y-overlap//2].shape)
				if pr.open_set_mode == True:
					scores_rebuilt[m-stride//2:m+stride//2,n-stride//2:n+stride//2] = openModel.scores[overlap//2:x-overlap//2,overlap//2:y-overlap//2]
				if pr.overlap_mode == 'replace':
					prediction_rebuilt[m-stride//2:m+stride//2,n-stride//2:n+stride//2] = pred_cl[overlap//2:x-overlap//2,overlap//2:y-overlap//2]
				elif pr.overlap_mode == 'average':
					pred_patch_prev = np.expand_dims(prediction_rebuilt[m-stride//2:m+stride//2,n-stride//2:n+stride//2], axis = 0)
					pred_patch = np.expand_dims(pred_cl[overlap//2:x-overlap//2,overlap//2:y-overlap//2], axis = 0)
					to_average = np.concatenate((pred_patch_prev, pred_patch), axis = 0)
					
					prediction_rebuilt[m-stride//2:m+stride//2,n-stride//2:n+stride//2] = np.median(
						to_average, axis = 0)
				elif pr.overlap_mode == 'central':
					ic(stride)
					ic(overlap)
					prediction_rebuilt[m-stride//4:m+stride//4,n-stride//4:n+stride//4] = pred_cl[overlap//4:x-overlap//4,overlap//4:y-overlap//4]

#				prediction_rebuilt[m-stride//2:m+stride//2,n-stride//2:n+stride//2] = predictions_openmodel[:,overlap//2:x-overlap//2,overlap//2:y-overlap//2]

				##deb.prints(np.unique(prediction_rebuilt, return_counts=True))
				#pdb.set_trace()
			count = count + 1
			if count % 50000 == 0:
				print(count)

		#if count == 40:
		#	deb.prints(np.unique(prediction_rebuilt, return_counts=True))
		#	break
	del full_ims_test
	print("loop time: ", time.time()-t0)
	if pr.add_padding_flag==True:
		ic(prediction_rebuilt.shape)
		ic(overlap)
		ic(step_row)
		prediction_rebuilt=prediction_rebuilt[overlap//2:-step_row,overlap//2:-step_col]

	print("---- pad was removed")

	print(prediction_rebuilt.shape, mask.shape, label_rebuilt.shape)

	print(np.unique(label_rebuilt, return_counts=True))
	print(np.unique(prediction_rebuilt, return_counts=True))

	#prediction_rebuilt=np.reshape(prediction_rebuilt,-1)

	np.save('prediction_rebuilt_'+dataset_date+'_'+name_id+'_overl'+str(pr.overlap)+'.npy',prediction_rebuilt)
	if pr.open_set_mode == True:
		np.save('scores_rebuilt_'+dataset_date+'_'+name_id+'_overl'+str(pr.overlap)+'.npy',scores_rebuilt)
	
else:
	prediction_rebuilt = np.load('prediction_rebuilt_'+dataset_date+'_'+name_id+'_overl'+str(pr.overlap)+'.npy')
	if pr.open_set_mode == True:
		scores_rebuilt = np.load('scores_rebuilt_'+dataset_date+'_'+name_id+'_overl'+str(pr.overlap)+'.npy')

# ==== checking scores
if pr.open_set_mode == True:
	if debug>-3:
		scores_test = scores_rebuilt[mask == 2]
		ic(np.min(scores_test), np.average(scores_test), 
			np.median(scores_test), np.max(scores_test))
		ic(np.min(scores_rebuilt), np.average(scores_rebuilt), 
			np.median(scores_rebuilt), np.max(scores_rebuilt))
		ic(scores_test.shape)
		ic(scores_test.flatten().shape)
		ic(scores_rebuilt.shape)

		deb.prints(np.unique(prediction_rebuilt,return_counts=True))

	prediction_rebuilt = openModel.predict(prediction_rebuilt, scores_rebuilt, debug = debug)


if debug>-1:
	print('*'*20, "Finished openModel predict")


deb.prints(np.unique(prediction_rebuilt,return_counts=True))
deb.prints(label_rebuilt.shape)
#label_rebuilt=np.reshape(label_rebuilt,-1)
print("label_rebuilt.unique",np.unique(label_rebuilt,return_counts=True))


#mask = np.reshape(mask,-1)
deb.prints(prediction_rebuilt.shape)
# THIS NEEDS TO BE DONE BEFORE THE OPEN SET
if pr.open_set_mode == False:
	prediction_rebuilt = predictionsLoaderTest.newLabel2labelTranslate(prediction_rebuilt, 
			translate_label_path + 'new_labels2labels_'+paramsTrain.dataset+'_'+dataset_date+'_S1.pkl',
			bcknd_flag=False)

deb.prints(prediction_rebuilt.shape)
#pdb.set_trace()
deb.prints(np.unique(prediction_rebuilt,return_counts=True))
metrics_flag=True
if metrics_flag==True:
	# ========== metrics get =======#
	def my_f1_score(label,prediction):
		f1_values=f1_score(label,prediction,average=None)

		#label_unique=np.unique(label) # [0 1 2 3 5]
		#prediction_unique=np.unique(prediction.argmax(axis-1)) # [0 1 2 3 4]
		#[ 0.8 0.8 0.8 0 0.7 0.7]

		f1_value=np.sum(f1_values)/len(np.unique(label))

		#print("f1_values",f1_values," f1_value:",f1_value)
		return f1_value



	def metrics_get(label, predictions, mask, small_classes_ignore=True):



		class_n = 15
		print("label predictions shape, beginning of metrics_get",label.shape,predictions.shape)
		predictions=predictions[mask==2]
		label=label[mask==2]
		print("label predictions shape, test area",label.shape,predictions.shape)
		print("Before small classes ignore")

		print("Metrics get predictions",np.unique(predictions, return_counts=True))
		print("Metrics get label",np.unique(label, return_counts=True))
		predictions=predictions[label!=0]
		label=label[label!=0]	

		predictions = predictions - 1
		label = label - 1
		print("label predictions shape, no bcknd",label.shape,predictions.shape)
		
		print("Metrics get predictions",np.unique(predictions, return_counts=True))
		print("Metrics get label",np.unique(label, return_counts=True))
		if small_classes_ignore==True:
			important_classes_idx = paramsTrain.known_classes
##			if dataset=='l2':
			#important_classes_idx=[0,1,2,6,8,10,12]
##				important_classes_idx=[0,1,2,6,12] # only for bcknd final value
				#important_classes_idx = [x+1 for x in important_classes_idx]
##				deb.prints(important_classes_idx)
##			elif dataset=='lm':
##				important_classes_idx=[0, 1, 10, 12]
			for idx in range(class_n):
				if idx not in important_classes_idx:
					predictions[predictions==idx]=20
					label[label==idx]=20	
			predictions[predictions==39] = 20
			label[label==39] = 20

		print("After small classes ignore")
		print("Metrics get predictions",np.unique(predictions, return_counts=True))
		print("Metrics get label",np.unique(label, return_counts=True))							
		metrics = {}
		metrics['f1_score']=my_f1_score(label,predictions) # [0.9 0.9 0.4 0.5] [1 2 3 4 5]
		metrics['f1_score_noavg']=f1_score(label,predictions,average=None) # [0.9 0.9 0.4 0.5] [1 2 3 4 5]
		
		metrics['overall_acc']=accuracy_score(label,predictions)
		return metrics


			
	metrics = metrics_get(label_rebuilt, prediction_rebuilt, mask, small_classes_ignore=False)
	print(metrics)
	f = open("metrics_fixed_"+dataset_date+".pkl", "wb")
	pickle.dump(metrics, f)
	f.close()

	metrics = metrics_get(label_rebuilt, prediction_rebuilt, mask, small_classes_ignore=True)
	print(metrics)
	f = open("metrics_fixed_"+dataset_date+"_small_classes_ignore.pkl", "wb")
	pickle.dump(metrics, f)
	f.close()

# bcknd to 255
label_rebuilt = label_rebuilt - 1
prediction_rebuilt = prediction_rebuilt - 1

#if dataset=='lm':
#	important_classes_idx = [0, 1, 10, 12]
important_classes_idx = paramsTrain.known_classes

def small_classes_ignore(label, predictions, important_classes_idx):
	class_n = 15
		
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

label_rebuilt, prediction_rebuilt, important_classes_idx = small_classes_ignore(
			label_rebuilt, prediction_rebuilt,important_classes_idx)

prediction_rebuilt[prediction_rebuilt==39] = 20
label_rebuilt[label_rebuilt==39] = 20

deb.prints(np.unique(label_rebuilt,return_counts=True))
deb.prints(np.unique(prediction_rebuilt,return_counts=True))
#pdb.set_trace()
deb.prints(label_rebuilt.shape)
deb.prints(prediction_rebuilt.shape)
deb.prints(important_classes_idx)

#custom_colormap = custom_colormap[important_classes_idx]

def save_prediction_label_rebuilt_Nto1(label_rebuilt, prediction_rebuilt, mask, 
		sequence_len, custom_colormap, small_classes_ignore=True, name_id=""):
#	for t_step in range(sequence_len):
	label_rebuilt[mask==0]=255
	prediction_rebuilt[mask==0]=255	
	deb.prints(np.unique(label_rebuilt,return_counts=True))
	deb.prints(np.unique(prediction_rebuilt,return_counts=True))


	print("everything outside mask is 255")
	print(np.unique(label_rebuilt,return_counts=True))
	print(np.unique(prediction_rebuilt,return_counts=True))


	# Paint it!


	print(custom_colormap.shape)
	#class_n=custom_colormap.shape[0]
	#=== change to rgb
	print("Gray",prediction_rebuilt.dtype)
	prediction_rgb=np.zeros((prediction_rebuilt.shape+(3,))).astype(np.uint8)
	label_rgb=np.zeros_like(prediction_rgb)
	print("Adding color...")


	prediction_rgb=cv2.cvtColor(prediction_rebuilt,cv2.COLOR_GRAY2RGB)
	label_rgb=cv2.cvtColor(label_rebuilt,cv2.COLOR_GRAY2RGB)

	print("RGB",prediction_rgb.dtype,prediction_rgb.shape)

#	for chan in [0,1,2]:
#		prediction_rgb[...,chan][prediction_rgb[...,chan]==255]=custom_colormap[idx,chan]
#		label_rgb[...,chan][label_rgb[...,chan]==255]=custom_colormap[idx,chan]


	deb.prints(custom_colormap)
	prediction_rgb_tmp = prediction_rgb.copy()
	label_rgb_tmp = label_rgb.copy()
	
	for idx in range(custom_colormap.shape[0]):
		print("Assigning color. class:",idx)

		for chan in [0,1,2]:
			deb.prints(np.unique(label_rgb[...,chan],return_counts=True))

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
	save_folder=dataset+"/"+paramsTrain.model_type+"/"+paramsTrain.seq_date+"/"
	pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
	deb.prints(save_folder)
	threshIdxName = "_TPR" + tpr_threshold_names[pr.threshold_idx]

	if pr.open_set_mode == True:
		prediction_savename = save_folder+"prediction_t_"+paramsTrain.seq_date+"_"+paramsTrain.model_type+"_"+name_id+threshIdxName+".png"
	else:
		prediction_savename = save_folder+"prediction_t_"+paramsTrain.seq_date+"_"+paramsTrain.model_type+"_closedset_"+name_id+"_overl"+str(pr.overlap)+".png"
	ic(prediction_savename)
	print("saving...")
	try:

		os.remove(prediction_savename)
	except:
		print("no file to remove")
	ret = cv2.imwrite(prediction_savename, prediction_rgb)
	deb.prints(ret)
	ic(save_folder+"label_t_"+paramsTrain.seq_date+"_"+paramsTrain.model_type+"_"+name_id+".png")
	ret = cv2.imwrite(save_folder+"label_t_"+paramsTrain.seq_date+"_"+paramsTrain.model_type+"_"+name_id+".png",label_rgb)
	deb.prints(ret)
	ret = cv2.imwrite(save_folder+"mask.png",mask*200)
	deb.prints(ret)



save_prediction_label_rebuilt_Nto1(label_rebuilt, prediction_rebuilt, mask, 
		sequence_len, custom_colormap, small_classes_ignore=True,
		name_id = name_id)

