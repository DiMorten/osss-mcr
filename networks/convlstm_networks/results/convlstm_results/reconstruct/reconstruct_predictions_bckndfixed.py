 
import numpy as np
import cv2
import glob
import argparse

import sys
#sys.path.append('../../../../../train_src/analysis/')
import pathlib
from PredictionsLoader import PredictionsLoaderNPY, PredictionsLoaderModel
parser = argparse.ArgumentParser(description='')
parser.add_argument('-ds', '--dataset', dest='dataset',
					default='cv', help='t len')
parser.add_argument('-mdl', '--model', dest='model',
					default='densenet', help='t len')

a = parser.parse_args()

dataset=a.dataset
model=a.model

direct_execution=True
if direct_execution==True:
	dataset='cv'
	model='unet'



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
	if model=='densenet':
		predictions_path=path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
	elif model=='biconvlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy'
	elif model=='convlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_batch16_full.npy'
	elif model=='unet':
		predictions_path=path+'prediction_BUnet4ConvLSTM_repeating1.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating4.npy'


	elif model=='atrous':
		predictions_path=path+'prediction_BAtrousConvLSTM_2convins5.npy'
	elif model=='atrousgap':
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
	if model=='densenet':
		predictions_path=path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
	elif model=='biconvlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy'
	elif model=='convlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_batch16_full.npy'		
	elif model=='unet':
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy'
		predictions_path=path+'model_best_BUnet4ConvLSTM_int16.h5'
				
	elif model=='atrous':
		predictions_path=path+'prediction_BAtrousConvLSTM_repeating2.npy'			
	elif model=='atrousgap':
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_raulapproved.npy'			
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating4.npy'			
		predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating6.npy'			
	elif model=='unetend':
		predictions_path=path+'prediction_unet_convlstm_temouri2.npy'			
	elif model=='allinputs':
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

print("Loading patch locations...")
order_id_load=False
if order_id_load==False:
	order_id=patch_file_id_order_from_folder(folder_load_path)
	np.save('order_id.npy',order_id)
else:
	order_id=np.load('order_id.npy')

cols=np.load(location_path+'locations_col.npy')
rows=np.load(location_path+'locations_row.npy')

print(cols.shape, rows.shape)
cols=cols[order_id]
rows=rows[order_id]

# ======== load labels and predictions 

#labels=np.load(path+'labels.npy').argmax(axis=4)
#predictions=np.load(predictions_path).argmax(axis=4)

print("Loading labels and predictions...")

prediction_type = 'model'
results_path="../"
#path=results_path+dataset+'/'
#prediction_path=path+predictions_path
path_test='../../../../../dataset/dataset/'+dataset+'_data/patches_bckndfixed/test/'
print('path_test',path_test)

#prediction_type = 'model'
if prediction_type=='npy':
	predictionsLoader = PredictionsLoaderNPY()
	predictions, labels = predictionsLoader.loadPredictions(predictions_path,path+'labels.npy')
elif prediction_type=='model':	
	#model_path=results_path + 'model/'+dataset+'/'+prediction_filename
	print('model_path',predictions_path)

	predictionsLoader = PredictionsLoaderModel(path_test)
	predictions, labels = predictionsLoader.loadPredictions(predictions_path)
	predictions=predictions.argmax(axis=-1)
	labels=labels.argmax(axis=-1)
	

#================= load labels and predictions






class_n=np.max(predictions)+1
print("class_n",class_n)
labels[labels==class_n]=255 # background

# Print stuff
print(cols.shape)
print(rows.shape)
print(labels.shape)
print(predictions.shape)
print("np.unique(labels,return_counts=True)",
	np.unique(labels,return_counts=True))
print("np.unique(predictions,return_counts=True)",
	np.unique(predictions,return_counts=True))

# Specify variables
sequence_len=labels.shape[1]
patch_len=labels.shape[2]

# Load mask
mask=cv2.imread(mask_path,-1)
mask[mask==1]=0 # training as background
print("Mask shape",mask.shape)
#print((sequence_len,)+mask.shape)

# Reconstruct the image
print("Reconstructing the labes and predictions...")
label_rebuilt=np.ones(((sequence_len,)+mask.shape)).astype(np.uint8)*255
prediction_rebuilt=np.ones(((sequence_len,)+mask.shape)).astype(np.uint8)*255
print("label_rebuilt.shape",label_rebuilt.shape)
for row,col,label,prediction in zip(rows,cols,labels,predictions):
	label_rebuilt[:,row:row+patch_len,col:col+patch_len]=label.copy()
	prediction_rebuilt[:,row:row+patch_len,col:col+patch_len]=prediction.copy()

print(np.unique(label_rebuilt,return_counts=True))
print(np.unique(prediction_rebuilt,return_counts=True))


# everything outside mask is 255
for t_step in range(sequence_len):
	label_rebuilt[t_step][mask==0]=255

	prediction_rebuilt[t_step][mask==0]=255
#label_rebuilt[label_rebuilt==class_n]=255
	
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

for t_step in range(sequence_len):
	prediction_rgb[t_step]=cv2.cvtColor(prediction_rebuilt[t_step],cv2.COLOR_GRAY2RGB)
	label_rgb[t_step]=cv2.cvtColor(label_rebuilt[t_step],cv2.COLOR_GRAY2RGB)

print("RGB",prediction_rgb.dtype,prediction_rgb.shape)

for idx in range(custom_colormap.shape[0]):
	print("Assigning color. t_step:",idx)
	for chan in [0,1,2]:
		prediction_rgb[:,:,:,chan][prediction_rgb[:,:,:,chan]==idx]=custom_colormap[idx,chan]
		label_rgb[:,:,:,chan][label_rgb[:,:,:,chan]==idx]=custom_colormap[idx,chan]

print("RGB",prediction_rgb.dtype,prediction_rgb.shape)

#for idx in range(custom_colormap.shape[0]):
#	for chan in [0,1,2]:
#		prediction_rgb[:,:,chan][prediction_rgb[:,:,chan]==correspondence[idx]]=custom_colormap[idx,chan]
print("Saving the resulting images for all dates...")
for t_step in range(sequence_len):

	label_rgb[t_step]=cv2.cvtColor(label_rgb[t_step],cv2.COLOR_BGR2RGB)
	prediction_rgb[t_step]=cv2.cvtColor(prediction_rgb[t_step],cv2.COLOR_BGR2RGB)
	save_folder=dataset+"/"+model+"/"
	pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
	cv2.imwrite(save_folder+"prediction_t"+str(t_step)+"_"+model+".png",prediction_rgb[t_step])
	cv2.imwrite(save_folder+"label_t"+str(t_step)+"_"+model+".png",label_rgb[t_step])

print(prediction_rgb[0,0,0,:])
