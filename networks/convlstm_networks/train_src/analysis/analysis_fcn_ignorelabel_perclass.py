 
import numpy as np

#import cv2
#import h5py
#import scipy.io as sio
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report,recall_score,precision_score
#from sklearn.externals import joblib
import matplotlib.pyplot as plt
#import pandas as pd
from PredictionsLoader import PredictionsLoaderNPY, PredictionsLoaderModel

#====================================
def labels_predictions_filter_transform(label_test,predictions,class_n,
		debug=1):
	predictions=predictions.argmax(axis=np.ndim(predictions)-1)
	predictions=np.reshape(predictions,-1)
	label_test=label_test.argmax(axis=np.ndim(label_test)-1)
	label_test=np.reshape(label_test,-1)
	predictions=predictions[label_test<class_n]

	label_test=label_test[label_test<class_n]
	if debug>0:
		print("Predictions",predictions.shape)
		print("Label_test",label_test.shape)
	return label_test,predictions
def metrics_get(label_test,predictions,only_basics=False,debug=1):
	if debug>0:
		print(predictions.shape,predictions.dtype)
		print(label_test.shape,label_test.dtype)

	metrics={}
	metrics['f1_score']=f1_score(label_test,predictions,average=None)

	if debug>0:
		print("f1_score",metrics['f1_score'])

	if only_basics==False:

		metrics['f1_score_weighted']=f1_score(label_test,predictions,average='weighted')
		        

		metrics['recall']=recall_score(label_test,predictions,average=None)
		metrics['precision']=precision_score(label_test,predictions,average=None)
		if debug>0:
			print(confusion_matrix_.sum(axis=1)[:, np.newaxis].diagonal())
			print(confusion_matrix_.diagonal())
			print(np.sum(confusion_matrix_,axis=1))

			print(metrics)
			print(confusion_matrix_)

			print(metrics['precision'])
			print(metrics['recall'])
	return metrics


#===== normy3
path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy/fcn/seq1/'
#path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy/fcn/seq2/'

#path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/fcn_16/'

#path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/fcn_8/'
#path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/fcn_8/'


#======== normy3_check

path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3_check/fcn/seq1/'
#=========== normy3_check2
path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy3_check2/fcn/'
path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3_check2/fcn/seq2/'
path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3_check2/fcn/seq1/'

# gold

path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3_check2/fcn/seq2/gold/'
# ====== normy3B

#path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3B/fcn/seq1/'
#path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3B/fcn/seq2/'


path='/home/lvc/Jorg/sbsr/fcn_model/keras_time_semantic_fcn/'
# ======= convlstm playground

path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/convlstm_playground/fcn_original500/'
path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/convlstm_playground/fcn_original/'

# === tranfer

path='/home/lvc/Jorg/igarss/fcn_transfer_learning_for_RS/results/transfer_fcn_seq2_to_seq1/'
# === normy3 check

path='/home/lvc/Jorg/igarss/fcn_transfer_learning_for_RS/results/normy3_check/seq1/fcn/'



# ======== ConvRNN

path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/cv/densenet/'
prediction_path=path+'prediction.npy'

#prediction_path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/cv/prediction_ConvLSTM_DenseNet_eyesight.npy'

# =========seq2seq 
def experiment_analyze(dataset='cv',
		prediction_filename='prediction_DenseNetTimeDistributed_blockgoer.npy',
		prediction_type='npy', mode='each_date',debug=1):
	#path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/seq2seq_ignorelabel/'+dataset+'/'
	base_path="../../results/convlstm_results/"
	path=base_path+dataset+'/'
	prediction_path=path+prediction_filename
	path_test='../../../../dataset/dataset/'+dataset+'_data/patches_bckndfixed/test/'
	print('path_test',path_test)
	
	#prediction_type = 'model'
	if prediction_type=='npy':
		predictionsLoader = PredictionsLoaderNPY()
		predictions, label_test = predictionsLoader.loadPredictions(prediction_path,path+'labels.npy')
	elif prediction_type=='model':	
		model_path=base_path + 'model/'+dataset+'/'+prediction_filename
		print('model_path',model_path)

		predictionsLoader = PredictionsLoaderModel(path_test)
		predictions, label_test = predictionsLoader.loadPredictions(model_path)


#		mode='each_date',debug=1):
#	path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/seq2seq_ignorelabel/'+dataset+'/'

#	prediction_path=path+prediction_filename
#	predictions=np.load(prediction_path)
#	label_test=np.load(path+'labels.npy')
#	if debug>0:
#		print(predictions.shape)
#		print(label_test.shape)
	class_n=predictions.shape[-1]

	if mode=='each_date':
		metrics_t={'f1_score':[],'overall_acc':[],
			'average_acc':[]}
		label_test_v=label_test.argmax(axis=4).flatten()
		label_test_v=label_test_v[label_test_v<class_n]

		label_unique=np.unique(label_test_v)
		print("label_unique",label_unique)
		labels_unique_t=[]
		for t in range(label_test.shape[1]):
			predictions_t = predictions[:,t,:,:,:]
			label_test_t = label_test[:,t,:,:,:]

			label_test_t,predictions_t = labels_predictions_filter_transform(
				label_test_t, predictions_t, class_n=class_n,
				debug=debug)
			print("predictions_t",np.unique(
				predictions_t).shape)
			print("label_test_t",np.unique(
				label_test_t).shape)

			label_unique_t=np.unique(label_test_t)
			predictions_unique_t=np.unique(predictions_t)
			classes_t = np.unique(np.concatenate((label_unique_t,predictions_unique_t),0))
			##print("classes_t.shape",classes_t.shape)
			metrics = metrics_get(label_test_t, predictions_t,
				only_basics=True, debug=debug)	
			##print("metrics['f1_score'].shape",metrics['f1_score'].shape)
			#metrics_t['f1_score'].append(metrics['f1_score'])
			#metrics_t['overall_acc'].append(metrics['overall_acc'])
			metrics_ordered={'f1_score':np.zeros(label_unique.shape)}
			valid_classes_counter=0
			##print(metrics_ordered['f1_score'])
			for clss in range(label_unique.shape[0]):
				#print(clss)
				if np.any(classes_t==clss): # If this timestep t has class clss
					##print("1",valid_classes_counter)
					##print("2",classes_t[valid_classes_counter])
					##print("3",metrics['f1_score'][valid_classes_counter])
					
					metrics_ordered['f1_score'][clss]=metrics['f1_score'][valid_classes_counter]
					valid_classes_counter+=1
				if np.any(label_unique_t==clss):
					pass
				else:
					metrics_ordered['f1_score'][clss]=np.nan

			metrics_t['f1_score'].append(metrics_ordered['f1_score'])
			labels_unique_t.append(label_unique_t)
			print("class_n",t,metrics['f1_score'].shape)

		print(metrics_t)
		return metrics_t
	elif mode=='global':
		
		label_test,predictions=labels_predictions_filter_transform(
			label_test,predictions, class_n=class_n)

		print(np.unique(predictions,return_counts=True))
		print(np.unique(label_test,return_counts=True))

		metrics=metrics_get(label_test,predictions)

		return metrics

def experiments_analyze(dataset,experiment_list,mode='each_date'):
	experiment_metrics=[]
	for experiment in experiment_list:
		print("Starting experiment:",experiment)
		experiment_metrics.append(experiment_analyze(
			dataset=dataset,
			prediction_filename=experiment,
			mode=mode,debug=0))
	return experiment_metrics

def experiment_groups_analyze(dataset,experiment_group,mode='each_date'):
	save=True
	if save==True:	

		experiment_metrics=[]
		for group in experiment_group:
			group_metrics=[]
			for experiment in group:
				print("======determining prediction type")

				if experiment[-3:]=='npy':
					prediction_type='npy'
				elif experiment[-2:]=='h5':
					prediction_type='model'

				print("Starting experiment: {}. Prediction type: {}".format(experiment,prediction_type))

				group_metrics.append(experiment_analyze(
					dataset=dataset,
					prediction_filename=experiment,
					mode=mode,debug=0,
					prediction_type=prediction_type))
			experiment_metrics.append(group_metrics)

	#	for model_id in range(len(experiment_metrics[0])):
	#		for date_id in range(len(experiment_metrics[0][model_id]):

		np.save('experiment_metrics.npy',experiment_metrics)
	else:
		experiment_metrics=np.load('experiment_metrics.npy')
	metrics={}
	total_metrics=[]

	for exp_id in range(len(experiment_metrics[0])):
		exp_id=int(exp_id)
		print(len(experiment_metrics))
		print(len(experiment_metrics[0]))
		print(experiment_metrics[0][0])
		print(experiment_metrics[0][0]['f1_score'])
		#print(experiment_metrics[1][0]['f1_score'])

		print("exp_id",exp_id)

		#[experiment_metrics[int(i)][exp_id]['f1_score'] for i in range(len(experiment_metrics))]
#		a=[experiment_metrics[int(i)][exp_id]['f1_score'] for i in range(len(experiment_metrics))]
#		print("1",a)
#		print("2",np.array(a))
#		print("3",np.array(a).shape)
#		print("4",a[0][0])
		metrics_by_date=[]
		for date_id in range(len(experiment_metrics[0][0])):
			date_id=int(date_id)
			print("1",len(experiment_metrics[0]))
			print("2",len(experiment_metrics[0][0]))
			print("3",len(experiment_metrics[0][0]['f1_score'][0])) # class f1 score for the first date
			print("4",experiment_metrics[0][exp_id]['f1_score'][date_id].shape) # class f1 score for the first date
			print("4",experiment_metrics[1][exp_id]['f1_score'][date_id].shape) # class f1 score for the first date
			
			metrics_by_group=[experiment_metrics[int(i)][exp_id]['f1_score'][date_id] for i in range(len(experiment_metrics))]

			print("1",np.asarray(metrics_by_group).shape)
			print("1",metrics_by_group)
			print("2",len(metrics_by_group))
			print("3",metrics_by_group[0].shape)
			metrics_by_group=np.stack((metrics_by_group[0],metrics_by_group[1]))
			print("concatenated metrics_by_group.shape",metrics_by_group.shape)
			metrics_average=np.average(metrics_by_group,axis=0)
			print("metrics_average.shape",metrics_average.shape)
			metrics_by_date.append(metrics_average)
		print("len(metrics_by_date)",len(metrics_by_date))




		metrics['f1_score']=np.average(np.array(
			[experiment_metrics[int(i)][exp_id]['f1_score'] for i in range(len(experiment_metrics))]),
			axis=0)
		#print("metrics f1 score",metrics['f1_score'])
		#metrics['overall_acc']=np.average(np.asarray(
		#	[experiment_metrics[int(i)][exp_id]['overall_acc'] for i in range(len(experiment_metrics))]),
		#	axis=0)

		#metrics['average_acc']=np.average(np.asarray(
		#	[experiment_metrics[int(i)][exp_id]['average_acc'] for i in range(len(experiment_metrics))]),
		#	axis=0)
		#total_metrics.append(metrics.copy())
		print("total metrics f1 score",metrics)
		if dataset=='cv':
			important_classes=[0,1,2,3,4,7,9]
			important_dates=[0,2,4,5,6,8,10,11,13]
		elif dataset=='lm':
			important_classes=[0,1,2,3,4,5,6,7]
			important_dates=range(metrics['f1_score'].shape[0])
		print("metrics['f1_score'].shape",metrics['f1_score'].shape)
		metrics['f1_score']=metrics['f1_score'][:,important_classes]
		print("metrics['f1_score'].shape",metrics['f1_score'].shape)
		metrics['f1_score']=metrics['f1_score'][important_dates,:]
		print("metrics['f1_score'].shape",metrics['f1_score'].shape)
		
#		print("metrics['f1_score'].shape",metrics['f1_score'].shape)
		metrics['f1_score']*=100
		print("Exp id",experiment_group[0][exp_id])
		np.savetxt(
			"averaged_metrics_"+dataset+"_"+experiment_group[0][exp_id]+".csv",
			np.transpose(metrics['f1_score']), delimiter=",",fmt='%1.1f')
	print("metrics['f1_score'].shape",metrics['f1_score'].shape)
	print("total merics len",len(metrics))
	#print(total_metrics)
	return metrics

def experiments_plot(metrics,experiment_list,dataset):


	print(metrics)
	t_len=len(metrics[0]['f1_score'])
	print("t_len",t_len)
	indices = range(t_len) # t_len
	X = np.arange(t_len)
	exp_id=0
	width=0.5
	colors=['b','y','c','m','r']
	#colors=['#7A9AAF','#293C4B','#FF8700']
	#colors=['#4225AC','#1DBBB9','#FBFA17']
	##colors=['b','#FBFA17','c']
	#colors=['#966A51','#202B3F','#DA5534']
	exp_handler=[] # here I save the plot for legend later
	exp_handler2=[] # here I save the plot for legend later
	exp_handler3=[] # here I save the plot for legend later

	figsize=(8,4)
	fig, ax = plt.subplots(figsize=figsize)
	#fig2, ax2 = plt.subplots(figsize=figsize)
	#fig3, ax3 = plt.subplots(figsize=figsize)

	fig.subplots_adjust(bottom=0.2)
	#fig2.subplots_adjust(bottom=0.2)
	#fig3.subplots_adjust(bottom=0.2)

	for experiment in experiment_list:
		print("experiment",experiment)
		print(exp_id)
		metrics[exp_id]['f1_score']=np.transpose(np.asarray(metrics[exp_id]['f1_score']))
		#metrics[exp_id]['overall_acc']=np.transpose(np.asarray(metrics[exp_id]['overall_acc']))
		#metrics[exp_id]['average_acc']=np.transpose(np.asarray(metrics[exp_id]['average_acc']))

		exp_handler.append(ax.bar(X + float(exp_id)*width/2, 
			metrics[exp_id]['f1_score'], 
			color = colors[exp_id], width = width/2))
		ax.set_title('Average F1 Score')
		ax.set_xlabel('Epoch')
		if dataset=='lm': 
			ax.set_xlim(-0.5,13)
		elif dataset=='cv': 
			ax.set_xlim(-0.5,14)
		
		exp_id+=1
	
	ax.legend(tuple(exp_handler), ('ConvLSTM-PL','BiConvLSTM-PL','RDenseNet-PL'),loc='lower center', bbox_to_anchor=(0.5, -0.29), shadow=True, ncol=3)
	ax2.legend(tuple(exp_handler2), ('ConvLSTM-PL','BiConvLSTM-PL','RDenseNet-PL'),loc='lower center', bbox_to_anchor=(0.5, -0.29), shadow=True, ncol=3)

	ax3.legend(tuple(exp_handler3), ('ConvLSTM-PL','BiConvLSTM-PL','RDenseNet-PL'),loc='lower center', bbox_to_anchor=(0.5, -0.29), shadow=True, ncol=3)

	fig.savefig("f1_score_"+dataset+".png")
	fig2.savefig("average_acc_"+dataset+".png")
	fig3.savefig("overall_acc_"+dataset+".png")

	#plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)
	#plt.bar(X + 0.25, data[1], color = 'g', width = 0.25)
	#plt.bar(X + 0.50, data[2], color = 'r', width = 0.25)


	#plot_metric=[x['f1_score'] for x in metrics]
	#print(plot_metric)
	
	plt.show()

dataset='cv'
load_metrics=False
#mode='global'
mode='each_date'
if dataset=='cv':
# 	experiment_groups=[[
# 		'prediction_ConvLSTM_seq2seq_batch16_full.npy',
# 		'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
# ##		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
# 		'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'],

# 		['prediction_ConvLSTM_seq2seq_redoing.npy',
# 		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
# 		'prediction_DenseNetTimeDistributed_128x2_redoing.npy'],

# 		['prediction_ConvLSTM_seq2seq_redoingz.npy',
# 		'prediction_ConvLSTM_seq2seq_bi_redoingz.npy',
# 		'prediction_DenseNetTimeDistributed_128x2_redoingz.npy'],
# 		['prediction_ConvLSTM_seq2seq_redoingz2.npy',
# 		'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
# 		'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy']]
	experiment_groups=[[
		'prediction_ConvLSTM_seq2seq_batch16_full.npy',
		'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
		'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy',
		'prediction_BUnet4ConvLSTM_repeating1.npy',
		'prediction_BAtrousGAPConvLSTM_raulapproved.npy',
		],

		['prediction_ConvLSTM_seq2seq_redoing.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoing.npy',
		'prediction_BUnet4ConvLSTM_repeating1.npy',
		'prediction_BAtrousGAPConvLSTM_raulapproved.npy',

		],
		['prediction_ConvLSTM_seq2seq_redoingz.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoingz.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy',
		'prediction_BUnet4ConvLSTM_repeating2.npy',
		'prediction_BAtrousGAPConvLSTM_repeating3.npy',

		],
		['prediction_ConvLSTM_seq2seq_redoingz2.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoing3.npy',
		'prediction_BUnet4ConvLSTM_repeating4.npy',
		'prediction_BAtrousGAPConvLSTM_repeating4.npy',
		]]
	experiment_groups=[['model_best_BUnet4ConvLSTM_windows_test.h5'],
		['model_best_BUnet4ConvLSTM_windows_test.h5']]
##		'prediction_DenseNetTimeDistributed_128x2_redoing.npy']
		##'prediction_ConvLSTM_seq2seq_loneish.npy',
		##'prediction_ConvLSTM_seq2seq_bi_loneish.npy',
		#'prediction_ConvLSTM_seq2seq_bi_60x2_loneish.npy',
		#'prediction_FCN_ConvLSTM_seq2seq_bi_skip_loneish.npy',
		#'prediction_DenseNetTimeDistributed_blockgoer.npy',
		#'prediction_DenseNetTimeDistributed_128x2_filtersizefix2.npy']
elif dataset=='lm':

	# experiment_groups=[[
	# 	'prediction_ConvLSTM_seq2seq_batch16_full.npy',
	# 	'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
	# 	'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'],
	# 	['prediction_ConvLSTM_seq2seq_redoing.npy',
	# 	'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
	# 	'prediction_DenseNetTimeDistributed_128x2_redoing.npy'],
	# 	['prediction_ConvLSTM_seq2seq_redoingz.npy',
	# 	'prediction_ConvLSTM_seq2seq_bi_redoingz.npy',
	# 	'prediction_DenseNetTimeDistributed_128x2_redoingz.npy'],
	# 	['prediction_ConvLSTM_seq2seq_redoingz2.npy',
	# 	'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
	# 	'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy']]
	experiment_groups=[[
			'prediction_ConvLSTM_seq2seq_batch16_full.npy',
			'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
			'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy',
			'prediction_BUnet4ConvLSTM_repeating1.npy',
			'prediction_BAtrousGAPConvLSTM_raulapproved.npy',
			],
			['prediction_ConvLSTM_seq2seq_redoing.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
			'prediction_DenseNetTimeDistributed_128x2_redoing.npy',
			'prediction_BUnet4ConvLSTM_repeating2.npy',
			'prediction_BAtrousGAPConvLSTM_repeating6.npy',
			],
			['prediction_ConvLSTM_seq2seq_redoingz.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoingz.npy',
			'prediction_DenseNetTimeDistributed_128x2_redoingz.npy',
			'prediction_BUnet4ConvLSTM_repeating4.npy',
			'prediction_BAtrousGAPConvLSTM_repeating4.npy',
			],
			['prediction_ConvLSTM_seq2seq_redoingz2.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
			'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy',
			'prediction_BUnet4ConvLSTM_repeating6.npy',
			'prediction_BAtrousGAPConvLSTM_repeating3.npy',
			],]
if load_metrics==False:
	experiment_metrics=experiment_groups_analyze(dataset,experiment_groups,
		mode=mode)
	np.save("experiment_metrics_"+dataset+".npy",experiment_metrics)

else:
	experiment_metrics=np.load("experiment_metrics_"+dataset+".npy")

if mode=='each_date':
	experiments_plot(experiment_metrics,experiment_groups[0],dataset)

#metrics['per_class_acc'][~np.isnan(metrics['per_class_acc'])]


