
import numpy as np

#import cv2
#import h5py
#import scipy.io as sio
import numpy as np
import scipy
import glob
import os, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report,recall_score,precision_score
#from sklearn.externals import joblib
import matplotlib.pyplot as plt
#import pandas as pd
import cv2
import pdb
file_id="importantclasses"

from PredictionsLoader import PredictionsLoaderNPY, PredictionsLoaderModel, PredictionsLoaderModelNto1, PredictionsLoaderModelNto1FixedSeqFixedLabel, PredictionsLoaderModelNto1FixedSeqFixedLabelAdditionalTestClsses, PredictionsLoaderModelNto1FixedSeqFixedLabelOpenSet
from colorama import init
init()
save_bar_flag=True
sys.path.append('../')
import deb
from open_set import SoftmaxThresholding, OpenPCS
import argparse
from parameters.parameters_reader import ParamsTrain, ParamsAnalysis

paramsTrain = ParamsTrain('../parameters/')
paramsAnalysis = ParamsAnalysis('parameters_analysis/')

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--seq_date', dest='seq_date', 
                    default='jun',
                    help='seq_date')
parser.add_argument('--dataset', dest='dataset', 
                    default='l2',
                    help='dataset')
parser.add_argument('--model_dataset', dest='model_dataset', 
                    default='l2',
                    help='model_dataset')

args = parser.parse_args()
#====================================
def dense_crf(probs, img=None, n_iters=10, n_classes=19,
			  sxy_gaussian=(1, 1), compat_gaussian=4,
			  sxy_bilateral=(49, 49), compat_bilateral=5,
			  srgb_bilateral=(13, 13, 13)):
	import pydensecrf.densecrf as dcrf
	from pydensecrf.utils import create_pairwise_bilateral
	_, h, w, _ = probs.shape

	probs = probs[0].transpose(2, 0, 1).copy(order='C')	 # Need a contiguous array.

	d = dcrf.DenseCRF2D(w, h, n_classes)  # Define DenseCRF model.
	U = -np.log(probs)	# Unary potential.
	U = U.reshape((n_classes, -1))	# Needs to be flat.
	d.setUnaryEnergy(U)
	d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
						  kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	if img is not None:
		assert (img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
		pairwise_bilateral = create_pairwise_bilateral(sdims=(10,10), schan=(0.01), img=img[0], chdim=2)
		d.addPairwiseEnergy(pairwise_bilateral, compat=10)

	Q = d.inference(n_iters)
	preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
	return np.expand_dims(preds, 0)


def labels_predictions_filter_transform(label_test,predictions,test_pred_proba, class_n,
		debug=1,small_classes_ignore=True,
		important_classes=None,dataset='cv',skip_crf=False,t=None, predictionsLoaderTest=None,
		label_train=None, predictions_train=None, train_pred_proba=None,
		threshold = None, openModel = None):

	if not skip_crf:
		# CRF
		for i,v in enumerate(predictions):
			img_in = imgs_in[i][t]
			img_in = np.array(img_in, dtype=np.uint8)
			img_in = np.expand_dims(img_in, axis=0)
			v = scipy.special.softmax(v, axis=-1)
			v = np.expand_dims(v, axis=0)
			predictions[i] = dense_crf(v,img=img_in,n_iters=10,sxy_gaussian=(3, 3), compat_gaussian=3,n_classes=class_n)


#			openModel = OpenPCS(loco_class = predictionsLoader.loco_class)
#	openModel = SoftmaxThresholding(loco_class = predictionsLoader.loco_class)
	#open_set_flag = False
	specify_unknown_classes = False
	if paramsAnalysis.open_set == True:
		if specify_unknown_classes==True:
			all_classes = np.unique(label_test)
			all_classes = all_classes[1:] - 1 # no bcknd
			deb.prints(all_classes)
			deb.prints(paramsTrain.unknown_classes)


			paramsTrain.known_classes = np.setdiff1d(all_classes, paramsTrain.unknown_classes)
			#known_classes = [x + 1 for x in known_classes]

			#pdb.set_trace()
#			known_classes.remove(predictionsLoaderTest.loco_class + 1)
#			known_classes.remove(0) #background
		#else:
		known_classes = [x + 1 for x in paramsTrain.known_classes]
		deb.prints(known_classes)
		
		if openModel == None:
			openModel = OpenPCS(loco_class = predictionsLoaderTest.loco_class,  known_classes = known_classes,
					n_components = 16)
				

		

		#loop_threshold_flag = False
		#if loop_threshold_flag == False:
		openModel.setThreshold(threshold)
		if openModel.fittedFlag == False:
			openModel.fit(label_train, predictions_train, train_pred_proba)
#		openModel.fit(label_test, predictions, test_pred_proba)
		predictions = openModel.predict(label_test, predictions, test_pred_proba)


	#predictions=predictions.argmax(axis=-1)
	predictions=np.reshape(predictions,-1)
	#label_test=label_test.argmax(axis=-1)
	label_test=np.reshape(label_test,-1)


	deb.prints(np.unique(predictions,return_counts=True))
	deb.prints(np.unique(label_test,return_counts=True))

	translate_mode = True
	deb.prints(translate_mode)
	if translate_mode == False:
		bcknd_id = np.unique(label_test)[-1]
		deb.prints(bcknd_id)
		predictions=predictions[label_test<bcknd_id]
		label_test=label_test[label_test<bcknd_id]
	else:
		predictions=predictions[label_test!=0]
		label_test=label_test[label_test!=0]	
		predictions = predictions - 1
		label_test = label_test - 1

	deb.prints(np.unique(predictions,return_counts=True))
	deb.prints(np.unique(label_test,return_counts=True))


	print("========================= Flattened the predictions and labels")	
	print("Loaded predictions unique: ",np.unique(predictions,return_counts=True))
	print("Loaded label test unique: ",np.unique(label_test,return_counts=True))
	
	print("Loaded predictions shape: ",predictions.shape)
	print("Loaded label test shape: ",label_test.shape)

	# map small classes to single class 20
	if small_classes_ignore==True:
		# Eliminate non important classes
		class_list,class_count = np.unique(label_test,return_counts=True)
		if debug>=0: print("Class unique before eliminating non important classes:",class_list,class_count)
		
		if dataset=='cv':
			important_classes_idx=[0,1,2,6,7,8]
		elif dataset=='lm':
			important_classes_idx=[0,1,2,6,8,10,12]
			important_classes_idx=[0,1,6,10,12]
		elif dataset=='l2':
#			important_classes_idx=[0,1,2,6,8,10,12]
#			important_classes_idx=[0,1,2,6,10,12]

#			important_classes_idx=[0,1,2,6,12]
			important_classes_idx=[0,1,6,10,12] #soybean, maize, cerrado, soil, millet
#jun			2,3,7,13
			
#			important_classes_idx=[0,1,4,5,11]

		mode=3
		if mode==1:
			for idx in range(class_n):
				if idx in important_classes_idx and idx in class_list:
					index=int(np.where(class_list==idx)[0])
					if class_count[index]<15000:
						predictions[predictions==idx]=20
						label_test[label_test==idx]=20
				else:
					predictions[predictions==idx]=20
					label_test[label_test==idx]=20
		elif mode==2:
			class_count_min=100000
			important_classes_class_count_min=15000
			#important_classes_class_count_min=1

			#print("Class count min:",class_count_min)

			for idx in range(class_n):
				if idx in class_list:
					class_count_min_idx = important_classes_class_count_min if idx in important_classes_idx else class_count_min
					index=int(np.where(class_list==idx)[0])
					#print("b",index)
					if class_count[index]<class_count_min_idx:
						predictions[predictions==idx]=20
						label_test[label_test==idx]=20
				else:
					predictions[predictions==idx]=20
					label_test[label_test==idx]=20
		elif mode==3: # Just take the important classes, no per-date analysis
			for idx in range(class_n):
				if idx not in important_classes_idx:
					predictions[predictions==idx]=20
					label_test[label_test==idx]=20




		if debug>=0: print("Class unique after eliminating non important classes:",np.unique(label_test,return_counts=True))
		print("Pred unique after eliminating non important classes:",np.unique(predictions,return_counts=True))


	if debug>0:
		print("Predictions",predictions.shape)
		print("Label_test",label_test.shape)
	return label_test,predictions, openModel
def my_f1_score(label,prediction):
	f1_values=f1_score(label,prediction,average=None)

	#label_unique=np.unique(label) # [0 1 2 3 5]
	#prediction_unique=np.unique(prediction.argmax(axis-1)) # [0 1 2 3 4]
	#[ 0.8 0.8 0.8 0 0.7 0.7]

	f1_value=np.sum(f1_values)/len(np.unique(label))

	#print("f1_values",f1_values," f1_value:",f1_value)
	return f1_value
def metrics_get(label_test,predictions,only_basics=False,debug=1, detailed_t=None):
	if debug>0:
		print(predictions.shape,predictions.dtype)
		print(label_test.shape,label_test.dtype)

	metrics={}

	#metrics['f1_score']=f1_score(label_test,predictions,average='macro')	
	metrics['f1_score']=my_f1_score(label_test,predictions) # [0.9 0.9 0.4 0.5] [1 2 3 4 5]
	metrics['f1_score_noavg']=f1_score(label_test,predictions,average=None) # [0.9 0.9 0.4 0.5] [1 2 3 4 5]
	
	metrics['overall_acc']=accuracy_score(label_test,predictions)
	confusion_matrix_=confusion_matrix(label_test,predictions)
	#print(confusion_matrix_)
	metrics['per_class_acc']=(confusion_matrix_.astype('float') / confusion_matrix_.sum(axis=1)[:, np.newaxis]).diagonal()
	acc=confusion_matrix_.diagonal()/np.sum(confusion_matrix_,axis=1)
	acc=acc[~np.isnan(acc)]
	metrics['average_acc']=np.average(metrics['per_class_acc'][~np.isnan(metrics['per_class_acc'])])
	if debug>0:
		print("acc",metrics['per_class_acc'])
		print("Acc",acc)
		print("AA",np.average(acc))
		print("OA",np.sum(confusion_matrix_.diagonal())/np.sum(confusion_matrix_))
		print("AA",metrics['average_acc'])
		print("OA",metrics['overall_acc'])

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
	#if detailed_t==6:
	print(confusion_matrix_)

	return metrics


# =========seq2seq 
def experiment_analyze(small_classes_ignore,dataset='cv',
		prediction_filename='prediction_DenseNetTimeDistributed_blockgoer.npy',
		prediction_type='npy', mode='each_date',debug=1,model_n=0):
	#path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/seq2seq_ignorelabel/'+dataset+'/'
	base_path="../../results/convlstm_results/"
	path=base_path+dataset+'/'
	prediction_path=path+prediction_filename
	path_test='../../../../dataset/dataset/'+dataset+'_data/patches_bckndfixed/test/'
	path_train='../../../../dataset/dataset/'+dataset+'_data/patches_bckndfixed/train/'
	
	print('path_test',path_test)
	
	#prediction_type = 'model'
	if prediction_type=='npy':
		predictionsLoader = PredictionsLoaderNPY()
		predictions, label_test = predictionsLoader.loadPredictions(prediction_path,path+'labels.npy')
	elif prediction_type=='model':	
		model_path=base_path + 'model/'+dataset+'/'+prediction_filename
		print('model_path',model_path)

		#predictionsLoader = PredictionsLoaderModel(path_test)
		#predictionsLoader = PredictionsLoaderModelNto1(path_test)
		additionalTestClssesFlag = False
		if additionalTestClssesFlag==True:
			additionalTestClsses = ['dec', 'jan', 'mar', 'may', 'aug']
			if args.seq_date in additionalTestClsses:
				predictionsLoader = PredictionsLoaderModelNto1FixedSeqFixedLabelAdditionalTestClsses(path_test, dataset=dataset)
			else:
				predictionsLoader = PredictionsLoaderModelNto1FixedSeqFixedLabel(path_test, dataset=dataset)
			deb.prints(args.seq_date in additionalTestClsses)
		else:
#			predictionsLoader = PredictionsLoaderModelNto1FixedSeqFixedLabel(path_test, dataset=dataset)
#			predictionsLoader = PredictionsLoaderModelNto1FixedSeqFixedLabelOpenSet(path_test, dataset=dataset, loco_class=8)
			predictionsLoaderTrain = PredictionsLoaderModelNto1FixedSeqFixedLabelOpenSet(path_train, dataset=dataset)
			predictionsLoaderTest = PredictionsLoaderModelNto1FixedSeqFixedLabelOpenSet(path_test, dataset=dataset)


		deb.prints(predictionsLoaderTest)

		predictions, label_test, test_pred_proba, model = predictionsLoaderTest.loadPredictions(model_path, seq_date=args.seq_date, 
				model_dataset=args.model_dataset)

		if paramsAnalysis.open_set==True:
#			predictions_train, label_train, train_pred_proba, _ = predictionsLoaderTrain.loadPredictions(model_path, seq_date=args.seq_date, 
#					model_dataset=args.model_dataset)
			predictions_train, label_train, train_pred_proba = predictions.copy(), label_test.copy(), test_pred_proba.copy()
			deb.prints(np.unique(np.concatenate((predictions,label_test),axis=0)))
		else:
			predictions_train, label_train, train_pred_proba = None, None, None	
	#predictions=np.load(prediction_path, allow_pickle=True)
	#label_test=np.load(path+'labels.npy', allow_pickle=True)

	
	print("Loaded predictions unique: ",np.unique(predictions,return_counts=True))
	print("Loaded label test unique: ",np.unique(label_test,return_counts=True))
	
	print("Loaded predictions shape: ",predictions.shape)
	print("Loaded label test shape: ",label_test.shape)

	prediction_unique,prediction_count = np.unique(predictions,return_counts=True)
	label_test_unique,label_test_count = np.unique(label_test,return_counts=True)
	print(np.sum(prediction_count[:]))
	print(np.sum(label_test_count[:-1]))
	
	#pdb.set_trace()
	class_n=predictions.shape[-1]
	mode='each_date'
	skip_crf=True
	if mode=='each_date':
		metrics_t={'f1_score':[],'overall_acc':[],
			'average_acc':[]}

		# if dataset=='cv':
		# 	important_classes=[]
		# 	for date in range(14):
		# 		if date<=7:
		# 			date_important_classes=[0,6,8]


#		for t in range(label_test.shape[1]):
#		thresholds = [-200, -100, -50, 0, 50, 100, 200, 400]
		#thresholds = [-100, -50, 0]
		thresholds = np.linspace(-500, 500, 20)
#		thresholds = [131.57]
#		thresholds = [400]
		thresholds = [-5000]
		thresholds = [-100, 0]
		thresholds = [-250]

		


		t=0
		openModel = None
		for threshold in thresholds:
			predictions_t = predictions.copy()
			label_test_t = label_test.copy()
			#skip_crf = model_n<2 #prediction_filename.startswith('model_best_BUnet4ConvLSTM_128fl_')
			print("###skip_crf###")
			print(skip_crf)
			print(prediction_filename)
		

			label_test_t,predictions_t, openModel = labels_predictions_filter_transform(
				label_test_t, predictions_t, test_pred_proba, class_n=class_n,
				debug=debug,small_classes_ignore=small_classes_ignore,
				important_classes=None, dataset=dataset, skip_crf=skip_crf, t=t,
#				predictionsLoaderTest = predictionsLoaderTest, label_train=label_train,
#				predictions_train=predictions_train, train_pred_proba=train_pred_proba)
				predictionsLoaderTest = predictionsLoaderTest, label_train=label_train,
				predictions_train=predictions_train, train_pred_proba=train_pred_proba,
				threshold = threshold, openModel = openModel)


			metrics = metrics_get(label_test_t, predictions_t,
				only_basics=True, debug=debug, detailed_t = t)	
			print(metrics)
	#		pdb.set_trace()
			metrics_t['f1_score'].append(round(metrics['f1_score']*100,2))
			metrics_t['overall_acc'].append(round(metrics['overall_acc']*100,2))
			metrics_t['average_acc'].append(round(metrics['average_acc']*100,2))
			print(args.seq_date)
		deb.prints(thresholds)
		print(metrics_t)
		sys.exit("fixed label analysis finished")
		#pdb.set_trace()
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

def experiment_groups_analyze(dataset,experiment_group,
	small_classes_ignore,mode='each_date',exp_id=1):
	save=True
	if save==True:	
		experiment_metrics=[]
		for group in experiment_group:
			group_metrics=[]
			for i,experiment in enumerate(group):
				print("Starting experiment:",experiment)

				print("======determining prediction type")

				if experiment[-3:]=='npy':
					prediction_type='npy'
				elif experiment[-2:]=='h5':
					prediction_type='model'

				print("Starting experiment: {}. Prediction type: {}".format(experiment,prediction_type))


				group_metrics.append(experiment_analyze(
					dataset=dataset,
					prediction_filename=experiment,
					mode=mode,debug=0,model_n=i,
					small_classes_ignore=small_classes_ignore,
					prediction_type=prediction_type))
			experiment_metrics.append(group_metrics)

	#	for model_id in range(len(experiment_metrics[0])):
	#		for date_id in range(len(experiment_metrics[0][model_id]):

		np.save('experiment_metrics'+str(exp_id)+'.npy',
			experiment_metrics)
	else:
		experiment_metrics=np.load(
			'experiment_metrics'+str(exp_id)+'.npy')
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
		metrics['f1_score']=np.average(np.asarray(
			[experiment_metrics[int(i)][exp_id]['f1_score'] for i in range(len(experiment_metrics))]),
			axis=0)
		
		print("metrics f1 score",metrics['f1_score'])
		metrics['overall_acc']=np.average(np.asarray(
			[experiment_metrics[int(i)][exp_id]['overall_acc'] for i in range(len(experiment_metrics))]),
			axis=0)

		metrics['average_acc']=np.average(np.asarray(
			[experiment_metrics[int(i)][exp_id]['average_acc'] for i in range(len(experiment_metrics))]),
			axis=0)
		total_metrics.append(metrics.copy())
		print("total metrics f1 score",total_metrics)

	print("metrics['f1_score'].shape",metrics['f1_score'].shape)
	print("total merics len",len(total_metrics))
	print(total_metrics)
	return total_metrics

def experiments_plot(metrics,experiment_list,dataset,
	experiment_id,small_classes_ignore=False):



	if dataset=='cv':
		valid_dates=[0,2,4,5,6,8,10,11,13]
		t_len=len(valid_dates)
	else:
		t_len=len(metrics[0]['f1_score'])

	print("t_len",t_len)
	indices = range(t_len) # t_len
	X = np.arange(t_len)

	exp_id=0
	#width=0.5
	width=0.25
	
	colors=['b','y','c','g','m','r','b','y']
	colors=['b','g','r','c','m','y','b','g']
	#colors=['#7A9AAF','#293C4B','#FF8700']
	#colors=['#4225AC','#1DBBB9','#FBFA17']
	##colors=['b','#FBFA17','c']
	#colors=['#966A51','#202B3F','#DA5534']
	exp_handler=[] # here I save the plot for legend later
	exp_handler2=[] # here I save the plot for legend later
	exp_handler3=[] # here I save the plot for legend later

	figsize=(8,4)
	fig, ax = plt.subplots(figsize=figsize)
	fig2, ax2 = plt.subplots(figsize=figsize)
	fig3, ax3 = plt.subplots(figsize=figsize)

	fig.subplots_adjust(bottom=0.2)
	fig2.subplots_adjust(bottom=0.2)
	fig3.subplots_adjust(bottom=0.2)
	#metrics=metrics[]
	print("Plotting")
	for experiment in experiment_list:
		if save_bar_flag==True:
			print("========== saving bar values npy")
			np.save('metrics/metrics_'+experiment+'.npy',metrics)
		#print("experiment",experiment)
		print(exp_id)
		metrics[exp_id]['f1_score']=np.transpose(np.asarray(metrics[exp_id]['f1_score']))*100
		metrics[exp_id]['overall_acc']=np.transpose(np.asarray(metrics[exp_id]['overall_acc']))*100
		metrics[exp_id]['average_acc']=np.transpose(np.asarray(metrics[exp_id]['average_acc']))*100
		
		print("Experiment:{}, dataset:{}. Avg F1:{}. Avg OA:{}. Avg AA:{}".format(
			experiment,dataset,
			np.average(metrics[exp_id]['f1_score']),
			np.average(metrics[exp_id]['overall_acc']),
			np.average(metrics[exp_id]['average_acc'])))

		if dataset=='cv':
			
			#print("metrics[exp_id]['average_acc'].shape",
			#	metrics[exp_id]['average_acc'].shape)
			metrics[exp_id]['f1_score']=metrics[exp_id]['f1_score'][valid_dates]
			metrics[exp_id]['overall_acc']=metrics[exp_id]['overall_acc'][valid_dates]
			metrics[exp_id]['average_acc']=metrics[exp_id]['average_acc'][valid_dates]
			#print("metrics[exp_id]['average_acc'].shape",
			#	metrics[exp_id]['average_acc'].shape)
		exp_handler.append(ax.bar(X + float(exp_id)*width/2, 
			metrics[exp_id]['f1_score'], 
			color = colors[exp_id], width = width/2))
		ax.set_title('Average F1 Score (%)')
		ax.set_xlabel('Month')
		if dataset=='lm':
			xlim=[-0.5,13] 
			ylim=[10,85]
			xticklabels=['Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun']
#			xticklabels=['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun']

			ax.set_xlim(xlim[0],xlim[1])
			ax3.set_xlim(xlim[0],xlim[1])
			ax.set_ylim(75,100)
			
			if small_classes_ignore==True:
				ax.set_ylim(10,80)
				ax3.set_ylim(70,100)
			else:
				ax.set_ylim(10,85)
				ax3.set_ylim(35,100)
			ax.set_xticks(X+width/2)
			ax.set_xticklabels(xticklabels)
			ax2.set_xticks(X+width/2)
			ax2.set_xticklabels(xticklabels)
			ax3.set_xticks(X+width/2)
			ax3.set_xticklabels(xticklabels)
			
		elif dataset=='cv': 
			xlim=[-0.3,8.9]
			xticklabels=['Oct','Nov','Dec','Jan','Feb','Mar','May','Jun','Jul']

			ax.set_xlim(xlim[0],xlim[1])
			ax3.set_xlim(xlim[0],xlim[1])
			if small_classes_ignore==True:
				ax.set_ylim(40,87)
			else:
				ax.set_ylim(7,85)	
			ax3.set_ylim(30,94)

			ax.set_xticks(X+width/2)
			ax.set_xticklabels(xticklabels)
			ax2.set_xticks(X+width/2)
			ax2.set_xticklabels(xticklabels)
			ax3.set_xticks(X+width/2)
			ax3.set_xticklabels(xticklabels)

		exp_handler2.append(ax2.bar(X + float(exp_id)*width/2, 
			metrics[exp_id]['average_acc'], 
			color = colors[exp_id], width = width/2))
		ax2.set_title('Average Accuracy')
		ax2.set_xlabel('Month')
		exp_handler3.append(ax3.bar(X + float(exp_id)*width/2, 
			metrics[exp_id]['overall_acc'], 
			color = colors[exp_id], width = width/2))
		ax3.set_title('Overall Accuracy (%)')
		ax3.set_xlabel('Month')

		#ax3.set_xticks(np.arange(5))
		#ax3.set_xticklabels(('Tom', 'Dick', 'Harry', 'Sally', 'Sue'))
		
		exp_id+=1
	
	#approach1='UConvLSTM'
	#approach2='BConvLSTM'
	#approach3='BDenseConvLSTM'
	
	approach1='BAtrousConvLSTM'
	approach3='BDenseConvLSTM'
	
	approach2='BUnetConvLSTM'
	approach3='BDenseConvLSTM'

	legends=('DeeplabRSConvLSTM','Deeplabv3ConvLSTM','BAtrousConvLSTM','BUnetConvLSTM','BDenseConvLSTM')

	legends=('DeeplabRSDecoderConvLSTM','DeeplabRSConvLSTM','Deeplabv3ConvLSTM','BAtrousConvLSTM','BUnetConvLSTM','BDenseConvLSTM')
	
#	legends=('DeeplabV3+','DeeplabRSDecoder','DeeplabRS','Deeplabv3','BAtrous','BUnet','BDense')
	if experiment_id==1:
		legends=('DeeplabRSDecoder','DeeplabRS','Deeplabv3','BAtrous','BUnet','BDense')
	elif experiment_id==2:
		legends=('BConvLSTM','BDenseConvLSTM','BUnetConvLSTM','BUnet2ConvLSTM','BAtrousGAPConvLSTM','BAtrousConvLSTM','BUnetAtrousConvLSTM','BFCNAtrousConvLSTM')
	elif experiment_id==3:
		legends=('Adagrad+crossentropy','Adagrad+FL','Adam+crossentropy','Adam+FL')
		#legends=('UConvLSTM','BConvLSTM','BDenseConvLSTM','BUnetConvLSTM','BAtrousConvLSTM','BAtrousGAPConvLSTM')
	elif experiment_id==4:
		legends=('UConvLSTM','BConvLSTM','BDenseConvLSTM','BUnetConvLSTM','BAtrousConvLSTM')
	elif experiment_id==6:
		legends=('BConvLSTM','BConvLSTM+WholeInput','UNet_EndConvLSTM','UNet_MidConvLSTM')
	elif experiment_id==6:
		legends=('BUnetConvLSTM','BUnetConvLSTM','BUnetConvLSTM','BUnetConvLSTM+Attention')
	elif experiment_id==7:
		legends=('BUnetConvLSTM','BUnetConvLSTM','BUnetConvLSTM','BUnetConvLSTM+Self attention','Self attention')
	elif experiment_id==8:
		legends=('BConvLSTM','BConvLSTM','BConvLSTM','BConvLSTM','BConvLSTM','BConvLSTM_SelfAttention','BConvLSTM_SelfAttention')
	elif experiment_id==8:
		legends=('BUnetConvlSTM','BUnetStandalone')
	elif experiment_id==9:
		legends=('UConvLSTM','BConvLSTM','BUNetConvLSTM','BAtrousConvLSTM')

	#ncol=len(legends)
	ncol=3

	ax.legend(tuple(exp_handler), legends,loc='lower center', bbox_to_anchor=(0.5, -0.29), shadow=True, ncol=ncol)
	ax2.legend(tuple(exp_handler2), legends,loc='lower center', bbox_to_anchor=(0.5, -0.29), shadow=True, ncol=ncol)
	ax3.legend(tuple(exp_handler3), legends,loc='lower center', bbox_to_anchor=(0.5, -0.29), shadow=True, ncol=ncol)

	#ax.set_rasterized(True)
	#ax2.set_rasterized(True)
	#ax3.set_rasterized(True)
	
	#fig.savefig("f1_score_"+dataset+".eps",format="eps",dpi=300)
	#fig2.savefig("average_acc_"+dataset+".eps",format="eps",dpi=300)
	#fig3.savefig("overall_acc_"+dataset+".eps",format="eps",dpi=300)
	if small_classes_ignore==True:
		small_classes_ignore_id="_sm"
	else:
		small_classes_ignore_id=""
	fig_names={'f1':"f1_score_importantclasses2_"+dataset+small_classes_ignore_id+".png",
		'aa':"average_acc_importantclasses2_"+dataset+small_classes_ignore_id+".png",
		'oa':"overall_acc_importantclasses2_"+dataset+small_classes_ignore_id+".png"}
	
	for f, filename in zip([fig, fig2, fig3],fig_names.values()):
		f.savefig(filename, dpi=300)

	def fig_crop(inpath):
		fig=cv2.imread(inpath)
		h,w,c =fig.shape
		fig=fig[:,150:w-150,:]
		cv2.imwrite(inpath[:-4]+'_crop.png',fig)

	for k, v in fig_names.items():
		fig_crop(v)
	
	#plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)
	#plt.bar(X + 0.25, data[1], color = 'g', width = 0.25)
	#plt.bar(X + 0.50, data[2], color = 'r', width = 0.25)


	#plot_metric=[x['f1_score'] for x in metrics]
	#print(plot_metric)
	
	##plt.show()

#dataset='lm_optical_clouds'
#dataset='lm'
dataset=args.dataset

#dataset='cv'
#dataset='lm_sarh'

#load images
path_img="../../../../dataset/dataset/"+dataset+"_data/patches_bckndfixed/test/patches_in.npy"
imgs_in = np.load(path_img,mmap_mode='r')

load_metrics=False
small_classes_ignore=False
#mode='global'
mode='each_date'
if dataset=='cv':
	experiment_groups=[[
		'prediction_ConvLSTM_seq2seq_batch16_full.npy',
		'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
##		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
		'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'],

		['prediction_ConvLSTM_seq2seq_redoing.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoing.npy'],
		['prediction_ConvLSTM_seq2seq_redoingz.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoingz.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz.npy'],
		['prediction_ConvLSTM_seq2seq_redoingz2.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy'],
		['prediction_ConvLSTM_seq2seq_redoingz2.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoing3.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoing3.npy']]
	exp_id=5

	if exp_id==1:
		experiment_groups=[[#'prediction_deeplabv3plus_v3plus2.npy',
			'prediction.npy',
			'prediction.npy']]
	# 		[#'prediction_deeplabv3plus_v3plus2.npy',
	# 		'prediction_deeplab_rs_multiscale_v3plus.npy',
	# 		'prediction_deeplab_rs_nowifi.npy',
	# 		'prediction_deeplabv3_lauras3.npy',
	# 		'prediction_pyramid_dilated_bconvlstm_lauras2.npy',
	# 		'prediction_FCN_ConvLSTM_seq2seq_bi_skip_lauras2.npy',
	# ##		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
	# 		'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy']]
	if exp_id==2:
		experiment_groups=[[#'prediction_deeplabv3plus_v3plus2.npy',
			'model_best_BUnet4ConvLSTM.h5',
			'model_best_UNet3D.h5',
			'model_best_BUnet4ConvLSTM_SkipLSTM.h5']]
	elif exp_id==3:
		experiment_groups=[[#'prediction_deeplabv3plus_v3plus2.npy',
			#'prediction_BUnet4ConvLSTM_repeating1.npy',
			'model_best_BUnet4ConvLSTM_focal_test.h5',
			'model_best_focal_loss_bunetconvlstm_cv.h5', #focal loss, adagrad
			'model_best_BUnet4ConvLSTM_adam_crossentropy.h5',
			'model_best_BUnet4ConvLSTM_adam_focal.h5',
			'model_best_BUnet4ConvLSTM_adam_weighted_focal.h5']]
	elif exp_id==4:
		experiment_groups=[['model_best_BUnet4ConvLSTM_adam_focal.h5',
			'model_best_BUnet4ConvLSTM_64_adam_focal.h5',
			'model_best_BUnet4ConvLSTM_SkipLSTM_adam_focal.h5',
			'model_best_Unet3D_16_adam_focal.h5',
			'model_best_Unet3D_adam_focal3.h5',
			'model_best_Unet3D_adam_focal4.h5']]
	elif exp_id==5:
		#experiment_groups=[['model_best_BUnet4ConvLSTM_windows_test.h5']]
		experiment_groups=[['model_best_BUnet4ConvLSTM_float32.h5',
			'model_best_BUnet4ConvLSTM_int16.h5',
			'model_best_BUnet4ConvLSTM_windows_test.h5']]

elif dataset=='l2':
		exp_id=1
		experiment_groups=[[
			'model_best_UUnet4ConvLSTM_doty_var_label_valalldates_hwtnorm.h5'
		]]	
		experiment_groups=[[
			'model_best_UUnet4ConvLSTM_doty_var_label_valalldates.h5'
		]]	
		experiment_groups=[[
			'model_best_UUnet4ConvLSTM_doty_fixed_label_len.h5'
		]]	
		experiment_groups=[[
			'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_dec_700perclass.h5'
		]]	
		experiment_groups=[[
			'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_dec_dummy.h5'
		]]	
		experiment_groups=[[
			'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_mar_l2_mar.h5'
		]]	
		experiment_groups=[[
			'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+args.seq_date+'_l2_rep2.h5'
		]]	
		experiment_groups=[[
			'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+args.seq_date+'_lm_firsttry.h5'
		]]	
		experiment_groups=[[
			'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+args.seq_date+'_l2_secondtry.h5'
		]]	
#		experiment_groups=[[
#			'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+args.seq_date+'_l2_rep2.h5'
#		]]	

#		experiment_groups=[[
#			'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+args.seq_date+'_l2.h5'
#		]]	

#		experiment_groups=[[
#			'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+args.seq_date+'_l2_secondtry.h5'
#		]]	

#		experiment_groups=[[
#			'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+args.seq_date+'_700perclass.h5'
#		]]	
elif dataset=='lm':

	#exp_id=8 # choose 4 for thesis and journal paper
	exp_id=9 # choose 4 for thesis and journal paper
	
	if exp_id==9: #Check against matlab f1 results
		experiment_groups=[[
			'model_best_BConvLSTM_2.h5'
		]]
		experiment_groups=[[ # to demonstrate that h5 is a bit better (not so much) than npy
			'model_best_compare.h5',
			'prediction_compare.npy'
		]]



		experiment_groups=[[ # to demonstrate that h5 is a bit better (not so much) than npy
			'model_best_BUnet4ConvLSTM_baseline.h5',
			'model_best_BUnet4ConvLSTM_skip.h5',
			'model_best_BUnet4ConvLSTM_3d.h5'
		],
		[ # to demonstrate that h5 is a bit better (not so much) than npy
			'model_best_BUnet4ConvLSTM_baseline2.h5',
			'model_best_BUnet4ConvLSTM_skip2.h5'
			'model_best_BUnet4ConvLSTM_3d2.h5'
		]]
		experiment_groups=[[
			'model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_alldates.h5'
		]]
		#dates_mode = 'less_one'
		dates_mode = 'less_two'
		dates_mode = 'less_mar18'
		

# lm less one date		
		if dates_mode=='less_one':
			'''
			experiment_groups=[[
				'model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_lessonedate1.h5',
				'model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_lessonedate2.h5',
				'model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_lessonedate3.h5',
				'model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_lessonedate4.h5',
				'model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_lessonedate5.h5',
				
			]]
			'''		
			experiment_groups=[[
				'model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_lessonedate1.h5'],
				['model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_lessonedate2.h5'],
				['model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_lessonedate3.h5'],
				['model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_lessonedate4.h5'],
				['model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_lessonedate5.h5']
				
			]	
		elif dates_mode == 'less_two':
			experiment_groups=[['model_best_BUnet4ConvLSTM_less_two_dates.h5']]	
			#'''
		elif dates_mode == 'less_mar18':
			experiment_groups=[[
				'model_best_BUnet4ConvLSTM_less_mar18_1.h5'],
				['model_best_BUnet4ConvLSTM_less_mar18_2.h5'],
				['model_best_BUnet4ConvLSTM_less_mar18_3.h5'],
				['model_best_BUnet4ConvLSTM_less_mar18_4.h5'],
				['model_best_BUnet4ConvLSTM_less_mar18_5.h5']
				
			]	
# lm all dates
		else:
			'''
			experiment_groups=[[
				'model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_alldates.h5',
				'model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_alldates2.h5',
				'model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_alldates3.h5',
				'model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_alldates4.h5',
				'model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_alldates5.h5',
				
			]]
			'''		
			experiment_groups=[[
				'model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_alldates.h5'],
				['model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_alldates2.h5'],
				['model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_alldates3.h5'],
				['model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_alldates4.h5'],
				['model_best_BUnet4ConvLSTM_lem_baseline_adam_focal_alldates5.h5']
				
			]		
			#'''
		experiment_groups=[['model_best_var_may18_ext_f1es_pt100.h5']]	
		experiment_groups=[['model_best_UUnet4ConvLSTM_doty_var_may18_ext_f1es_rep1.h5']]	
		experiment_groups=[['model_best_UUnet4ConvLSTM_var_may18_ext_f1es_rep1.h5']]	
		experiment_groups=[['model_best_UUnet4ConvLSTM_doty_var_label_valalldates.h5']]	
		experiment_groups=[['model_best_UUnet4ConvLSTM_doty_var_label_valalldates_hwtnorm.h5']]	

		experiment_groups=[[
			'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+args.seq_date+'_700perclass.h5'
		]]	
		experiment_groups=[[
			'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+args.seq_date+'_l2_traintimes2.h5'
		]]	

		loco_class = 8
		experiment_groups=[[
			'model_best_UUnet4ConvLSTM_fixed_label_fixed_'+args.seq_date+'_loco'+str(loco_class)+'_lm_testlm.h5'
		]]	

		experiment_groups=[['model_best_UUnet4ConvLSTM_fixed_label_fixed_'+args.seq_date+'_loco'+str(loco_class)+'_lm_testlm_fewknownclasses.h5']]	

		experiment_groups=[['model_best_UUnet4ConvLSTM_fixed_label_fixed_mar_loco8_lm_testlm_lessclass8_2.h5']]	

#model_best_UUnet4ConvLSTM_fixed_label_fixed_mar_loco8_lm_testlm_stratifiedval
elif dataset=='lm_optical':
	exp_id=1
	experiment_groups=[[
		'prediction_bunetconvlstm_ok.npy'
	]]
elif dataset=='lm_optical_clouds':
	exp_id=1
	experiment_groups=[[
		#'prediction_bunetconvlstm_clouds.npy',
		'prediction_bunetconvlstm_clouds2.npy',
		'prediction_unet3d_clouds.npy'
	]]
	#experiment_groups=[[
		#'prediction_bunetconvlstm_clouds.npy',
	#	'prediction_bunetconvlstm_clouds2.npy',
	#	'prediction_bunetconvlstm_clouds_febnew.npy'
	#]]
elif dataset=='lm_sarh':
	exp_id=1
	experiment_groups=[[
		#'model_best_BUnet4ConvLSTM_adam_focal4_noHmasking.h5',
		#'model_best_BUnet4ConvLSTM_sarh_tvalue20.h5',
		'model_best_BUnet4ConvLSTM_sarh_tvalue40fixed.h5',	
		'model_best_BUnet4ConvLSTM_sarh_tvalue20repeat.h5',		
		#'model_best_BUnet4ConvLSTM_adam_focal_hmasking.h5'
	],
	[
		#'model_best_BUnet4ConvLSTM_adam_focal4_noHmasking.h5',
		#'model_best_BUnet4ConvLSTM_sarh_tvalue20.h5',
		'model_best_BUnet4ConvLSTM_sarh_tvalue40fixed.h5',	
		'model_best_BUnet4ConvLSTM_sarh_tvalue20repeat.h5',		
		#'model_best_BUnet4ConvLSTM_adam_focal_hmasking.h5'
	]]
	
print("Experiment groups",experiment_groups)
if load_metrics==False:
	experiment_metrics=experiment_groups_analyze(dataset,experiment_groups,
		mode=mode,exp_id=exp_id,small_classes_ignore=small_classes_ignore)
	np.save("experiment_metrics_"+dataset+"_"+file_id+".npy",experiment_metrics)

else:
	experiment_metrics=np.load("experiment_metrics_"+dataset+"_"+file_id+".npy")
	print("Difference F1 in percentage",np.average(experiment_metrics[2]['f1_score']-experiment_metrics[1]['f1_score']))
	print("Difference OA in percentage",np.average(experiment_metrics[2]['overall_acc']-experiment_metrics[1]['overall_acc']))


if mode=='each_date':
	experiments_plot(experiment_metrics,experiment_groups[0],
		dataset,experiment_id=exp_id,
		small_classes_ignore=small_classes_ignore)

#metrics['per_class_acc'][~np.isnan(metrics['per_class_acc'])]


