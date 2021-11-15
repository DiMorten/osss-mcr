from tensorflow.keras import backend as K
import deb
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report
from icecream import ic
import sklearn
import matplotlib.pyplot as plt
from icecream import ic
import pathlib
class Metrics():

	def __init__(self, paramsTrain):
		self.paramsTrain = paramsTrain

	def my_f1_score(self,label,prediction):
		f1_values=f1_score(label,prediction,average=None)

		#label_unique=np.unique(label) # [0 1 2 3 5]
		#prediction_unique=np.unique(prediction.argmax(axis-1)) # [0 1 2 3 4]
		#[ 0.8 0.8 0.8 0 0.7 0.7]

		f1_value=np.sum(f1_values)/len(np.unique(label))

		#print("f1_values",f1_values," f1_value:",f1_value)
		return f1_value
	def mosaicGet(self, prediction, label, ignore_bcknd=True, debug=2):
		pass
#		ic(prediction.shape, label.shape)
		
#		self.get(prediction, label)


	def get(self,prediction, label,ignore_bcknd=True,debug=2): #requires batch['prediction'],batch['label']
		print("======================= METRICS GET")
		ic(prediction.shape, label.shape)
		ic(np.unique(prediction, return_counts=True))
		ic(np.unique(label, return_counts=True))
		ic(len(np.unique(label, return_counts=True)))
		
#		if prediction.shape[-1] < 20:
#		class_n=prediction.shape[-1]
		class_n = len(np.unique(label)) - 1
		ic(class_n)

#		else:
		#print("label unque at start of metrics_get",
		#	np.unique(label.argmax(axis=4),return_counts=True))
		

		#label[label[:,],:,:,:,:]
		#data['label_copy']=data['label_copy'][:,:,:,:,:-1] # Eliminate bcknd dimension after having eliminated bcknd samples
		
		#print("label_copy unque at start of metrics_get",
	#		np.unique(data['label_copy'].argmax(axis=4),return_counts=True))
		deb.prints(prediction.shape,debug,2)
		deb.prints(label.shape,debug,2)
		#deb.prints(data['label_copy'].shape,debug,2)


		if len(prediction.shape) > 3:
			print("Computing argmax of predictions")
			prediction=prediction.argmax(axis=-1) #argmax de las predicciones. Softmax no es necesario aqui.
		deb.prints(prediction.shape)
		prediction=np.reshape(prediction,-1) #convertir en un vector
		deb.prints(prediction.shape)
		if len(label.shape) > 3:
			print("Computing argmax of labels")
			label=label.argmax(axis=-1) #igualmente, sacar el maximo valor de los labels (se pierde la ultima dimension; saca el valor maximo del one hot encoding es decir convierte a int)
		label=np.reshape(label,-1) #flatten

		ic(np.unique(prediction, return_counts=True))
		ic(np.unique(label, return_counts=True))
		
		prediction, label = self.filterSamples(prediction, label, class_n)


		#============= TEST UNIQUE PRINTING==================#
		unique,count=np.unique(label,return_counts=True)
		ic(np.unique(label,return_counts=True))
#		print("Metric label unique+1,count",unique+1,count)
		unique,count=np.unique(prediction,return_counts=True)
		ic(np.unique(prediction,return_counts=True))
#		print("Metric prediction unique+1,count",unique+1,count)
		
		#========================METRICS GET================================================#

		metrics={}
#		metrics['f1_score']=f1_score(label,prediction,average='macro')
		metrics['f1_score'] = self.my_f1_score(label,prediction)
		metrics['f1_score_weighted']=f1_score(label,prediction,average='weighted')
		metrics['f1_score_noavg']=f1_score(label,prediction,average=None)
		metrics['overall_acc']=accuracy_score(label,prediction)
		metrics['confusion_matrix']=confusion_matrix(label,prediction)
		#print(confusion_matrix_)
		metrics['per_class_acc']=(metrics['confusion_matrix'].astype('float') / metrics['confusion_matrix'].sum(axis=1)[:, np.newaxis]).diagonal()
		acc=metrics['confusion_matrix'].diagonal()/np.sum(metrics['confusion_matrix'],axis=1)
		acc=acc[~np.isnan(acc)]
		metrics['average_acc']=np.average(metrics['per_class_acc'][~np.isnan(metrics['per_class_acc'])])

		# open set metrics
		if self.paramsTrain.group_bcknd_classes == True:
			metrics['f1_score_known'] = np.average(metrics['f1_score_noavg'][:-1])
			metrics['f1_score_unknown'] = metrics['f1_score_noavg'][-1]
			
			
			precision = precision_score(label,prediction, average=None)
			recall = recall_score(label,prediction, average=None)
			
			deb.prints(precision)
			deb.prints(recall)
			metrics['precision_known'] = np.average(precision[:-1])
			metrics['recall_known'] = np.average(recall[:-1])
			metrics['precision_unknown'] = precision[-1]
			metrics['recall_unknown'] = recall[-1]
			
#		metrics['precision_avg'] = np.average(precision[:-1])
#		metrics['recall_avg'] = np.average(recall[:-1])
		return metrics
			
	def filterSamples(self, prediction, label, class_n):
		prediction=prediction[label<class_n] #logic
		label=label[label<class_n] #logic
		return prediction, label

	def plotROCCurve(self, y_test, y_pred, modelId, nameId, unknown_class_id = 39, pos_label=0):
		print("y_test.shape", y_test.shape)
		print("y_pred.shape", y_pred.shape)
		print("y_test.dtype", y_test.dtype)
		print("y_pred.dtype", y_pred.dtype)
		deb.prints(np.unique(y_test))   
		deb.prints(np.unique(y_pred))
		y_test = y_test.copy()
		y_test[y_test!=unknown_class_id] = 0
		y_test[y_test==unknown_class_id] = 1
		deb.prints(np.unique(y_test))   
		deb.prints(np.unique(y_pred))

		# =========================== Get metric value


		fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred, pos_label=pos_label)
#        roc_auc = metrics.auc(tpr, fpr)
		roc_auc = sklearn.metrics.auc(fpr, tpr)

		deb.prints(roc_auc)
		deb.prints(thresholds)
		deb.prints(tpr)
#        pdb.set_trace()

		optimal_idx = np.argmax(tpr - fpr)
		#optimal_idx = np.argmax(fpr - tpr)
		
		optimal_threshold = thresholds[optimal_idx]
		deb.prints(optimal_threshold)

		# =================== Find thresholds for specified TPR value
		tpr_threshold_values = [0.1, 0.3, 0.5, 0.7, 0.9]
		deb.prints(tpr_threshold_values)
		tpr_idxs = [np.where(tpr>tpr_threshold_values[0])[0][0],
			np.where(tpr>tpr_threshold_values[1])[0][0],
			np.where(tpr>tpr_threshold_values[2])[0][0],
			np.where(tpr>tpr_threshold_values[3])[0][0],
			np.where(tpr>tpr_threshold_values[4])[0][0]
		]
		deb.prints(tpr_idxs)
		
		thresholds_by_tpr = thresholds[tpr_idxs]
		deb.prints(thresholds_by_tpr)
#        pdb.set_trace()
		# ========================== Plot
		pathlib.Path("results/open_set/roc_curve/").mkdir(parents=True, exist_ok=True)

		np.savez("results/open_set/roc_curve/roc_curve_"+modelId+"_"+nameId+".npz", fpr=fpr, tpr=tpr)
		plt.figure()
		plt.plot([0, 1], [0, 1], 'k--')
#        plt.plot(tpr, fpr, label = 'AUC = %0.2f' % roc_auc)
		plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('AUC = %0.2f' % roc_auc)
		plt.savefig('roc_auc_'+modelId+"_"+nameId+'.png', dpi = 500)
#        plt.gca().set_aspect('equal', adjustable='box')
		#plt.show()
		return optimal_threshold

class MetricsTranslated(Metrics):
	def filterSamples(self, prediction, label, class_n):
		prediction=prediction[label!=255] #logic
		label=label[label!=255] #logic
		return prediction, label
def binary_accuracy(y_true, y_pred):
	return K.mean(K.equal(y_true, K.round(y_pred)))


def categorical_accuracy(y_true, y_pred):
	return K.mean(K.equal(K.argmax(y_true, axis=-1),
						  K.argmax(y_pred, axis=-1)))


def sparse_categorical_accuracy(y_true, y_pred):
	return K.mean(K.equal(K.max(y_true, axis=-1),
						  K.cast(K.argmax(y_pred, axis=-1), K.floatx())))


def top_k_categorical_accuracy(y_true, y_pred, k=5):
	return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k))


def mean_squared_error(y_true, y_pred):
	return K.mean(K.square(y_pred - y_true))


def mean_absolute_error(y_true, y_pred):
	return K.mean(K.abs(y_pred - y_true))


def mean_absolute_percentage_error(y_true, y_pred):
	diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
											K.epsilon(),
											None))
	return 100. * K.mean(diff)


def mean_squared_logarithmic_error(y_true, y_pred):
	first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
	second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
	return K.mean(K.square(first_log - second_log))


def hinge(y_true, y_pred):
	return K.mean(K.maximum(1. - y_true * y_pred, 0.))


def squared_hinge(y_true, y_pred):
	return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)))


def categorical_crossentropy(y_true, y_pred):
	return K.mean(K.categorical_crossentropy(y_pred, y_true))


def sparse_categorical_crossentropy(y_true, y_pred):
	return K.mean(K.sparse_categorical_crossentropy(y_pred, y_true))


def binary_crossentropy(y_true, y_pred):
	return K.mean(K.binary_crossentropy(y_pred, y_true))


def kullback_leibler_divergence(y_true, y_pred):
	y_true = K.clip(y_true, K.epsilon(), 1)
	y_pred = K.clip(y_pred, K.epsilon(), 1)
	return K.mean(K.sum(y_true * K.log(y_true / y_pred), axis=-1))


def poisson(y_true, y_pred):
	return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()))


def cosine_proximity(y_true, y_pred):
	y_true = K.l2_normalize(y_true, axis=-1)
	y_pred = K.l2_normalize(y_pred, axis=-1)
	return -K.mean(y_true * y_pred)


def matthews_correlation(y_true, y_pred):
	"""Matthews correlation metric.

	It is only computed as a batch-wise average, not globally.

	Computes the Matthews correlation coefficient measure for quality
	of binary classification problems.
	"""
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos

	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos

	tp = K.sum(y_pos * y_pred_pos)
	tn = K.sum(y_neg * y_pred_neg)

	fp = K.sum(y_neg * y_pred_pos)
	fn = K.sum(y_pos * y_pred_neg)

	numerator = (tp * tn - fp * fn)
	denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

	return numerator / (denominator + K.epsilon())


def precision(y_true, y_pred):
	"""Precision metric.

	Only computes a batch-wise average of precision.

	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision


def recall(y_true, y_pred):
	"""Recall metric.

	Only computes a batch-wise average of recall.

	Computes the recall, a metric for multi-label classification of
	how many relevant items are selected.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall


def fbeta_score(y_true, y_pred, beta=1):
	"""Computes the F score.

	The F score is the weighted harmonic mean of precision and recall.
	Here it is only computed as a batch-wise average, not globally.

	This is useful for multi-label classification, where input samples can be
	classified as sets of labels. By only using accuracy (precision) a model
	would achieve a perfect score by simply assigning every class to every
	input. In order to avoid this, a metric should penalize incorrect class
	assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
	computes this, as a weighted mean of the proportion of correct class
	assignments vs. the proportion of incorrect class assignments.

	With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
	correct classes becomes more important, and with beta > 1 the metric is
	instead weighted towards penalizing incorrect class assignments.
	"""
	if beta < 0:
		raise ValueError('The lowest choosable beta is zero (only precision).')

	# If there are no true positives, fix the F score at 0 like sklearn.
	if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
		return 0

	p = precision(y_true, y_pred)
	r = recall(y_true, y_pred)
	bb = beta ** 2
	fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
	return fbeta_score


def fmeasure(y_true, y_pred):
	"""Computes the f-measure, the harmonic mean of precision and recall.

	Here it is only computed as a batch-wise average, not globally.
	"""
	return fbeta_score(y_true, y_pred, beta=1)


# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity
fscore = f1score = fmeasure

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score



def compute_metrics(true_labels, predicted_labels):
	accuracy = accuracy_score(true_labels, predicted_labels)
	f1score = 100*f1_score(true_labels, predicted_labels, average=None)
	recall = 100*recall_score(true_labels, predicted_labels, average=None)
	prescision = 100*precision_score(true_labels, predicted_labels, average=None)
	return accuracy, f1score, recall, prescision
