from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
import deb
import numpy as np
import sys
from icecream import ic
from open_set import OpenPCS, SoftmaxThresholding, Uncertainty, ScaledSoftmaxThresholding
import pdb
class OpenSetManager():
	def __init__(self, paramsTrain, h, w):
		self.paramsTrain = paramsTrain
		self.h = h
		self.w = w
	def openSetActivate(self, openSetMethod, known_classes):

		self.scores_mosaic=np.zeros((self.h,self.w)).astype(np.float16)
		self.openSetMethod = openSetMethod

		threshold = -2000

		if self.openSetMethod == 'OpenPCS' or self.openSetMethod == 'OpenPCS++':
			self.openModel = OpenPCS(known_classes = known_classes,
		#			n_components = 16)
				n_components = 90)
			makeCovMatrixIdentity = True if self.openSetMethod == 'OpenPCS++' else False
			self.openModel.makeCovMatrixIdentitySet(makeCovMatrixIdentity)
			if self.openSetMethod == 'OpenPCS++':
				threshold = -184.4

		elif self.openSetMethod == 'SoftmaxThresholding':
			self.openModel = SoftmaxThresholding() if self.paramsTrain.confidenceScaling == False else ScaledSoftmaxThresholding()
			threshold = 0.9
		
		elif self.openSetMethod == 'Uncertainty':
			self.openModel = Uncertainty()
			threshold = 0.9
		self.openModel.setThreshold(threshold)

	def load_intermediate_features(self, model, in_, pred_logits_patches, debug = 1): # duplicate with main.py:fitOpenSet() 265
		if self.openSetMethod =='OpenPCS' or self.openSetMethod == 'OpenPCS++':
			open_features = model.load_decoder_features(in_, debug = 1)
		else:
			open_features = pred_logits_patches.copy()
			if debug>0:
				ic(open_features.shape) # h, w, classes
			open_features = np.reshape(open_features, (open_features.shape[0], -1, open_features.shape[-1]))
		return open_features

	def predictPatch(self, pred_cl, test_pred_proba, row, col, stride, overlap, debug = 0):

		self.openModel.predictScores(pred_cl.flatten() - 1, test_pred_proba,
									debug = debug)
		x, y = pred_cl.shape
		self.openModel.scores = np.reshape(self.openModel.scores, (x, y))
		# this will be made in an upper method
		self.scores_mosaic[row-stride//2:row+stride//2,col-stride//2:col+stride//2] = self.openModel.scores[overlap//2:x-overlap//2,overlap//2:y-overlap//2]        


	def applyThreshold(self, prediction_mosaic, debug = 0):
		return self.openModel.applyThreshold(prediction_mosaic, self.scores_mosaic, debug = debug)



	def loadFittedModel(self):
		self.openModel.setModelSaveNameID(self.paramsTrain.seq_date, self.paramsTrain.dataset)
		if self.openSetMethod == 'OpenPCS' or self.openSetMethod == 'OpenPCS++':
			try:
				self.openModel.loadFittedModel(path = 'results/open_set/', nameID = self.openModel.nameID)
				return 0

			except:
				print("Exception: No fitted model method.")
				ic(self.openModel.nameID)
				sys.exit()
				return 1 # error


	def fit(self, modelManager, data):

		if self.paramsTrain.openSetMethod == 'OpenPCS' or self.paramsTrain.openSetMethod == 'OpenPCS++' or \
			(self.paramsTrain.openSetMethod == 'SoftmaxThresholding' and self.paramsTrain.confidenceScaling == True):

			if self.paramsTrain.openSetMethod == 'SoftmaxThresholding':
				coords = data.patches['val']['coords']
			else:
				coords = data.patches['train']['coords']	
			label = data.full_label_train

			prediction_dtype = np.float16

			# first, translate self.train_label

			ic(label.shape)

			ic(np.unique(label, return_counts=True))

			label_with_unknown_train = data.getTrainLabelWithUnknown()

			ic(label.shape)
			ic(np.unique(label, return_counts=True))

			data.full_ims_train = data.addPaddingToInput(
				modelManager.model_t_len, data.full_ims_train)

			data.patches_in = data.getSequencePatchesFromCoords(
				data.full_ims_train, coords).astype(prediction_dtype) # test coords is called self.coords, make custom init in this class. self.full_ims is also set independent
			data.patches_label = data.getPatchesFromCoords(
				label_with_unknown_train, coords)
#       	self.coords = coords # not needed. use train coords directly
			

			data.logits = modelManager.model.predict(data.patches_in)
			
			data.predictions = data.logits.argmax(axis=-1).astype(np.uint8) 

			data.setDateList(self.paramsTrain)

			ic(np.unique(data.predictions, return_counts=True))

			data.predictions = data.newLabel2labelTranslate(data.predictions, 
					'results/label_translations/new_labels2labels_'+self.paramsTrain.dataset+'_'+data.dataset_date+'_S1.pkl')

			if self.paramsTrain.openSetMethod =='OpenPCS' or self.paramsTrain.openSetMethod =='OpenPCS++':
				data.intermediate_features = modelManager.load_decoder_features(data.patches_in).astype(prediction_dtype)
			else:
				data.intermediate_features = np.expand_dims(data.predictions.copy(), axis=-1) # to-do: avoid copy
			ic(data.patches_in.shape, data.patches_label.shape)
			ic(data.predictions.shape)
			ic(data.intermediate_features.shape)
			# pdb.set_trace()
			data.patches_label = data.patches_label.flatten()
			data.predictions = data.predictions.flatten()

			data.intermediate_features = np.reshape(data.intermediate_features,(-1, data.intermediate_features.shape[-1]))
			data.logits = np.reshape(data.logits,(-1, data.logits.shape[-1]))

			ic(data.intermediate_features.shape,
				data.patches_label.shape)
			# pdb.set_trace()
			data.intermediate_features = data.intermediate_features[data.patches_label!=0]
			data.logits = data.logits[data.patches_label!=0]
			
			data.predictions = data.predictions[data.patches_label!=0]
			data.patches_label = data.patches_label[data.patches_label!=0]

			data.predictions = data.predictions - 1
			data.patches_label = data.patches_label - 1

			ic(np.unique(data.patches_label, return_counts=True))
			ic(np.unique(data.predictions, return_counts=True))

			ic(data.patches_in.shape, data.patches_label.shape)
			ic(data.predictions.shape)
			ic(data.intermediate_features.shape)
			
#			pdb.set_trace()
			if self.paramsTrain.confidenceScaling == True:
				self.openModel.addLogits(data.logits)
			self.fitPreprocessed(data)

	def fitPreprocessed(self, data):
		self.openModel.appendToSaveNameId('_'+self.paramsTrain.seq_date)
		self.openModel.appendToSaveNameId('_'+self.paramsTrain.dataset)
		self.openModel.setModelSaveNameID(self.paramsTrain.seq_date, self.paramsTrain.dataset)
		self.openModel.fit(data.patches_label, data.predictions, data.intermediate_features)
		