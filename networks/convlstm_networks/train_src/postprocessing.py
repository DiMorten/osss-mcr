from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
import deb
import numpy as np
import sys
from icecream import ic
from open_set import OpenPCS, SoftmaxThresholding
class PostProcessingMosaic():
	def __init__(self, paramsTrain, h, w):
		self.paramsTrain = paramsTrain
		self.h = h
		self.w = w
	def openSetActivate(self, openSetMethod, known_classes):
		self.openSetMosaic = OpenSetMosaic(openSetMethod, known_classes, self.h, self.w, self.paramsTrain)

	def load_intermediate_features(self, model, in_, pred_logits_patches, debug = 1): # duplicate with main.py:fitPostProcessing() 265
		return self.openSetMosaic.load_intermediate_features(model, in_, pred_logits_patches)
	def predictPatch(self, pred_cl, test_pred_proba, row, col, stride, overlap, debug = 0):
		self.openSetMosaic.predictPatch(pred_cl, test_pred_proba, row,col,stride, overlap, debug = debug)
	
	def applyThreshold(self, prediction_mosaic, debug = 0):
		return self.openSetMosaic.applyThreshold(prediction_mosaic, debug = debug)

	def fit(self, data):
		self.openSetMosaic.openModel.appendToSaveNameId('_'+self.paramsTrain.seq_date)
		self.openSetMosaic.openModel.appendToSaveNameId('_'+self.paramsTrain.dataset)
		self.openSetMosaic.openModel.setModelSaveNameID(self.paramsTrain.seq_date, self.paramsTrain.dataset)
		self.openSetMosaic.openModel.fit(self.data.patches_label, self.data.predictions, self.data.intermediate_features)
		

#        self.openSetMosaic.openModel.fit(label_train, predictions_train, train_pred_proba)

#        for clss in self.paramsTrain.known_classes:
#            batch['label_with_unknown'][batch['label_with_unknown']==int(clss) + 1] = 0
#        batch['label_with_unknown'][batch['label_with_unknown']!=0] = 40

class OpenSetMosaic():

	def __init__(self, openSetMethod, known_classes, h, w, paramsTrain):
		self.paramsTrain = paramsTrain
		self.scores_mosaic=np.zeros((h,w)).astype(np.float16)
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
			self.openModel = SoftmaxThresholding()
		

		self.openModel.setThreshold(threshold)

	def fit(self, data):
		pass
#        data.full_ims_train
#        data.full_labels_train




		

	def loadFittedModel(self):
		try:
			self.openModel.setModelSaveNameID(self.paramsTrain.seq_date, self.paramsTrain.dataset)
			self.openModel.loadFittedModel(path = 'analysis/', nameID = self.openModel.nameID)
			return 0

		except:
			print("Exception: No fitted model method.")
			ic(self.openModel.nameID)
			sys.exit()
			return 1 # error
	def applyThreshold(self, prediction_mosaic, debug = 0):
		return self.openModel.applyThreshold(prediction_mosaic, self.scores_mosaic, debug = debug)

		
	def predictPatch(self, pred_cl, test_pred_proba, row, col, stride, overlap, debug = 0):
		self.openModel.predictScores(pred_cl.flatten() - 1, test_pred_proba,
									debug = debug)
		x, y = pred_cl.shape
		self.openModel.scores = np.reshape(self.openModel.scores, (x, y))
		self.scores_mosaic[row-stride//2:row+stride//2,col-stride//2:col+stride//2] = self.openModel.scores[overlap//2:x-overlap//2,overlap//2:y-overlap//2]        

	def load_intermediate_features(self, model, in_, pred_logits_patches, debug = 1):
		if self.openSetMethod =='OpenPCS' or self.openSetMethod == 'OpenPCS++':
			open_features = model.load_decoder_features(in_, debug = 1)
		else:
			open_features = pred_logits_patches.copy()
			if debug>0:
				ic(open_features.shape) # h, w, classes
			open_features = np.reshape(open_features, (open_features.shape[0], -1, open_features.shape[-1]))
		return open_features

