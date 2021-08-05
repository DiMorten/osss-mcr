from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, UpSampling2D
from keras.models import Model
import deb
import numpy as np
class PostProcessingMosaic():
    def __init__(self, paramsTrain, h, w):
        self.paramsTrain = paramsTrain
        self.h = h
        self.w = w
    def openSetActivate(self, openSetMethod):
        self.openSetMosaic = OpenSetMosaic(openSetMethod, self.h, self.w)
    def load_intermediate_features(self, openSetMethod, model, in_, pred_logits_patches, debug = 1):
        self.openSetMosaic.load_intermediate_features(openSetMethod, model, in_, pred_logits_patches)
    def predictPatch(self, pred_cl, test_pred_proba):
        self.openSetMosaic.predictPatch(pred_cl, test_pred_proba)
    
    def applyThreshold(self, prediction_mosaic):
        self.openSetMosaic.applyThreshold(prediction_mosaic)


class OpenSetMosaic():

    def __init__(self, openSetMethod, h, w):
        self.scores_mosaic=np.zeros((h,w)).astype(np.float16)
        self.openSetMethod = openSetMethod
        if self.openSetMethod == 'OpenPCS' or self.openSetMethod == 'OpenPCS++':
            self.openModel = OpenPCS(known_classes = known_classes,
        #			n_components = 16)
                n_components = 90)
            makeCovMatrixIdentity = True if self.openSetMethod == 'OpenPCS++' else False
            self.openModel.makeCovMatrixIdentitySet(makeCovMatrixIdentity)
        elif self.openSetMethod == 'SoftmaxThresholding':
            self.openModel = SoftmaxThresholding()

        self.openModel.setThreshold(threshold)

        try:
            self.openModel.setModelSaveNameID(paramsTrain.seq_date, paramsTrain.dataset)
            self.openModel.loadFittedModel(path = 'analysis/', nameID = self.openModel.nameID)

        except:
            print("Exception: No fitted model method")
    def applyThreshold(self, prediction_mosaic):
        self.openSetMethod.applyThreshold(prediction_mosaic, self.scores_mosaic)
        
    def predictPatch(self, pred_cl, test_pred_proba):
        self.openModel.predictScores(pred_cl.flatten() - 1, test_pred_proba,
									debug = debug)
        x, y = pred_cl.shape
        self.openModel.scores = np.reshape(openModel.scores, (x, y))
        self.scores_mosaic[m-stride//2:m+stride//2,n-stride//2:n+stride//2] = self.openModel.scores[overlap//2:x-overlap//2,overlap//2:y-overlap//2]        

    def load_intermediate_features(self, model, in_, pred_logits_patches, debug = 1):
        if self.openSetMethod =='OpenPCS' or self.openSetMethod == 'OpenPCS++':
            open_features = model.load_decoder_features(model, in_, debug = 1)
        else:
            open_features = pred_logits_patches.copy()
            if debug>0:
                ic(open_features.shape) # h, w, classes
            open_features = np.reshape(open_features, (open_features.shape[0], -1, open_features.shape[-1]))
        return open_features

