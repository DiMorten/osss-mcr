
import numpy as np
import sys
sys.path.append('../')
import matplotlib
import scipy

import deb

class OpenSetMethod(): # abstract
    def __init__(self, loco_class):
        self.loco_class = loco_class
    def setThreshold(self, threshold):
        self.threshold = threshold
class SoftmaxThresholding(OpenSetMethod):
    def postprocess(self, label_test, predictions_test, pred_proba_test):
        # pred proba shape is (n_samples, h, w, classes)
        pred_proba_test = scipy.special.softmax(pred_proba_test, axis=-1)

        print("pred_proba_max stats min, avg, max",np.min(pred_proba_test),
                np.average(pred_proba_test),np.max(pred_proba_test))
        pred_proba_max = np.amax(pred_proba_test, axis=-1) # shape (n_samples, h, w)

        print("pred_proba_max stats min, avg, max",np.min(pred_proba_max),
                np.average(pred_proba_max),np.max(pred_proba_max))
        deb.prints(predictions_test.shape)
        deb.prints(pred_proba_max.shape)

        predictions_test[pred_proba_max < self.threshold] = self.loco_class + 1

        return predictions_test



