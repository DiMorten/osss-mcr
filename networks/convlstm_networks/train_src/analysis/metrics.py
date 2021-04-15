
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import deb
import numpy as np
class Metrics():
    def plotROCCurve(self, y_test, y_pred, modelId, nameId):
        print("y_test.shape", y_test.shape)
        print("y_pred.shape", y_pred.shape)
        print("y_test.dtype", y_test.dtype)
        print("y_pred.dtype", y_pred.dtype)
        deb.prints(np.unique(y_test))   
        deb.prints(np.unique(y_pred))
        y_test = y_test.copy()
        y_test[y_test!=39] = 0
        y_test[y_test==39] = 1
        deb.prints(np.unique(y_test))   
        deb.prints(np.unique(y_pred))


        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        roc_auc = metrics.auc(tpr, fpr)
        deb.prints(roc_auc)
        deb.prints(thresholds)

        np.savez("roc_curve_"+modelId+"_"+nameId+".npz", fpr=fpr, tpr=tpr)
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(tpr, fpr, label = 'AUC = %0.2f' % roc_auc)
#        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
