
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import deb
import numpy as np
from icecream import ic
import pdb
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

        # =========================== Get metric value


        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=0)
#        roc_auc = metrics.auc(tpr, fpr)
        roc_auc = metrics.auc(fpr, tpr)

        deb.prints(roc_auc)
        deb.prints(thresholds)
        deb.prints(tpr)
#        pdb.set_trace()

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
        np.savez("roc_curve_"+modelId+"_"+nameId+".npz", fpr=fpr, tpr=tpr)
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
#        plt.plot(tpr, fpr, label = 'AUC = %0.2f' % roc_auc)
        plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
#        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
