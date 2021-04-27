from sklearn import metrics
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import deb
import numpy as np


def plotMultipleRocCurves(rocCurvesNames, nameID):
    for name in rocCurvesNames:
        data = np.load(name)
        tpr = data['fpr']
        fpr = data['tpr']
        roc_auc = metrics.auc(fpr, tpr)*100
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label = name[10:-9] + ' AUC = %0.2f' % roc_auc)
    plt.legend(loc='best')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig('roc_curves' + nameID + '.png')
    plt.show()
        




if __name__=='__main__':
    date = 'jun'
    if date == 'mar':
        rocCurvesNames = ['roc_curve_SoftmaxThresholding__test.npz',
                'roc_curve_OpenGMMS_comp8_test.npz',
                'roc_curve_OpenPCS_comp90_test.npz',
                'roc_curve_OpenPCS_comp90_nocovidentity_test.npz']
    elif date == 'jun':
        rocCurvesNames = ['roc_curve_SoftmaxThresholding__test_jun.npz',
                'roc_curve_OpenGMMS_comp4_full_test_jun.npz',
#                'roc_curve_OpenGMMS_comp8_full_test_jun.npz',
                'roc_curve_OpenPCS_comp90_test_jun.npz',
                'roc_curve_OpenPCS_comp90_nocovidentity_test_jun.npz']
    plotMultipleRocCurves(rocCurvesNames, date)