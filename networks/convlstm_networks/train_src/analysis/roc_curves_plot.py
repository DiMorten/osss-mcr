from sklearn import metrics
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import deb
import numpy as np


def plotMultipleRocCurves(rocCurvesNames, nameID, legendNames):
    for name, legendName in zip(rocCurvesNames, legendNames):
        data = np.load(name)
        tpr = data['fpr']
        fpr = data['tpr']
        roc_auc = metrics.auc(fpr, tpr)*100
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label = legendName + ' (AUC = %0.2f)' % roc_auc)
    plt.legend(loc='best')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig('roc_curves_' + nameID + '.png')
    plt.show()
        




if __name__=='__main__':
    dataset = 'lm'

    if dataset == 'lm':
        date = 'mar'
#        date = 'jun'
#        date = 'jun'

        if date == 'mar':
            rocCurvesNames = ['roc_curve_SoftmaxThresholding__test.npz',
#                    'roc_curve_OpenGMMS_comp1_full_test_mar_lm.npz',
#                    'roc_curve_OpenGMMS_comp2_full_test_mar_lm.npz',
#                    'roc_curve_OpenGMMS_comp4_full_test_mar_lm.npz',
#                    'roc_curve_OpenGMMS_comp8_test.npz',

#                    'roc_curve_OpenPCS_comp20_test_mar_lm.npz',
                    'roc_curve_OpenPCS_comp90_nocovidentity_test.npz',
                    'roc_curve_OpenPCS_comp90_test.npz']
            rocCurvesNames = ['roc_curve_SoftmaxThresholding__test_mar_lm.npz',
#                    'roc_curve_OpenGMMS_comp1_full_test_mar_lm.npz',
#                    'roc_curve_OpenGMMS_comp2_full_test_mar_lm.npz',
#                    'roc_curve_OpenGMMS_comp4_full_test_mar_lm.npz',
#                    'roc_curve_OpenGMMS_comp8_test.npz',

#                    'roc_curve_OpenPCS_comp20_test_mar_lm.npz',
                    'roc_curve_OpenPCS_comp90_nocovidentity_test_mar_lm.npz',
                    'roc_curve_OpenPCS_comp90_test_mar_lm.npz']

            legendNames = ['Softmax Thresholding',
#                'OpenGMMS 1 components',
#                'OpenGMMS 2 components',
#                'OpenGMMS 4 components',
#                'OpenGMMS 8 components',
                'OpenPCS',
#                'OpenPCA + Inverse Covariance 20',
                'OpenPCS + Inverse Covariance']
        elif date == 'jun':
            rocCurvesNames = ['roc_curve_SoftmaxThresholding__test_jun.npz',
#                    'roc_curve_OpenGMMS_comp1_full_test_jun_lm.npz',
#                    'roc_curve_OpenGMMS_comp2_full_test_jun_lm.npz',
#                    'roc_curve_OpenGMMS_comp4_full_test_jun_lm.npz',
#                    'roc_curve_OpenGMMS_comp8_full_test_jun_lm.npz',
                    'roc_curve_OpenPCS_comp90_nocovidentity_test_jun.npz',
                    'roc_curve_OpenPCS_comp90_test_jun.npz']
            legendNames = ['Softmax Thresholding',
#                'OpenGMMS 1 components',
#                'OpenGMMS 2 components',
#                'OpenGMMS 4 components',
#                'OpenGMMS 8 components',
                'OpenPCS',
                'OpenPCS + Inverse Covariance']
    elif dataset == 'cv':
        date = 'jun'
        if date == 'jun':
            rocCurvesNames = ['roc_curve_SoftmaxThresholding__test_jun_cv.npz',
#                    'roc_curve_OpenGMMS_comp8_full_test_jun_cv.npz',
    #                'roc_curve_OpenGMMS_comp8_full_test_jun.npz',
                    'roc_curve_OpenPCS_comp90_test_jun_cv.npz',
                    'roc_curve_OpenPCS_comp90_nocovidentity_test_jun_cv.npz']
            legendNames = ['Softmax Thresholding',
#                'OpenGMMS 8 components',
                'OpenPCS + Inverse Covariance',
                'OpenPCS']

    plotMultipleRocCurves(rocCurvesNames, date+'_'+dataset, legendNames)