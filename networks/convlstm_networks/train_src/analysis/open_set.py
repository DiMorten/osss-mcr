
import numpy as np
import sys
sys.path.append('../')
import matplotlib
import scipy
from sklearn import decomposition
import time
import pdb 
from sklearn.preprocessing import MinMaxScaler
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


class OpenPCS(OpenSetMethod):
    def __init__(self, loco_class, known_classes, n_components):
        super().__init__(loco_class)
        self.known_classes = known_classes
        self.n_components = n_components
        

    def postprocess(self, label_test, predictions_test, pred_proba_test):
        # pred proba shape is (n_samples, h, w, classes)
        ##pred_proba_test = scipy.special.softmax(pred_proba_test, axis=-1)

        ##print("pred_proba_max stats min, avg, max",np.min(pred_proba_test),
        ##        np.average(pred_proba_test),np.max(pred_proba_test))

        deb.prints(predictions_test.shape)


        print("*"*20, " Flattening the results")

        deb.prints(label_test.shape)
        deb.prints(predictions_test.shape)
        deb.prints(pred_proba_test.shape)

        label_test = label_test.flatten()
        predictions_test = predictions_test.flatten()
        #pred_proba_test = pred_proba_test.reshape((pred_proba_test.shape[0], -1))
        #deb.prints(pred_proba_test.shape)

        ##pred_proba_test = pred_proba_test.reshape((-1, pred_proba_test.shape[-1]))
        print("*"*20, " Flattened the results")
        deb.prints(label_test.shape)
        deb.prints(predictions_test.shape)
        deb.prints(pred_proba_test.shape)
        #pdb.set_trace()

        self.fit_pca_models(label_test, predictions_test, pred_proba_test)
        deb.prints(np.unique(predictions_test, return_counts=True))
        predictions_test, _ = self.predict_unknown_class(predictions_test, pred_proba_test)
        deb.prints(np.unique(predictions_test, return_counts=True))
        #pdb.set_trace()
        return predictions_test

    def predict_unknown_class(self, predictions_test, open_features): # self.model_list, self.threshold
        scores = np.zeros_like(predictions_test, dtype=np.float)
        print('*'*20, 'predict_unknown_class')
        for idx, c in enumerate(self.known_classes):
            print('idx, class', idx, c)
            feat_msk = (predictions_test == c)
            if np.any(feat_msk):
                try:
                    scores[feat_msk] = self.model_list[idx].score_samples(open_features[feat_msk, :])
                except:
                    scores[feat_msk] = 0
        
        print("scores stats min, avg, max",np.min(scores),
                np.average(scores),np.max(scores))


        #scaler = MinMaxScaler()
        #scores = np.squeeze(scaler.fit_transform(scores.reshape(1, -1)))

        #print("scores stats min, avg, max",np.min(scores),
        #        np.average(scores),np.max(scores))
        #deb.prints(scores.shape)
        predictions_test[scores < self.threshold] = self.loco_class + 1
        return predictions_test, scores #scores in case you want to threshold them again

    def fit_pca_models(self, label_test, predictions_test, open_features):
        self.model_list = []
        print('*'*20, 'fit_pca_models')
        for c in self.known_classes:
            
            print('Fitting model for class %d...' % (c))
            sys.stdout.flush()
            
            tic = time.time()
            
            # Computing PCA models from features.
            model = self.fit_pca_model_perclass(label_test, predictions_test, open_features, c)#feat_list, true_list, prds_list, c)
            
            self.model_list.append(model)
            
            toc = time.time()
            print('    Time spent fitting model %d: %.2f' % (c, toc - tic))


        #predictions_test[pred_proba_max < self.threshold] = self.loco_class + 1


    def fit_pca_model_perclass(self, label_test, predictions_test, open_features, cl):
        model = decomposition.PCA(n_components=self.n_components, random_state=12345)
        
        #deb.prints(np.unique(label_test,return_counts=True))
        #deb.prints(np.unique(predictions_test,return_counts=True))

        deb.prints(cl)
        #deb.prints(np.unique(label_test == cl))
        #deb.prints(np.unique(predictions_test == cl))
 #       deb.prints(np.unique(predictions_test == cl))

        cl_feat_flat = open_features[(label_test == cl) & (predictions_test == cl), :]
        min_samples = 50
        deb.prints(cl_feat_flat.shape)
        if cl_feat_flat.shape[0]>min_samples:
            
            perm = np.random.permutation(cl_feat_flat.shape[0])
            
            if perm.shape[0] > 32768:
                cl_feat_flat = cl_feat_flat[perm[:32768], :]
            
            model.fit(cl_feat_flat)
            return model
        else:
            return None


