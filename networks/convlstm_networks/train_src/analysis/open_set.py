
import numpy as np
import sys
sys.path.append('../')
import matplotlib
import scipy
from sklearn import decomposition, mixture
import time
import pdb 
from sklearn.preprocessing import MinMaxScaler
import deb
import pickle
from icecream import ic 
import os
from scipy.spatial.distance import mahalanobis
from scipy.linalg import fractional_matrix_power
from scipy import linalg
from math import log
ic.configureOutput(includeContext=False, prefix='[@debug] ')


def fast_logdet(A):
    """Compute log(det(A)) for A symmetric.
    Equivalent to : np.log(nl.det(A)) but more robust.
    It returns -Inf if det(A) is non positive or is not defined.
    Parameters
    ----------
    A : array-like
        The matrix.
    """
    sign, ld = np.linalg.slogdet(A)
    if not sign > 0:
        return -np.inf
    return ld

class OpenSetMethod(): # abstract
    def __init__(self, loco_class):
        self.loco_class = loco_class
        self.scoresNotCalculated = True
        self.saveNameId = ''
    def setThreshold(self, threshold):
        self.threshold = threshold
    def storeScores(self):
        np.save('scores_'+self.name+'_'+self.saveNameId+'.npy',self.scores)
    def loadScores(self):
        self.scores = np.load('scores_'+self.name+'_'+self.saveNameId+'.npy')
    def appendToSaveNameId(self, saveNameId):
        self.saveNameId = self.saveNameId + saveNameId
class SoftmaxThresholding(OpenSetMethod):

    def __init__(self, loco_class=0):
        super().__init__(loco_class)

        self.fittedFlag = True
        self.name = 'SoftmaxThresholding'

    def fit(self, label_train, predictions_train, pred_proba_train):
        pass

    def predictScores(self, predictions_test, pred_proba_test, debug=1):

        # pred proba shape is (n_samples, h, w, classes)
        pred_proba_test = scipy.special.softmax(pred_proba_test, axis=-1)
        if debug>0:
            print("pred_proba_max stats min, avg, max",np.min(pred_proba_test),
                    np.average(pred_proba_test),np.max(pred_proba_test))
        pred_proba_max = np.amax(pred_proba_test, axis=-1) # shape (n_samples, h, w)

        if debug>0:
            print("pred_proba_max stats min, avg, max",np.min(pred_proba_max),
                    np.average(pred_proba_max),np.max(pred_proba_max))
            deb.prints(predictions_test.shape)
            deb.prints(pred_proba_max.shape)

        self.scores = pred_proba_max
        #ic(self.scores.shape)
        #ic()
    def predict(self, predictions_test, scores = None, debug = 1):
        if np.all(scores) == None:
            scores = self.scores
        predictions_test[scores < self.threshold] = 40 #self.loco_class + 1
        return predictions_test


class OpenSetMethodGaussian(OpenSetMethod):
    def __init__(self, known_classes, n_components, loco_class=0):
        super().__init__(loco_class)
        self.known_classes = known_classes
        self.n_components = n_components
        self.fittedFlag = False
#        self.name = 'OpenPCS'
#        self.model_type = decomposition.PCA(n_components=self.n_components, random_state=12345)

        
    def fit(self, label_train, predictions_train, pred_proba_train):
        # pred proba shape is (n_samples, h, w, classes)
        ##pred_proba_test = scipy.special.softmax(pred_proba_test, axis=-1)

        ##print("pred_proba_max stats min, avg, max",np.min(pred_proba_test),
        ##        np.average(pred_proba_test),np.max(pred_proba_test))
        deb.prints(predictions_train.shape)


        print("*"*20, " Flattening the results")

        deb.prints(label_train.shape)
        deb.prints(predictions_train.shape)
        deb.prints(pred_proba_train.shape)

        label_train = label_train.flatten()
        predictions_train = predictions_train.flatten()
        #pred_proba_test = pred_proba_test.reshape((pred_proba_test.shape[0], -1))
        #deb.prints(pred_proba_test.shape)
        deb.prints(np.unique(label_train, return_counts=True))

        ##pred_proba_test = pred_proba_test.reshape((-1, pred_proba_test.shape[-1]))
        print("*"*20, " Flattened the results")
        deb.prints(label_train.shape)
        deb.prints(predictions_train.shape)
        deb.prints(pred_proba_train.shape)
        #pdb.set_trace()
        #if self.load_model == False:
        self.fit_pca_models(label_train, predictions_train, pred_proba_train)
#            self.modelSave()
#        else:
#            self.modelLoad()
        deb.prints(np.unique(predictions_train, return_counts=True))
        self.fittedFlag = True
    def listLoadFromPickle(self, pickle_name = "models.pckl"):
        loaded_list = []
        with open(pickle_name, "rb") as f:
            while True:
                try:
                    loaded_list.append(pickle.load(f))
                except EOFError:
                    break
        print("*"*20, "model was loaded")
        #self.fittedFlag = True
        return loaded_list
    def predict(self, predictions_test, scores = None, debug = 1):
        if np.all(scores) == None:
            scores = self.scores
        if debug > 0:
            deb.prints(self.threshold)
            deb.prints(predictions_test.shape)

#            print("*"*20, " Flattening the results")

#        predictions_test = predictions_test.flatten()
        if debug > 0:
            deb.prints(predictions_test.shape)

#            print("*"*20, " Flattened the results")

        predictions_test[scores < self.threshold] = 40 #self.loco_class + 1
        
        if debug > -1:
            deb.prints(np.unique(predictions_test, return_counts=True))
        #pdb.set_trace()
        return predictions_test

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
    def makeCovMatrixIdentitySet(self, makeCovMatrixIdentityFlag):
        self.makeCovMatrixIdentityFlag = makeCovMatrixIdentityFlag
    
    def predictScores(self, predictions_test, open_features, debug=1):
        self.scores = np.zeros_like(predictions_test, dtype=np.float)
        if debug>0:
            print('*'*20, 'predict_unknown_class')
            deb.prints(self.model_list)
        covariance_matrix_list = self.covariance_matrix_list.copy()
        if debug>0:          
            deb.prints(np.unique(predictions_test, return_counts=True))
            deb.prints(self.known_classes)
        for idx, c in enumerate(self.known_classes):
            c = c - 1
            if debug>0:
                print('idx, class', idx, c)            
                deb.prints(predictions_test.shape)
                deb.prints(np.unique(predictions_test))
                deb.prints(c)
            feat_msk = (predictions_test == c)
            
            if debug>0:            
                deb.prints(np.unique(feat_msk,return_counts=True))
                print("open_features stats",np.min(open_features),np.average(open_features),np.max(open_features))

            ##print("Model components",self.model_list[idx].components_)
#            deb.stats_print(open_features)
            
            if np.any(feat_msk):
                #try:
                if debug>0:                
                    deb.prints(open_features.shape)
                    deb.prints(feat_msk.shape)
                    deb.prints(open_features[feat_msk, :].shape)
                #mahalanobis_threshold = True
                if self.mahalanobis_threshold==False:
#                    deb.prints(np.round(linalg.pinvh(self.model_list[idx].get_precision(), check_finite=False), 2))
#                    deb.prints(self.model_list[idx].get_precision().shape)
#                    pdb.set_trace()
                    
                    self.scores[feat_msk] = self.model_list[idx].score_samples(open_features[feat_msk, :])
                    if debug>0:
                        deb.prints(self.model_list[idx].score(open_features[feat_msk, :]))
                    
                    '''
                    # for comparison
                    features_pca = self.model_list[idx].transform(open_features[feat_msk, :])
                    deb.prints(self.avgLogLikelihoodGet(features_pca, 
                            self.covariance_matrix_list[idx]))
                    '''
                    # mahalanobis threshold from framework loglikelihood
                    #self.scores[feat_msk] = self.mahalanobisFromLogLikelihood(self.scores[feat_msk], 
                    #            self.covariance_matrix_list[idx])
                else:
                    features_class = open_features[feat_msk, :]
                    features_pca = self.model_list[idx].transform(features_class)
                    avg_features_pca = np.average(features_pca, axis = 0)
                    if debug > 0:
                        ic(np.round(avg_features_pca, 2)) # average over all samples
                    #self.myLogLikelihoodFlag = True
                    scores_class = np.zeros(features_class.shape[0])
                    if self.myLogLikelihoodFlag == False:
                        
                        for sample_id in range(features_class.shape[0]):
                            #scores_class[sample_id] = self.mahalanobis_distance(
                            #    features_pca[sample_id], self.covariance_matrix_list[idx])
#                            scores_class[sample_id] = mahalanobis(features_pca[sample_id],
#                                        np.zeros_like(features_pca[sample_id]),
#                                        self.covariance_matrix_list[idx])    
                            scores_class[sample_id] = self.mahalanobis_distance2(
                                features_pca[sample_id], self.covariance_matrix_list[idx])                    
                    elif self.myLogLikelihoodFlag == True:

                        #makeCovMatrixIdentityFlag = True
                        if self.makeCovMatrixIdentityFlag == True:
                            features_pca, covariance_matrix_list[idx] = self.makeCovMatrixIdentity(
                                    features_pca,
                                    self.covariance_matrix_list[idx])
                        else:
                            covariance_matrix_list[idx] = self.covariance_matrix_list[idx].copy()

                        #for sample_id in range(features_class.shape[0]):
                            
                        #scores_class = self.logLikelihoodGet(
                        #    features_pca, covariance_matrix_list[idx])
                        #scores_class = self.score_mahalanobis(features_pca, 
                        #        covariance_matrix_list[idx])                            
                        scores_class = self.score_loglike(features_pca, 
                                covariance_matrix_list[idx])

                    self.scores[feat_msk] = scores_class
                    if debug>0:
                        print("scores_class stats min, avg, max, std",np.min(self.scores[feat_msk]),
                            np.average(self.scores[feat_msk]),np.max(self.scores[feat_msk]),np.std(self.scores[feat_msk]))
                        deb.prints(self.scores.shape)
                #print("self.scores stats min, avg, max",np.min(self.scores[feat_msk]),
                #    np.average(self.scores[feat_msk]),np.max(self.scores[feat_msk]))
                #deb.stats_print(self.scores[feat_msk])
#                pdb.set_trace()

                #except:
                #    print("No samples in class",c,"score is 0")
                 #   self.scores[feat_msk] = 0
        self.scores[np.isneginf(self.scores)] = -600
        if debug>0:                 
            print("scores stats min, avg, max, std",np.min(self.scores),
                    np.average(self.scores),np.max(self.scores),np.std(self.scores))
            ic(self.scores.shape)
            
        self.scoresNotCalculated = False
            
    def predict_unknown_class(self, predictions_test, open_features, debug=1): # self.model_list, self.threshold
        deb.prints(self.threshold)

        predictions_test[self.scores < self.threshold] = 40 #self.loco_class + 1
#        
        return predictions_test, _ 

    def mahalanobis_distance(self, feature, covariance_matrix): 
        # covariance_matrix shape: (16, 16)
        
        feature = np.expand_dims(feature, axis=0) # shape: (1, 16)
#        deb.prints(feature.shape)
#        deb.prints(covariance_matrix.shape)
#        deb.prints(np.matrix.transpose(feature).shape)
        out = np.matmul(feature, np.linalg.inv(covariance_matrix)) # shape: (1, 16)
#        deb.prints(out.shape)
        out = np.squeeze(np.matmul(out, np.matrix.transpose(feature))) # shape: (1)
#        deb.prints(out.shape)
#        deb.prints(out)

        return out
    def mahalanobis_distance2(self, feature, covariance_matrix): 
        # covariance_matrix shape: (16, 16)
        
        feature = np.expand_dims(feature, axis=0) # shape: (1, 16)
#        deb.prints(feature.shape)
#        deb.prints(covariance_matrix.shape)
#        deb.prints(np.transpose(feature).shape)
        out = np.dot(feature, np.linalg.inv(covariance_matrix)) # shape: (1, 16)
#        out1 = np.matmul(feature, np.linalg.inv(covariance_matrix))
#        deb.prints(out.shape)
        out = np.squeeze(np.dot(out, np.transpose(feature))) # shape: (1)
#        out1 = np.squeeze(np.matmul(out1, np.matrix.transpose(feature))) # shape: (1)
#        deb.prints(out.shape)
#        deb.prints(out)
#        deb.prints(out1)

        return out
    def mahalanobis_distance3(self, feature, covariance_matrix): 
        # covariance_matrix shape: (16, 16)
        
#        feature = np.expand_dims(feature, axis=0) # shape: (1, 16)
#        deb.prints(feature.shape)
#        deb.prints(covariance_matrix.shape)
#        deb.prints(np.transpose(feature).shape)
        out = np.dot(feature, np.linalg.inv(covariance_matrix)) # shape: (n, 16)
#        deb.prints(out.shape)
        out = np.dot(out, np.transpose(feature)) # shape: (n, n)
        deb.prints(out.shape)
        out = np.squeeze(np.diag(out)) # shape: (1)
        deb.prints(out.shape)

#        deb.prints(out)
#        deb.prints(out1)

        return out
    def logLikelihoodGet(self, feature, covariance_matrix):
        n = feature.shape[1] # 16
        #deb.prints(feature.shape)
        deb.prints(n)
        distance = np.zeros((feature.shape[0]))
        for sample_id in range(feature.shape[0]):
            distance[sample_id] = self.mahalanobis_distance2(feature[sample_id], covariance_matrix)

#        distance = self.mahalanobis_distance3(feature, covariance_matrix)

#        distance = np.power(mahalanobis(feature,
#                                        np.zeros_like(feature),
#                                        covariance_matrix), 2)
        out = ( 1 / ( np.power(2*np.pi, n/2) * np.sqrt(np.linalg.det(covariance_matrix)) ) ) * np.exp(- distance / 2) 
        out = np.log(out)
        return out

    def score_loglike(self, Xr, covariance_matrix):
        """Return the log-likelihood of each sample.
        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        Returns
        -------
        ll : ndarray of shape (n_samples,)
            Log-likelihood of each sample under the current model.
        """

        inv_covariance_matrix = linalg.pinvh(covariance_matrix, check_finite=False)
#        X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
#        Xr = X - mean_ # data is predicted thus its mean should have been subtracted?
        n_features = Xr.shape[1]
        #precision = self.get_precision()
        log_like = -.5 * (Xr * (np.dot(Xr, inv_covariance_matrix))).sum(axis=1)
        log_like -= .5 * (n_features * log(2. * np.pi) -
                          fast_logdet(inv_covariance_matrix))
        return log_like
    def score_mahalanobis(self, Xr, covariance_matrix, mean_):
        """Return the log-likelihood of each sample.
        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        Returns
        -------
        ll : ndarray of shape (n_samples,)
            Log-likelihood of each sample under the current model.
        """

        inv_covariance_matrix = linalg.pinvh(covariance_matrix, check_finite=False)
#        X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
#        Xr = X - mean_ # data is predicted thus its mean should have been subtracted?
        n_features = X.shape[1]
        #precision = self.get_precision()
        mahalanobis = (Xr * (np.dot(Xr, inv_covariance_matrix))).sum(axis=1)
        return mahalanobis

    def mahalanobisFromLogLikelihood(self, log_likelihood_values, covariance_matrix):
        n_components = covariance_matrix.shape[0]
        c = np.log(np.linalg.det(covariance_matrix))/2 + n_components*np.log(2*np.pi)
        x =-2*(log_likelihood_values + c)
        print("before sqrt min, avg, max, std", np.min(x), np.average(x), np.max(x), np.std(x))
        mahalanobis = np.sqrt(x) 
        print("mahalanobis min, avg, max, std", np.min(mahalanobis), np.average(mahalanobis), np.max(mahalanobis), np.std(mahalanobis))
        
        return mahalanobis

    def avgLogLikelihoodGet(self, features, covariance_matrix):
        N = features.shape[0] # n. of samples
        D = features.shape[1] # 16

        mu = np.average(features, axis = 0) #1x16

        distance = 0
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        for idx in range(N):
            a = (features[idx] - mu)
            distance = distance + np.dot(np.dot(a, inv_covariance_matrix), np.transpose(a))
        avgLogLikelihood = - N*D*np.log(2*np.pi)/2 - N*np.log(np.linalg.det(covariance_matrix))/2 - distance/2
        
        return avgLogLikelihood

    def avgLogLikelihoodGet2(self, features, covariance_matrix):
        N = features.shape[0] # n. of samples
        D = features.shape[1] # 16

        mu = np.average(features, axis = 0) #1x16

        S = 0
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        for idx in range(N):
            a = (features[idx] - mu)
            S = S + np.dot(a, np.transpose(a))

            #distance = distance + np.dot(np.dot(a, inv_covariance_matrix), np.transpose(a))
        avgLogLikelihood = - N*D*np.log(2*np.pi)/2 - N*np.log(np.linalg.det(covariance_matrix))/2 - distance/2
        
        return avgLogLikelihood

    def makeCovMatrixIdentity(self, features, covariance_matrix):
        #features = np.expand_dims(features, axis=0)
        #deb.prints(features.shape)
        features = np.dot(features, fractional_matrix_power(covariance_matrix, -1/2)) # 1x16 * 16x16
#        deb.prints(features.shape)
#        pdb.set_trace()
        return features, np.eye(covariance_matrix.shape[0])

    def setModelSaveNameID(self, dataset="", seq_date=""):
        self.nameID = self.name 
        if self.makeCovMatrixIdentityFlag:
            self.nameID = self.nameID + "_covmatrix"
        self.nameID = self.nameID + "_" + dataset
        self.nameID = self.nameID + "_" + seq_date          

    def fit_pca_models(self, label_test, predictions_test, open_features):
        self.model_list = []
        self.covariance_matrix_list = []
        print('*'*20, 'fit_pca_models')
        for c in self.known_classes:
            c = c - 1
            print('Fitting model for class %d...' % (c))
            sys.stdout.flush()
            
            tic = time.time()
            
            # Computing PCA models from features.
            model, covariance_matrix = self.fit_pca_model_perclass(label_test, 
                                        predictions_test, open_features, c)#feat_list, true_list, prds_list, c)
            #print("Model components",model.components_)
            #print("Model components",model.mean_)
            
            self.model_list.append(model)
            self.covariance_matrix_list.append(covariance_matrix)
            #print(np.round(covariance_matrix,2))
            #print(model.explained_variance_)
            #pdb.set_trace()
            toc = time.time()
            print('    Time spent fitting model %d: %.2f' % (c, toc - tic))

        def save_list_in_pickle(list_, filename):
            with open(filename, "wb") as f:
                for model in list_:
                    pickle.dump(model, f)
            print("*"*30, "List was saved in pickle")

        #self.setModelSaveNameID()
        save_list_in_pickle(self.model_list, "models_"+self.nameID+".pckl")
        save_list_in_pickle(self.covariance_matrix_list, "covariance_matrix_list_"+self.nameID+".pckl")
            
        #predictions_test[pred_proba_max < self.threshold] = self.loco_class + 1


    def fit_pca_model_perclass(self, label_test, predictions_test, open_features, cl):
#        model = self.model_type(n_components=self.n_components, covariance_type='diag', random_state=12345)
        model = self.model_type(**self.model_type_args)

#        model = decomposition.PCA(n_components=self.n_components, random_state=12345)
        
        #deb.prints(np.unique(label_test,return_counts=True))
        #deb.prints(np.unique(predictions_test,return_counts=True))

        deb.prints(cl)
        #deb.prints(np.unique(label_test == cl))
        #deb.prints(np.unique(predictions_test == cl))
 #       deb.prints(np.unique(predictions_test == cl))
        deb.prints(open_features.shape)
        print("==== debugging")
        deb.prints(np.unique(label_test, return_counts=True))
        deb.prints(np.unique(predictions_test, return_counts=True))
        deb.prints(np.unique((label_test == cl), return_counts=True))
        deb.prints(np.unique((predictions_test == cl), return_counts=True))
        deb.prints(np.unique((label_test == cl) & (predictions_test == cl), return_counts=True))
        
        cl_feat_flat = open_features[(label_test == cl) & (predictions_test == cl), :]
        min_samples = 50
        deb.prints(cl_feat_flat.shape)
        if cl_feat_flat.shape[0]>min_samples:
            print("cl_feat_flat stats",np.min(cl_feat_flat),np.average(cl_feat_flat),np.max(cl_feat_flat))
            
            perm = np.random.permutation(cl_feat_flat.shape[0])
            deb.prints(cl_feat_flat.shape)
            deb.prints(perm.shape[0])
          
            if perm.shape[0] > 32768:
                cl_feat_flat = cl_feat_flat[perm[:32768], :]
            deb.prints(cl_feat_flat.shape)
#            pdb.set_trace()
            model.fit(cl_feat_flat)
            x_pca_train = model.transform(cl_feat_flat)
            deb.prints(x_pca_train.shape)
            ic(np.round(np.average(x_pca_train, axis = 0), 2)) # average over all samples

            covariance_matrix = np.cov(x_pca_train, rowvar = False)
            ic(np.round(covariance_matrix,2))
            ic(np.round(model.explained_variance_, 2))
            ic(np.round(model.explained_variance_ratio_.cumsum()*100, 2))
#            ic(covariance_matrix.shape)
#            deb.prints(covariance_matrix)

#            pdb.set_trace()
            return model, covariance_matrix
        else:
            print('!'*20, 'minimum samples not met for class',cl)
            return None, None
    def loadFittedModel(self, path, nameID=""):
        #
        cwd = os.getcwd()
        deb.prints(cwd)
        #pdb.set_trace()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ic(path + "models_"+nameID+".pckl")
        ic(path + "covariance_matrix_list_"+nameID+".pckl")

#        pdb.set_trace()
        self.model_list = self.listLoadFromPickle(path + "models_"+nameID+".pckl")
        self.covariance_matrix_list = self.listLoadFromPickle(path + "covariance_matrix_list_"+nameID+".pckl")
        self.fittedFlag = True

        
class OpenPCS(OpenSetMethodGaussian):
    def __init__(self, known_classes, n_components, loco_class=0):
        super().__init__(known_classes, n_components)
        self.name = 'OpenPCS'
        self.model_type = decomposition.PCA
        self.model_type_args = dict(n_components=self.n_components, random_state=12345)
        self.mahalanobis_threshold = True
        self.myLogLikelihoodFlag = True

#class OpenPCS_Mahalanobis(OpenPCS):
#    pass

class OpenGMMS(OpenSetMethodGaussian):
    def __init__(self, known_classes, n_components, loco_class=0):
        super().__init__(known_classes, n_components)
        self.name = 'OpenGMMS'
        self.model_type = mixture.GaussianMixture
        #self.covariance_type = 'diag'
        self.covariance_type = 'full'
        
        self.model_type_args = dict(n_components=self.n_components, 
            covariance_type=self.covariance_type, random_state=12345)
#        self.model_type_args = dict(n_components=self.n_components,random_state=12345)

        self.mahalanobis_threshold = False

    def fit_pca_model_perclass(self, label_test, predictions_test, open_features, cl):
        model = self.model_type(n_components=self.n_components, random_state=12345)
#        model = decomposition.PCA(n_components=self.n_components, random_state=12345)
        
        #deb.prints(np.unique(label_test,return_counts=True))
        #deb.prints(np.unique(predictions_test,return_counts=True))

        deb.prints(cl)
        #deb.prints(np.unique(label_test == cl))
        #deb.prints(np.unique(predictions_test == cl))
 #       deb.prints(np.unique(predictions_test == cl))
        deb.prints(open_features.shape)
        print("==== debugging")
        deb.prints(np.unique(label_test, return_counts=True))
        deb.prints(np.unique(predictions_test, return_counts=True))
        deb.prints(np.unique((label_test == cl), return_counts=True))
        deb.prints(np.unique((predictions_test == cl), return_counts=True))
        deb.prints(np.unique((label_test == cl) & (predictions_test == cl), return_counts=True))
        
        cl_feat_flat = open_features[(label_test == cl) & (predictions_test == cl), :]
        min_samples = 50
        deb.prints(cl_feat_flat.shape)
        if cl_feat_flat.shape[0]>min_samples:
            print("cl_feat_flat stats",np.min(cl_feat_flat),np.average(cl_feat_flat),np.max(cl_feat_flat))
            
            perm = np.random.permutation(cl_feat_flat.shape[0])
            
            if perm.shape[0] > 32768:
                cl_feat_flat = cl_feat_flat[perm[:32768], :]
            deb.prints(cl_feat_flat.shape)
            model.fit(cl_feat_flat)
 

##            ic(np.round(model.explained_variance_, 2))
##            ic(np.round(model.explained_variance_ratio_.cumsum()*100, 2))
#            ic(covariance_matrix.shape)
#            deb.prints(covariance_matrix)

#            pdb.set_trace()
            return model, None
        else:
            print('!'*20, 'minimum samples not met for class',cl)
            return None, None

    def fit_pca_models(self, label_test, predictions_test, open_features):
        self.model_list = []

        print('*'*20, 'fit_pca_models')
        for c in self.known_classes:
            c = c - 1
            print('Fitting model for class %d...' % (c))
            sys.stdout.flush()
            
            tic = time.time()
            
            # Computing PCA models from features.
            model, _ = self.fit_pca_model_perclass(label_test, 
                                        predictions_test, open_features, c)#feat_list, true_list, prds_list, c)
            #print("Model components",model.components_)
            #print("Model components",model.mean_)
            
            self.model_list.append(model)
            #print(np.round(covariance_matrix,2))
            #print(model.explained_variance_)
            #pdb.set_trace()
            toc = time.time()
            print('    Time spent fitting model %d: %.2f' % (c, toc - tic))

        def save_list_in_pickle(list_, filename = "models.pckl"):
            with open(filename, "wb") as f:
                for model in list_:
                    pickle.dump(model, f)
            print("*"*30, "List was saved in pickle")
        save_list_in_pickle(self.model_list, "models_gmm.pckl")
            
        #predictions_test[pred_proba_max < self.threshold] = self.loco_class + 1
    def predictScores(self, predictions_test, open_features, debug=1):
        self.scores = np.zeros_like(predictions_test, dtype=np.float)

        print('*'*20, 'predict_unknown_class')
        deb.prints(self.model_list)
        deb.prints(np.unique(predictions_test, return_counts=True))
        deb.prints(self.known_classes)
        for idx, c in enumerate(self.known_classes):
            c = c - 1
            print('idx, class', idx, c)
            
            deb.prints(predictions_test.shape)
            feat_msk = (predictions_test == c)
            
            deb.prints(np.unique(feat_msk,return_counts=True))
            print("open_features stats",np.min(open_features),np.average(open_features),np.max(open_features))
            ##print("Model components",self.model_list[idx].components_)
#            deb.stats_print(open_features)
            if np.any(feat_msk):
                #try:
                
                deb.prints(open_features.shape)
                deb.prints(feat_msk.shape)

                deb.prints(open_features[feat_msk, :].shape)
                
                self.scores[feat_msk] = self.model_list[idx].score_samples(open_features[feat_msk, :])
                deb.prints(self.model_list[idx].score(open_features[feat_msk, :]))

                print("scores_class stats min, avg, max, std",np.min(self.scores[feat_msk]),
                    np.average(self.scores[feat_msk]),np.max(self.scores[feat_msk]),np.std(self.scores[feat_msk]))
                deb.prints(self.scores.shape)

        self.scores[np.isneginf(self.scores)] = -600
                 
        print("scores stats min, avg, max, std",np.min(self.scores),
                np.average(self.scores),np.max(self.scores),np.std(self.scores))
        self.scoresNotCalculated = False
