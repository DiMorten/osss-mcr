
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
import pickle
from icecream import ic 
import os
from scipy.spatial.distance import mahalanobis
ic.configureOutput(includeContext=False, prefix='[@debug] ')

class OpenSetMethod(): # abstract
    def __init__(self, loco_class):
        self.loco_class = loco_class
    def setThreshold(self, threshold):
        self.threshold = threshold
class SoftmaxThresholding(OpenSetMethod):

    def __init__(self, loco_class):
        super().__init__(loco_class)

        self.fittedFlag = True
        self.name = 'SoftmaxThresholding'

    def predict(self, predictions_test, pred_proba_test):
        # pred proba shape is (n_samples, h, w, classes)
        pred_proba_test = scipy.special.softmax(pred_proba_test, axis=-1)

        print("pred_proba_max stats min, avg, max",np.min(pred_proba_test),
                np.average(pred_proba_test),np.max(pred_proba_test))
        pred_proba_max = np.amax(pred_proba_test, axis=-1) # shape (n_samples, h, w)

        print("pred_proba_max stats min, avg, max",np.min(pred_proba_max),
                np.average(pred_proba_max),np.max(pred_proba_max))
        deb.prints(predictions_test.shape)
        deb.prints(pred_proba_max.shape)

        predictions_test[pred_proba_max < self.threshold] = 40 #self.loco_class + 1

        return predictions_test


class OpenPCS(OpenSetMethod):
    def __init__(self, loco_class, known_classes, n_components):
        super().__init__(loco_class)
        self.known_classes = known_classes
        self.n_components = n_components
        self.fittedFlag = False
        self.name = 'OpenPCS'

        
    def fit(self, label_train, predictions_train, pred_proba_train):
        # pred proba shape is (n_samples, h, w, classes)
        ##pred_proba_test = scipy.special.softmax(pred_proba_test, axis=-1)

        ##print("pred_proba_max stats min, avg, max",np.min(pred_proba_test),
        ##        np.average(pred_proba_test),np.max(pred_proba_test))
        deb.prints(self.threshold)
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
    def predict(self, predictions_test, pred_proba_test, debug = 1):
        if debug > 0:
            deb.prints(self.threshold)
            deb.prints(predictions_test.shape)

            print("*"*20, " Flattening the results")

            deb.prints(predictions_test.shape)
            deb.prints(pred_proba_test.shape)

        predictions_test = predictions_test.flatten()
        #pred_proba_test = pred_proba_test.reshape((pred_proba_test.shape[0], -1))
        #deb.prints(pred_proba_test.shape)
        if debug > 0:
            deb.prints(predictions_test.shape)
            deb.prints(pred_proba_test.shape)

            ##pred_proba_test = pred_proba_test.reshape((-1, pred_proba_test.shape[-1]))
            print("*"*20, " Flattened the results")

        predictions_test, _ = self.predict_unknown_class(predictions_test, pred_proba_test,
            debug = debug)
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

    def predict_unknown_class(self, predictions_test, open_features, debug=1): # self.model_list, self.threshold
        scores = np.zeros_like(predictions_test, dtype=np.float)
        if debug>0:
            print('*'*20, 'predict_unknown_class')
            deb.prints(self.model_list)
        for idx, c in enumerate(self.known_classes):
            if debug > 0:
                print('idx, class', idx, c)
                deb.prints(predictions_test.shape)
            feat_msk = (predictions_test == c)
            if debug > 0:
                deb.prints(np.unique(feat_msk,return_counts=True))
                print("open_features stats",np.min(open_features),np.average(open_features),np.max(open_features))
            ##print("Model components",self.model_list[idx].components_)
#            deb.stats_print(open_features)
            if np.any(feat_msk):
                #try:
                if debug > 0:
                    deb.prints(open_features.shape)
                    deb.prints(feat_msk.shape)

                    deb.prints(open_features[feat_msk, :].shape)
                mahalanobis_threshold = False
                if mahalanobis_threshold==False:
                    scores[feat_msk] = self.model_list[idx].score_samples(open_features[feat_msk, :])
                else:
                    features_class = open_features[feat_msk, :]
                    features_pca = self.model_list[idx].transform(features_class)
                    avg_features_pca = np.average(features_pca, axis = 0)
                    if debug > 0:
                        ic(np.round(avg_features_pca, 2)) # average over all samples
                    myLogLikelihoodFlag = True
                    scores_class = np.zeros(features_class.shape[0])
                    if myLogLikelihoodFlag == False:
                        
                        for sample_id in range(features_class.shape[0]):
                            #scores_class[sample_id] = self.mahalanobis_distance(
                            #    features_pca[sample_id], self.covariance_matrix_list[idx])
#                            scores_class[sample_id] = mahalanobis(features_pca[sample_id],
#                                        np.zeros_like(features_pca[sample_id]),
#                                        self.covariance_matrix_list[idx])    
                            scores_class[sample_id] = self.mahalanobis_distance2(
                                features_pca[sample_id], self.covariance_matrix_list[idx])                    
                    elif myLogLikelihoodFlag == True:
                        for sample_id in range(features_class.shape[0]):
                            scores_class[sample_id] = self.logLikelihoodGet(
                                features_pca[sample_id], self.covariance_matrix_list[idx])

                    scores[feat_msk] = scores_class

                #except:
                #    print("No samples in class",c,"score is 0")
                 #   scores[feat_msk] = 0
        if debug > 0:            
            print("scores stats min, avg, max",np.min(scores),
                    np.average(scores),np.max(scores))
            deb.prints(self.threshold)


        #scaler = MinMaxScaler()
        #scores = np.squeeze(scaler.fit_transform(scores.reshape(1, -1)))

        #print("scores stats min, avg, max",np.min(scores),
        #        np.average(scores),np.max(scores))
        #deb.prints(scores.shape)
        predictions_test[scores < self.threshold] = 40 #self.loco_class + 1

        return predictions_test, scores #scores in case you want to threshold them again

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
    def logLikelihoodGet(self, feature, covariance_matrix):
        n = feature.shape[0] # 16
        #deb.prints(feature.shape)
        #deb.prints(n)
        distance = self.mahalanobis_distance(feature, covariance_matrix)
#        distance = np.power(mahalanobis(feature,
#                                        np.zeros_like(feature),
#                                        covariance_matrix), 2)
        out = ( 1 / ( np.power(2*np.pi, 16/2) * np.sqrt(np.linalg.det(covariance_matrix)) ) ) * np.exp(- distance / 2) 
        out = np.log(out)
        return out


    def fit_pca_models(self, label_test, predictions_test, open_features):
        self.model_list = []
        self.covariance_matrix_list = []
        print('*'*20, 'fit_pca_models')
        for c in self.known_classes:
            
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

        def save_list_in_pickle(list_, filename = "models.pckl"):
            with open(filename, "wb") as f:
                for model in list_:
                    pickle.dump(model, f)
            print("*"*30, "List was saved in pickle")
        save_list_in_pickle(self.model_list, "models.pckl")
        save_list_in_pickle(self.covariance_matrix_list, "covariance_matrix_list.pckl")
            
        #predictions_test[pred_proba_max < self.threshold] = self.loco_class + 1


    def fit_pca_model_perclass(self, label_test, predictions_test, open_features, cl):
        model = decomposition.PCA(n_components=self.n_components, random_state=12345)
        
        #deb.prints(np.unique(label_test,return_counts=True))
        #deb.prints(np.unique(predictions_test,return_counts=True))

        deb.prints(cl)
        #deb.prints(np.unique(label_test == cl))
        #deb.prints(np.unique(predictions_test == cl))
 #       deb.prints(np.unique(predictions_test == cl))
        deb.prints(open_features.shape)
        cl_feat_flat = open_features[(label_test == cl) & (predictions_test == cl), :]
        min_samples = 50
        deb.prints(cl_feat_flat.shape)
        print("cl_feat_flat stats",np.min(cl_feat_flat),np.average(cl_feat_flat),np.max(cl_feat_flat))
        if cl_feat_flat.shape[0]>min_samples:
            
            perm = np.random.permutation(cl_feat_flat.shape[0])
            
            #if perm.shape[0] > 32768:
            #    cl_feat_flat = cl_feat_flat[perm[:32768], :]
            
            model.fit(cl_feat_flat)
            x_pca_train = model.transform(cl_feat_flat)
            deb.prints(x_pca_train.shape)
            ic(np.round(np.average(x_pca_train, axis = 0), 2)) # average over all samples

            covariance_matrix = np.cov(x_pca_train, rowvar = False)
            ic(np.round(covariance_matrix,2))
            ic(np.round(model.explained_variance_, 2))
#            ic(covariance_matrix.shape)
#            deb.prints(covariance_matrix)

#            pdb.set_trace()
            return model, covariance_matrix
        else:
            print('!'*20, 'minimum samples not met for class',cl)
            return None, None
    def loadFittedModel(self, path):
        #
        cwd = os.getcwd()
        deb.prints(cwd)
        #pdb.set_trace()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model_list = self.listLoadFromPickle(path + "models.pckl")
        self.covariance_matrix_list = self.listLoadFromPickle(path + "covariance_matrix_list.pckl")
        self.fittedFlag = True
        

     

