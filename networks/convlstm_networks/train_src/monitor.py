
import numpy as np
from time import time
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv3D, Conv3DTranspose, AveragePooling3D
from tensorflow.keras.layers import AveragePooling2D, Flatten, BatchNormalization, Dropout, TimeDistributed, ConvLSTM2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ELU, Lambda
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, jaccard_score
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve
import pdb
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, AveragePooling2D, Bidirectional, Activation
from icecream import ic
from pathlib import Path
import cv2
import joblib
elu_alpha = 0.1
import deb
from icecream import ic
class Monitor(Callback):
    def __init__(self, validation, patience, classes, sample_validation_store=False):   
        super(Monitor, self).__init__()
        self.validation = validation 
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.classes = classes
        self.f1_history = []
        self.oa_history = []
        self.sample_validation_store = sample_validation_store   
    def on_train_begin(self, logs={}):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = 0

        
    def on_epoch_begin(self, epoch, logs={}):        
        self.pred = []
        self.targ = []

     
    def on_epoch_end(self, epoch, logs={}):

        self.getValidationData()

        f1 = np.round(f1_score(self.targ, self.pred, average=None)*100,2)
        precision = np.round(precision_score(self.targ, self.pred, average=None)*100,2)
        recall= np.round(recall_score(self.targ, self.pred, average=None)*100,2)

        #update the logs dictionary:
        mean_f1 = np.sum(f1)/self.classes
        logs["mean_f1"]=mean_f1

        self.f1_history.append(mean_f1)
        
        print(f' — val_f1: {f1}\n — val_precision: {precision}\n — val_recall: {recall}')
        print(f' — mean_f1: {mean_f1}')

        oa = np.round(accuracy_score(self.targ, self.pred)*100,2)
        print("oa",oa)        
        self.oa_history.append(oa)

        current = logs.get("mean_f1")
        if np.less(self.best, current):
            self.best = current
            self.wait = 0
            print("Found best weights at epoch {}".format(epoch + 1))
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            if self.sample_validation_store:
#                save_np_sample_as_png(val_targ[0].argmax(axis=-1).astype(np.uint8)*255, 'targ')
#                save_np_sample_as_png(val_pred[0].argmax(axis=-1).astype(np.uint8)*255, 'pred')
#                save_np_sample_as_png((val_input[0]*255.).astype(np.uint8), 'input')
                pass

        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping".format(self.stopped_epoch + 1))
        print("f1 history",self.f1_history)
        print("oa history",self.oa_history)
        

class MonitorNPY(Monitor):
    def getValidationData(self):
        val_targ = self.validation[:][1]   
        val_input = self.validation[:][0]
        val_pred = self.model.predict(val_input)

        val_pred = val_pred.argmax(axis=-1)            
#        val_target = val_targ.argmax(axis=-1)

        self.pred = val_pred.flatten()
        self.targ = val_targ.flatten()     
        targ_unique = np.unique(self.targ)
        # ignore bckdn samples
        self.pred = self.pred[self.targ!=targ_unique[-1]]
        self.targ = self.targ[self.targ!=targ_unique[-1]]


class MonitorGenerator(Monitor):
    def getValidationData(self):
#        deb.prints(range(len(self.validation)))
        for batch_index in range(len(self.validation)):
            val_targ = self.validation[batch_index][1]   
            val_pred = self.model.predict(self.validation[batch_index][0])
            #deb.prints(val_pred.shape) # was programmed to get two outputs> classif. and depth
            #deb.prints(val_targ.shape) # was programmed to get two outputs> classif. and depth
            #deb.prints(len(self.validation[batch_index][1])) # was programmed to get two outputs> classif. and depth

            val_prob = val_pred.copy()
            val_predict = np.argmax(val_prob,axis=-1)
            if batch_index == 0:
                #plot_figures(self.validation[batch_index][0],val_targ,val_predict,
                #             val_prob,self.model_dir,epoch, 
                #             self.classes,'val')
                #plot_figures_timedistributed(self.validation[batch_index][0],val_targ,val_predict,
                #             val_prob,self.model_dir,epoch, 
                #             self.classes,'val')
                pass
            val_targ = np.squeeze(val_targ)
            #ic(val_predict.shape, val_targ.shape)
            val_predict = val_predict[val_targ<self.classes]
            val_targ = val_targ[val_targ<self.classes]
            self.pred.extend(val_predict)
            self.targ.extend(val_targ)        



class MonitorNPYAndGenerator(Monitor):
    def on_epoch_begin(self, epoch, logs={}):        
        self.pred_npy = []
        self.targ_npy = []

        self.pred_generator = []
        self.targ_generator = []

    def getValidationData(self):
#        deb.prints(range(len(self.validation)))
# self.validation[0] is npy and self.validation[1] is generator (tuple)
        for batch_index in range(len(self.validation[1])):
            val_targ = self.validation[1][batch_index][1]   
            val_pred = self.model.predict(self.validation[1][batch_index][0])
            #deb.prints(val_pred.shape) # was programmed to get two outputs> classif. and depth
            #deb.prints(val_targ.shape) # was programmed to get two outputs> classif. and depth
            #deb.prints(len(self.validation[batch_index][1])) # was programmed to get two outputs> classif. and depth

            val_prob = val_pred.copy()
            val_predict = np.argmax(val_prob,axis=-1)
            if batch_index == 0:
                #plot_figures(self.validation[batch_index][0],val_targ,val_predict,
                #             val_prob,self.model_dir,epoch, 
                #             self.classes,'val')
                #plot_figures_timedistributed(self.validation[batch_index][0],val_targ,val_predict,
                #             val_prob,self.model_dir,epoch, 
                #             self.classes,'val')
                pass
            val_targ = np.squeeze(val_targ)
            #ic(val_predict.shape, val_targ.shape)
            val_predict = val_predict.flatten()
            val_targ = val_predict.flatten()
#            val_predict = val_predict[val_targ<self.classes]
#            val_targ = val_targ[val_targ<self.classes]
            self.pred_generator.extend(val_predict)
            self.targ_generator.extend(val_targ)     
            #pdb.set_trace()   
        self.pred_generator = np.asarray(self.pred_generator)
        self.targ_generator = np.asarray(self.targ_generator)

        ic(val_predict.shape, val_targ.shape)
        pdb.set_trace()

        self.pred_generator = self.pred_generator[self.targ_generator<self.classes]
        self.targ_generator = self.targ_generator[self.targ_generator<self.classes]

        pdb.set_trace()

        val_targ = self.validation[0][:][1]   
        val_input = self.validation[0][:][0]
        val_pred = self.model.predict(val_input)

        val_pred = val_pred.argmax(axis=-1)            
#        val_target = val_targ.argmax(axis=-1)

        self.pred_npy = val_pred.flatten()
        self.targ_npy = val_targ.flatten()     
        targ_unique = np.unique(self.targ_npy)
        # ignore bckdn samples
        self.pred_npy = self.pred_npy[self.targ_npy!=targ_unique[-1]]
        self.targ_npy = self.targ_npy[self.targ_npy!=targ_unique[-1]]


        pdb.set_trace()
        deb.prints(self.pred_npy.shape)
        deb.prints(self.pred_generator.shape)

        deb.prints(self.targ_npy.shape)
        deb.prints(self.targ_generator.shape)
        
        
