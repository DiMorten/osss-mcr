
import numpy as np
from time import time
import numpy as np
import keras.backend as K
import keras
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv3D, Conv3DTranspose, AveragePooling3D
from keras.layers import AveragePooling2D, Flatten, BatchNormalization, Dropout, TimeDistributed, ConvLSTM2D
from keras.models import Model
from keras.layers import ELU, Lambda
from keras import layers
from keras import regularizers
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, jaccard_score
from sklearn.metrics import classification_report
from keras.callbacks import Callback
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve
import pdb
from keras.regularizers import l1,l2
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, AveragePooling2D, Bidirectional, Activation
from icecream import ic
from pathlib import Path
import cv2
import joblib
elu_alpha = 0.1


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

#    for batch_index in range(len(self.validation)):
        val_targ = self.validation[:][1]   
        val_input = self.validation[:][0]
        val_pred = self.model.predict(val_input)

        val_predict = val_pred.argmax(axis=-1)            
#        val_target = val_targ.argmax(axis=-1)

        self.pred = val_predict.flatten()
        self.targ = val_targ.flatten()     
        targ_unique = np.unique(self.targ)
        # ignore bckdn samples
        self.pred = self.pred[self.targ!=targ_unique[-1]]
        self.targ = self.targ[self.targ!=targ_unique[-1]]

#        ic(self.pred.shape)
#        ic(self.targ.shape)
#        ic(np.unique(self.targ, return_counts=True))       
#        ic(np.unique(self.pred, return_counts=True))       
#        pdb.set_trace()

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
        

