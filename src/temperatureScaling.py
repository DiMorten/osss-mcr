from icecream import ic
import pdb
from sklearn import model_selection
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
from loss import categorical_focal, categorical_focal_ignoring_last_label
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
from tensorflow.keras.models import load_model
class TemperatureScalingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # T_init = tf.random_normal_initializer()
        self.T = tf.Variable(
            initial_value=1., # T_init(shape=(1,), dtype="float32")
            trainable=True,
        )
    def call(self, inputs):
        return tf.divide(inputs, self.T)

def temperatureModel(inputShape):
    in_ = Input(shape = inputShape)
    out = TemperatureScalingLayer()(in_)
    model = Model(in_, out)
    ic(model.summary())
    return model

class TemperatureScaling():
    def __init__(self):
        self.model_name = 'model_best_scale.h5'
        self.fitConfidenceScaling = False
    def translateLabels(self, label):
        scale = {0: 0,
            1: 1,
            10: 2,
            12: 3,
            39: 4}
        for key in scale.keys():
            label[label == key] = scale[key]
        return label
    def fitModel(self, label, logits):
        logits = logits[label != 39]
        label = label[label != 39]
        label =  self.translateLabels(label)
        ic(label.shape, logits.shape)

        x_calibration_train, x_calibration_val, y_calibration_train, y_calibration_val = model_selection.train_test_split(
            logits, label
        )

        # idxs = range(40)
        # idxs = np.random.choice(y_calibration_train.shape[0], 80000)
        # x_calibration_train = x_calibration_train[idxs]
        # y_calibration_train = y_calibration_train[idxs]

        # idxs = range(40)
        # idxs = np.random.choice(y_calibration_val.shape[0], 40000)        
        # x_calibration_val = x_calibration_val[idxs]
        # y_calibration_val = y_calibration_val[idxs]


        ic(x_calibration_train.shape, x_calibration_val.shape, y_calibration_train.shape, y_calibration_val.shape)
        ic(np.unique(label, return_counts = True))
        calibrationModel = temperatureModel(logits.shape[-1])

        
        if self.fitConfidenceScaling == False:
            calibrationModel.load_weights(self.model_name)
        else:


            es = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001)
            mc = ModelCheckpoint(self.model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            callbacks = [es, mc]

            calibrationModel.compile(
                optimizer = Adam(lr=1e-3, beta_1=0.9),
                loss = 'sparse_categorical_crossentropy',
                # loss = categorical_focal_ignoring_last_label(alpha=0.25,gamma=2),
                metrics = ['accuracy']

            )
            history = calibrationModel.fit(x_calibration_train, 
                    y_calibration_train,
                    batch_size = 64,
                    epochs = 600,
                    validation_data = (x_calibration_val, y_calibration_val),
                    callbacks = callbacks
            )



            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc = 'upper left')
            plt.savefig('history_scaling.png', dpi=200)

        self.T = calibrationModel.layers[1].get_weights()[0]
        ic(self.T)
        
    def scale(self, logits):
        return logits / self.T

        


