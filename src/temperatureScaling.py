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

'''  

import scipy
from sklearn.metrics import f1_score
from stellargraph.calibration import plot_reliability_diagram, expected_calibration_error
from sklearn.calibration import calibration_curve


def label_unknown_classes_as_id(label_test, known_classes = [0, 1, 10, 12],
    unknown_id = 20):
    unique = np.unique(label_test)
    for unique_value in unique:
        if unique_value not in known_classes:
            label_test[label_test == unique_value] = unknown_id
    ic(np.unique(label_test, return_counts = True))
    return label_test

def delete_unknown_samples(softmax, label_test):
    label_test_tmp = label_test[label_test!=unknown_id]
    ic(np.unique(label_test_tmp, return_counts = True))

    softmax_tmp = np.zeros((label_test_tmp.shape[0], softmax.shape[-1]))

    for chan in range(softmax_tmp.shape[-1]):
        softmax_tmp[..., chan] = softmax[..., chan][label_test!=unknown_id]
    label_test = label_test_tmp
    softmax = softmax_tmp
    return softmax, label_test


def vector_to_one_hot(a):
    def idx_to_incremental(a):
        unique = np.unique(a)
        for idx, value in enumerate(unique):
            a[a==value] = idx
        return a
    a = idx_to_incremental(a)
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b


#pdb.set_trace()
def get_calibration_data(softmax, label_test):
    calibration_data = []
    for i in range(softmax.shape[1]):  # iterate over classes
        calibration_data.append(
            calibration_curve(
                y_prob=softmax[:, i], y_true=label_test[:, i], n_bins=10, normalize=True
            )
        )
    return calibration_data

def get_ece(softmax, calibration_data):
    ece = []
    for i in range(softmax.shape[1]):
        fraction_of_positives, mean_predicted_value = calibration_data[i]
        ece.append(
            expected_calibration_error(
                prediction_probabilities=softmax[:, i],
                accuracy=fraction_of_positives,
                confidence=mean_predicted_value,
            )
        )
    return ece

'''  

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
        # self.T = 0.3
        self.T = 8.612723
        return logits / self.T

    '''  
    def getExpectedCalibrationCurve(self, label, logits):
        # pred_prob_test = pred_prob_flatten_test(pred_prob)


        ic(label.shape, logits.shape)
        ic(np.unique(label, return_counts=True))
        pdb.set_trace()
        softmax = scipy.special.softmax(logits, axis=-1)
        ic(softmax.shape, label_test.shape)

        label_test = label_test - 1

        unknown_id = 20

        label_test = label_unknown_classes_as_id(label_test, unknown_id = unknown_id)
        ic(softmax.shape, label_test.shape)

        softmax, label_test = delete_unknown_samples(softmax, label_test)
        ic(np.unique(label_test, return_counts = True))

        label_test = vector_to_one_hot(label_test)    
        ic(softmax.shape, label_test.shape)

        calibration_data = get_calibration_data(softmax, label_test)
        ece = get_ece(softmax, calibration_data)
        ic(ece)
        plot_reliability_diagram(calibration_data, softmax, ece=ece)
    '''  

