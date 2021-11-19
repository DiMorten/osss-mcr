"""
A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
"""
from tensorflow.keras import backend as K
import tensorflow as tf
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def weighted_categorical_crossentropy_ignoring_last_label(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        log_softmax = tf.nn.log_softmax(y_pred)
        #log_softmax = tf.log(y_pred)
        #log_softmax = K.log(y_pred)

        y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1)

        cross_entropy = -K.sum(y_true * log_softmax * weights , axis=1)
        loss = K.mean(cross_entropy)

        return loss
    
    return loss
# 

def categorical_focal_ignoring_last_label(alpha=0.25,gamma=2):
    """
    Focal loss implementation
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    #alpha = K.variable(alpha)
    #gamma = K.variable(gamma)
    def loss(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        y_pred_softmax = tf.nn.softmax(y_pred) # I should do softmax before the loss
        #log_softmax = tf.nn.log_softmax(y_pred)
        #log_softmax = tf.log(y_pred)
        #log_softmax = K.log(y_pred)
        y_true = K.one_hot(tf.cast(K.flatten(y_true), tf.int32), K.int_shape(y_pred)[-1]+1)
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1)
        focal_term = alpha * K.pow(1. - y_pred_softmax, gamma)
        cross_entropy = -K.sum(focal_term * y_true * K.log(y_pred_softmax), axis=1)
        loss = K.mean(cross_entropy)

        return loss
    
    return loss
def KL(alpha, K):
    beta=tf.constant(np.ones((1,K)),dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha,axis=-1,keepdims=True)
    
    KL = tf.reduce_sum((alpha - beta)*(tf.math.digamma(alpha)-tf.math.digamma(S_alpha)),axis=-1,keepdims=True) + \
         tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha),axis=-1,keepdims=True) + \
         tf.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(tf.reduce_sum(beta,axis=-1,keepdims=True))
    return KL

def loss_eq5(p, alpha, K, global_step, annealing_step):
    S = tf.reduce_sum(alpha, axis=-1, keepdims=True)
    loglikelihood = tf.reduce_sum((p-(alpha/S))**2, axis=-1, keepdims=True) + tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=-1, keepdims=True)
    #global_step = tf.compat.v1.train.get_global_step
    KL_reg =  tf.minimum(1.0, tf.cast(global_step/annealing_step, tf.float32)) * KL((alpha - 1)*(1-p) + 1 , K)
    return loglikelihood + KL_reg
def evidential_categorical_focal_ignoring_last_label(alpha=0.25,gamma=2, current_epoch = 0):
    """
    Focal loss implementation
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    #alpha = K.variable(alpha)
    #gamma = K.variable(gamma)
    def loss(y_true, y_pred):
        
        class_n = K.int_shape(y_pred)[-1]
        y_pred = K.reshape(y_pred, (-1, class_n))
        evidence = y_pred # I should do softmax before the loss

        y_true = K.one_hot(tf.cast(K.flatten(y_true), tf.int32), class_n + 1)
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1)

        alpha = evidence + 1
        u = class_n / tf.reduce_sum(alpha, axis= -1, keepdims=True)
        prob = alpha / tf.reduce_sum(alpha, axis = -1, keepdims=True) 
        print("Loss current epoch", current_epoch)
#        loss = loss_eq5(y_true, alpha, class_n, current_epoch, 30)
        loss = loss_eq5(y_true, alpha, class_n, current_epoch, 30)

        loss = tf.reduce_mean(loss)

        return loss
    
    return loss
def weighted_categorical_focal_ignoring_last_label(weights, alpha=0.25,gamma=2):
    """
    Focal loss implementation
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        y_pred_softmax = tf.nn.softmax(y_pred) # I should do softmax before the loss

        y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1)
        focal_term = alpha * K.pow(1. - y_pred_softmax, gamma)
        cross_entropy = -K.sum(focal_term * y_true * K.log(y_pred_softmax) * weights, axis=1)
        loss = K.mean(cross_entropy)

        return loss
    
    return loss
def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    #y_pred = K.argmax(y_pred,axis=3)
        
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))

import numpy as np
from tensorflow.keras.activations import softmax
from tensorflow.keras.losses import categorical_crossentropy

# # init tests
# samples=3
# maxlen=4
# vocab=5

# y_pred_n = np.random.random((samples,maxlen,vocab)).astype(K.floatx())
# y_pred = K.variable(y_pred_n)
# y_pred = softmax(y_pred)

# y_true_n = np.random.random((samples,maxlen,vocab)).astype(K.floatx())
# y_true = K.variable(y_true_n)
# y_true = softmax(y_true)

# # test 1 that it works the same as categorical_crossentropy with weights of one
# weights = np.ones(vocab)

# loss_weighted=weighted_categorical_crossentropy(weights)(y_true,y_pred).eval(session=K.get_session())
# loss=categorical_crossentropy(y_true,y_pred).eval(session=K.get_session())
# np.testing.assert_almost_equal(loss_weighted,loss)
# print('OK test1')


# # test 2 that it works differen't than categorical_crossentropy with weights of less than one
# weights = np.array([0.1,0.3,0.5,0.3,0.5])

# loss_weighted=weighted_categorical_crossentropy(weights)(y_true,y_pred).eval(session=K.get_session())
# loss=categorical_crossentropy(y_true,y_pred).eval(session=K.get_session())
# np.testing.assert_array_less(loss_weighted,loss)
# print('OK test2')

# # same keras version as I tested it on?
# import keras
# assert keras.__version__.split('.')[:2]==['2', '0'], 'this was tested on keras 2.0.6 you have %s' % keras.__version
# print('OK version')
