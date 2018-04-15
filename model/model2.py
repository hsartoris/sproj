import numpy as np
import signal, sys, math, shutil, os, functools
import tensorflow as tf
from .scripts import loadMats, initMats, saveMats

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class Model():
    def __init__(self, b, d, n, data, labels, batchSize, structure=[1,1], 
            learnRate=.0025, matDir=None, trainable=None):
        # Reference:
        # b: timesteps of input slices
        # d: metalayer size
        # n: num neurons
        # lr: learning rate
        # structure[0]: num of layer 1s (can this be >1?)
        # structure[1]: num of layer 2s (not implemented >1)
        # TODO: use structure at all

        self.batchSize = batchSize
        self.b = b
        self.d = d
        self.weights = dict()
        # for layer 1, weights['layer1]'[0] is in, and 1 is out
        self.biases = dict()
        biases_stddev = .01
        weights_stddev = .1
        self.weights['layer0'] = tf.Variable(tf.random_normal([d, 2*b], 
            stddev=weights_stddev))
        self.weights['layer2'] = tf.Variable(tf.random_normal([4,1]))
        self.weights['final'] = tf.Variable(tf.random_normal([1,d], 
            stddev=weights_stddev))
        self.n = n
        self.lr = learnRate
        self.data = data
        self.labels = labels
        self.layer0
        self.layer2
        self.layerFinal
        self.loss
        self.prediction
        self.optimize

    def lessThanN2(self, dataIn, dataOut, idx):
        return idx < tf.square(self.n)

    def lessThanN(self, kMat, dataIn, i, j, k):
        return k < self.n

    def perKcompute(self, kMat, dataIn, i, j, k):
        dik = dataIn[:,i*n + k]
        dki = dataIn[:,k*n + i]
        dkj = dataIn[:,k*n + j]
        djk = dataIn[:,j*n + k]
        return [tf.add(kMat, tf.concat([dik*dkj, dik*djk, dki*dkj, dki*djk], 1)),
                dataIn, i, j, tf.add(k, 1)]


    def outerLoop(self, dataIn, dataOut, idx):
        i = tf.floordiv(idx, self.n)
        j = tf.floormod(idx, self.n)
        k = tf.constant(0)
        kMat = tf.constant(0, shape=[self.d, 4])
        kMat, _, _, _, k = tf.while_loop(lessThanN, perKcompute, [kMat, dataIn, i, j, k])
        kMat = kMat * weights['layer2']
        return [dataIn, dataOut[:, i*n + k].assign(kMat), tf.add(idx, 1)]


    @lazy_property
    def layer2(self):
        dataIn = self.layer0
        dataOut = tf.constant(0, shape=[d, self.n*self.n])
        idx = tf.constant(0)
        _, dataOut, idx = tf.while_loop(lessThanN2, dataIn, dataOut, idx)
        return dataOut


    @lazy_property
    def layerFinal(self):
        return tf.einsum('ij,ljk->lik', self.weights['final'], self.layer2)

    @lazy_property
    def prediction(self):
        #return tf.multiply(tf.nn.tanh(self.layerFinal),1.2)
        return tf.nn.tanh(self.layerFinal)

    @lazy_property
    def loss(self):
        #return tf.reduce_mean(tf.losses.mean_squared_error(self.labels, 
        #    self.prediction, reduction=tf.losses.Reduction.NONE))
        return tf.divide(tf.reduce_sum(tf.losses.mean_squared_error(self.labels, 
            self.prediction, reduction=tf.losses.Reduction.NONE)),
            tf.reduce_sum(self.labels))

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        #return optimizer.minimize(self.loss, global_step=global_step)
        return optimizer.minimize(self.loss)

#correct = tf.equal(tf.argmax(pred, 1), tf.argmax(_labels, 1))
