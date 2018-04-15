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
        self.initTiles()
        self.layer0
        self.lessThanN2
        self.layer2
        self.layerFinal
        self.loss
        self.prediction
        self.optimize

    def initTiles(self):
        n = self.n
        expand = np.array([[1]*n + [0]*n*(n-1)])
        for i in range(1, n):
            expand = np.append(expand, [[0]*n*i + [1]*n + [0]*n*(n-1-i)], 0)
        self.expand = tf.constant(expand, tf.float32)
        
        tile = np.array([([1] + [0]*(n-1))*n])
        for i in range(1, n):
            tile = np.append(tile, [([0]*i + [1] + [0]*(n-1-i))*n], 0)
        self.tile = tf.constant(tile, tf.float32)

    @lazy_property
    def layer0(self):
        # print(<thing>.get_shape().as_list()
        total = tf.concat([tf.matmul(self.data,self.expand), 
            tf.tile(self.data, [1,self.n])], 0)
        return tf.nn.relu(tf.matmul(self.weights['layer0'] , total))


    def lessThanN(self, kMat, dataIn, i, j, k):
        return k < self.n

    def perKcompute(self, kMat, dataIn, i, j, k):
        dik = tf.expand_dims(dataIn[:,i*self.n + k], 1)
        dki = tf.expand_dims(dataIn[:,k*self.n + i], 1)
        dkj = tf.expand_dims(dataIn[:,k*self.n + j], 1)
        djk = tf.expand_dims(dataIn[:,j*self.n + k], 1)
        return [tf.add(kMat, tf.concat([dik*dkj, dik*djk, dki*dkj, dki*djk], 1)),
                dataIn, i, j, tf.add(k, 1)]

    def lessThanN2(self, dataIn, idx):
        return idx < tf.square(self.n)

    def outerLoop(self, dataIn, idx):
        i = tf.floordiv(idx, self.n)
        j = tf.floormod(idx, self.n)
        k = tf.constant(0)
        kMat = tf.zeros([self.d, 4], tf.float32)
        kMat, _, _, _, k = tf.while_loop(self.lessThanN, self.perKcompute, 
            [kMat, dataIn, i, j, k])
        kMat = tf.matmul(kMat, self.weights['layer2'])
        self.dataOut = tf.concat([self.dataOut, kMat], 1)
        return [dataIn, tf.add(idx, 1)]


    @lazy_property
    def layer2(self):
        dataIn = self.layer0
        # first run the fucking inner loop because fuck everything
        k = tf.constant(0)
        kMat = tf.zeros([self.d, 4], tf.float32)
        kMat, _, _, _, k = tf.while_loop(self.lessThanN, self.perKcompute,
                [kMat, dataIn, 0, 0, k])
        self.dataOut = tf.matmul(kMat, self.weights['layer2'])
        
        idx = tf.constant(0)
        _, idx = tf.while_loop(self.lessThanN2, self.outerLoop, 
            [dataIn, idx])
        return self.dataOut


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
