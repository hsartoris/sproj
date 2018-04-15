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
        if matDir is not None:
            if trainable is None: trainable = [True, True, True]
            # attempt to load matrices from previous run
            if os.path.exists(matDir):
                self.weights, self.biases = loadMats(matDir, trainable)
            else:
                print(matDir)
                print("Matrix directory not found, exiting")
                sys.exit()
        else:
            self.weights, self.biases = initMats(weights_stddev, biases_stddev, d, b)
        self.n = n
        self.lr = learnRate
        self.data = data
        self.labels = labels
        self.initTiles()
        self.layer0
        self.layer1
        self.layer2
        self.layerFinal
        self.loss
        self.output0
        self.output1
        self.outputFinal
        self.prediction
        self.optimize

    def saveMats(self, outDir, sess):
        saveMats(self.weights, self.biases, outDir, sess)

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
    

    @lazy_property
    def layerFinal(self):
        return tf.einsum('ij,ljk->lik', self.weights['final'], self.layer1revised)

    @lazy_property
    def output0(self):
        return self.layer0

    @lazy_property
    def output1(self):
        return self.layer1

    @lazy_property
    def outputFinal(self):
        return self.layerFinal

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