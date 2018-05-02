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
            learnRate=.0025, matDir=None, trainable=None, dumb=False):
        # Reference:
        # b: timesteps of input slices
        # d: metalayer size
        # n: num neurons
        # lr: learning rate
        # structure[0]: num of layer 1s (can this be >1?)
        # structure[1]: num of layer 2s (not implemented >1)
        # TODO: use structure at all

        self.numLayers = 5
        self.dumb = dumb
        self.batchSize = batchSize
        self.b = b
        self.d = d
        self.weights = dict()
        # for layer 1, weights['layer1]'[0] is in, and 1 is out
        self.biases = dict()
        biases_stddev = .01
        weights_stddev = .05
        if matDir is not None:
            if trainable is None: trainable = [True, True, True]
            # attempt to load matrices from previous run
            if os.path.exists(matDir):
                self.weights, self.biases = loadMats(matDir, trainable,
                    self.numLayers)
            else:
                print(matDir)
                print("Matrix directory not found, exiting")
                sys.exit()
        else:
            self.weights, self.biases = initMats(weights_stddev, biases_stddev, d, b,
                    layers = self.numLayers)

        self.weights['layer0_1'] = tf.Variable(tf.random_normal([d,d],
            stddev=weights_stddev))

        self.biases['layer0_1'] = tf.Variable(tf.truncated_normal([1,d,1],
            stddev=biases_stddev))
        '''
        #### for mods made to layer1revised
        self.weights['layer1'][0] = tf.Variable(tf.random_normal([d,d],
            stddev=weights_stddev))
        self.weights['layer1'][1] = tf.Variable(tf.random_normal([d,d],
            stddev=weights_stddev))
        #this matrix controls addition of input/output to make final mat
        self.weights['layer1'].append(tf.Variable(tf.random_normal([d, 2*d],
            stddev=.5)))
        ###
        '''
        self.lr = learnRate
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.activation = tf.nn.relu
        self.n = n
        self.data = data
        self.labels = labels
        self.initTiles()
        self.layer0
        self.layer0_1
        #self.layer1 # doesn't work with new weights
        #self.layer2
        self.layer1
        self.layer1dumb
        self.layer2
        self.layerFinal
        self.loss
        self.prediction
        self.optimize

    def saveMats(self, outDir, sess):
        saveMats(self.weights, self.biases, outDir, sess, layers=self.numLayers)

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
        total = tf.concat([tf.einsum('ijk,kl->ijl',self.data,self.expand), 
            tf.tile(self.data,[1,1,self.n])], 1)
        return self.activation(tf.add(tf.einsum('ij,kjl->kil', self.weights['layer0'], 
                total), tf.tile(self.biases['layer0'], [self.batchSize,1,self.n*self.n])))

    @lazy_property
    def layer0_1(self):
        return self.activation(tf.add(tf.einsum('ij,kjl->kil', 
            self.weights['layer0_1'], self.layer0), 
           tf.tile(self.biases['layer0_1'], [self.batchSize, 1, self.n*self.n])))

    @lazy_property
    def layer1dumb(self):
        return self.activation(tf.add(tf.einsum('ij,kjl->kil', 
            self.weights['layer1'][0], self.layer0), tf.tile(self.biases['layer1'],
                [self.batchSize, 1, self.n*self.n])))

    @lazy_property
    def layer1(self):
        #convolutional
        data = self.layer0

        horizCompress = tf.einsum('ijk,kl->ijl', data, tf.transpose(self.expand))
        horizCompress = tf.divide(horizCompress, self.n)
        horiz = tf.einsum('ijk,kl->ijl', horizCompress, self.tile)
        in_total = horiz*data
        in_total = tf.einsum('ij,ljk->lik', self.weights['layer1'][0], in_total)
        
        vertCompress = tf.einsum('ijk,kl->ijl', data, tf.transpose(self.tile))
        vertCompress = tf.divide(vertCompress, self.n)
        vert = tf.einsum('ijk,kl->ijl', vertCompress, self.expand)
        out_total = vert*data
        out_total = tf.einsum('ij,ljk->lik', self.weights['layer1'][1], out_total)

        output = tf.concat([in_total, out_total], 1)
        output = tf.einsum('ij,kjl->kil', self.weights['layer1'][2], output)
        return self.activation(tf.add(output, tf.tile(self.biases['layer1'], 
                    [self.batchSize,1,self.n*self.n])))

    @lazy_property
    def layer2(self):
        #convolutional
        data = self.layer1

        horizCompress = tf.einsum('ijk,kl->ijl', data, tf.transpose(self.expand))
        horizCompress = tf.divide(horizCompress, self.n)
        horiz = tf.einsum('ijk,kl->ijl', horizCompress, self.tile)
        in_total = horiz*data
        in_total = tf.einsum('ij,ljk->lik', self.weights['layer2'][0], in_total)
        
        vertCompress = tf.einsum('ijk,kl->ijl', data, tf.transpose(self.tile))
        vertCompress = tf.divide(vertCompress, self.n)
        vert = tf.einsum('ijk,kl->ijl', vertCompress, self.expand)
        out_total = vert*data
        out_total = tf.einsum('ij,ljk->lik', self.weights['layer2'][1], out_total)

        output = tf.concat([in_total, out_total], 1)
        output = tf.einsum('ij,kjl->kil', self.weights['layer2'][2], output)
        return self.activation(tf.add(output, tf.tile(self.biases['layer2'], 
                    [self.batchSize,1,self.n*self.n])))

    @lazy_property
    def layerFinal(self):
        if self.dumb:
            return tf.einsum('ij,ljk->lik', self.weights['final'], self.layer1dumb)
        return tf.einsum('ij,ljk->lik', self.weights['final'], self.layer1)

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
        #optimizer = tf.train.AdamOptimizer(self.lr)
        #return optimizer.minimize(self.loss, global_step=global_step)
        return self.optimizer.minimize(self.loss)

#correct = tf.equal(tf.argmax(pred, 1), tf.argmax(_labels, 1))
