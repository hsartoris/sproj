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
        weights_stddev = .5
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
        #### for mods made to layer1revised
        self.weights['layer1'][0] = tf.Variable(tf.random_normal([2,1],
            stddev=weights_stddev))
        self.weights['layer1'][1] = tf.Variable(tf.random_normal([2,1],
            stddev=weights_stddev))
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
    def layer0(self):
        # print(<thing>.get_shape().as_list()
        total = tf.concat([tf.einsum('ijk,kl->ijl',self.data,self.expand), 
            tf.tile(self.data,[1,1,self.n])], 1)
        return tf.nn.relu(tf.add(tf.einsum('ij,kjl->kil', self.weights['layer0'], total),
                tf.tile(self.biases['layer0'], 
                [self.batchSize,1,self.n*self.n])))

    @lazy_property
    def layer1revised(self):
        #convolutional
        data = self.layer0
        dataFlip = tf.reshape(tf.transpose(data, [0,2,1]),
                [self.batchSize, self.n*self.n, self.d, 1])

        horizCompress = tf.einsum('ijk,kl->ijl', data, tf.transpose(self.expand))
        horizCompress = tf.divide(horizCompress, self.n)
        horiz = tf.einsum('ijk,kl->ijl', horizCompress, self.tile)
        horiz = tf.reshape(tf.transpose(horiz, [0,2,1]), 
                [self.batchSize, self.n*self.n, self.d, 1])
        in_total = tf.concat([horiz, dataFlip], 3)
        in_total = tf.einsum('ijkl,lx->ijkx', in_total, self.weights['layer1'][0])
        in_total = tf.transpose(tf.reshape(in_total, 
            [self.batchSize, self.n*self.n, self.d]), [0,2,1])
        #in_total = tf.concat([horiz, data], 1)
        #in_total = tf.einsum('ij,ljk->lik', self.weights['layer1'][0], in_total)
        
        vertCompress = tf.einsum('ijk,kl->ijl', data, tf.transpose(self.tile))
        vertCompress = tf.divide(vertCompress, self.n)
        vert = tf.einsum('ijk,kl->ijl', vertCompress, self.expand)
        vert = tf.reshape(tf.transpose(vert, [0,2,1]),
                [self.batchSize, self.n*self.n, self.d, 1])
        out_total = tf.concat([dataFlip, vert], 3)
        out_total = tf.einsum('ijkl,lx->ijkx', out_total, self.weights['layer1'][1])
        out_total = tf.transpose(tf.reshape(out_total, 
            [self.batchSize, self.n*self.n, self.d]), [0,2,1])
        #out_total = tf.concat([data, vert], 1)
        #out_total = tf.einsum('ij,ljk->lik', self.weights['layer1'][1], out_total)
        return tf.nn.relu(tf.add(tf.add(in_total, out_total),
                tf.tile(self.biases['layer1'], 
                [self.batchSize,1,self.n*self.n])))

    @lazy_property
    def layer1revised2(self):
        #convolutional
        data = self.layer0
        horizCompress = tf.einsum('ijk,kl->ijl', data, tf.transpose(self.expand))
        horizCompress = tf.divide(horizCompress, self.n)
        horiz = tf.einsum('ijk,kl->ijl', horizCompress, self.tile)
        
        vertCompress = tf.einsum('ijk,kl->ijl', data, tf.transpose(self.tile))
        vertCompress = tf.divide(vertCompress, self.n)
        vert = tf.einsum('ijk,kl->ijl', vertCompress, self.expand)

        total = tf.concat([horiz, vert], 1)
        total = tf.einsum('ij,ljk->lik', self.weights['layer1'][0], total)
        total = tf.add(total, data)
        return tf.nn.relu(tf.add(total,
                tf.tile(self.biases['layer1'], 
                [self.batchSize,1,self.n*self.n])))

    @lazy_property
    def layer2revised2(self):
        #convolutional
        data = self.layer1revised2
        horizCompress = tf.einsum('ijk,kl->ijl', data, tf.transpose(self.expand))
        horizCompress = tf.divide(horizCompress, self.n)
        horiz = tf.einsum('ijk,kl->ijl', horizCompress, self.tile)
        
        vertCompress = tf.einsum('ijk,kl->ijl', data, tf.transpose(self.tile))
        vertCompress = tf.divide(vertCompress, self.n)
        vert = tf.einsum('ijk,kl->ijl', vertCompress, self.expand)

        total = tf.concat([horiz, vert], 1)
        total = tf.einsum('ij,ljk->lik', self.weights['layer2'][0], total)
        total = tf.add(total, data)
        return tf.nn.relu(tf.add(total,
                tf.tile(self.biases['layer2'], 
                [self.batchSize,1,self.n*self.n])))

    @lazy_property
    def layer3revised2(self):
        #convolutional
        data = self.layer2revised2
        horizCompress = tf.einsum('ijk,kl->ijl', data, tf.transpose(self.expand))
        horizCompress = tf.divide(horizCompress, self.n)
        horiz = tf.einsum('ijk,kl->ijl', horizCompress, self.tile)
        
        vertCompress = tf.einsum('ijk,kl->ijl', data, tf.transpose(self.tile))
        vertCompress = tf.divide(vertCompress, self.n)
        vert = tf.einsum('ijk,kl->ijl', vertCompress, self.expand)

        total = tf.concat([horiz, vert], 1)
        total = tf.einsum('ij,ljk->lik', self.weights['layer3'], total)
        total = tf.add(total, data)
        return tf.nn.relu(tf.add(total,
                tf.tile(self.biases['layer3'], 
                [self.batchSize,1,self.n*self.n])))

    @lazy_property
    def layer1(self):
        #convolutional
        data = self.layer0
        in_total = tf.concat([tf.einsum('ijk,kl->ijl', 
            tf.einsum('ijk,kl->ijl', data, tf.transpose(self.expand)), 
            self.expand), data], 1)
        in_total = tf.einsum('ij,ljk->lik', self.weights['layer1'][0], in_total)
        in_total = tf.divide(in_total, 2*self.n)
        
        out_total = tf.concat([self.layer0, tf.einsum('ijk,kl->ijl', 
            tf.einsum('ijk,kl->ijl', data, tf.transpose(self.tile)), self.tile)], 1)
        out_total = tf.einsum('ij,ljk->lik', self.weights['layer1'][1], out_total)
        out_total = tf.divide(out_total, 2*self.n)
        return tf.nn.relu(tf.add(tf.add(in_total, out_total),
                tf.tile(self.biases['layer1'], 
                [self.batchSize,1,self.n*self.n])))

    @lazy_property
    def layer2(self):
        #convolutional
        data = self.layer1
        a_total = tf.concat([tf.einsum('ijk,kl->ijl', 
            tf.einsum('ijk,kl->ijl', data, tf.transpose(self.expand)), 
            self.expand), data], 1)
        a_total = tf.einsum('ij,ljk->lik', self.weights['layer2'][0], a_total)
        a_total = tf.divide(a_total, 2 * self.n)
        b_total = tf.concat([self.layer0, tf.einsum('ijk,kl->ijl', 
            tf.einsum('ijk,kl->ijl', data, tf.transpose(self.tile)), self.tile)], 1)
        b_total = tf.einsum('ij,ljk->lik', self.weights['layer2'][1], b_total)
        b_total = tf.divide(b_total, 2 * self.n)
        return tf.nn.relu(tf.add(tf.add(a_total, b_total),
                tf.tile(self.biases['layer2'], 
                [self.batchSize,1,self.n*self.n])))
    
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
