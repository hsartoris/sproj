import numpy as np
import signal, sys, math, shutil, os, functools
import tensorflow as tf

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
        self.initBias = .1
        if matDir is not None:
            if trainable is None: trainable = [True, True, True]
            # attempt to load matrices from previous run
            if os.path.exists(matDir):
                self.loadMats(matDir, trainable)
            else:
                print(matDir)
                print("Matrix directory not found, exiting")
                sys.exit()
        else:
            self.initMats()
        self.n = n
        self.lr = learnRate
        self.data = data
        self.labels = labels
        self.initTiles()
        self.layer0
        self.layer1
        self.layerFinal
        self.loss
        self.output0
        self.output1
        self.outputFinal
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
        self.tile = tf.constant(tile, tf.float32);

    def loadMats(self, matDir, trainable):
        print("Loading matrices from", matDir)
        self.weights['layer0'] = tf.Variable(np.loadtxt(matDir + "/l0.weights", 
            delimiter=','), trainable = trainable[0], dtype="float32")
        self.weights['layer1'] = [tf.Variable(np.loadtxt(matDir + "/l1in.weights", 
            delimiter=','), trainable = trainable[1], dtype="float32"),
            tf.Variable(np.loadtxt(matDir + "/l1out.weights", delimiter=','),
                trainable = trainable[1], dtype = "float32")]
        self.weights['final'] = tf.Variable(np.loadtxt(matDir + "/final.weights", 
            delimiter=','), trainable = trainable[2], dtype = "float32")
        self.weights['final'] = tf.expand_dims(self.weights['final'], 0)

        
        self.biases['layer0'] = tf.Variable(np.loadtxt(matDir + "/l0.biases",
            delimiter=','), trainable = trainable[0], dtype="float32")
        self.biases['layer0'] = tf.expand_dims(self.biases['layer0'], 1)
        self.biases['layer0'] = tf.expand_dims(self.biases['layer0'], 0)

        self.biases['layer1'] = tf.Variable(np.loadtxt(matDir + "/l1.biases",
            delimiter=','), trainable = trainable[1], dtype="float32")
        self.biases['layer1'] = tf.expand_dims(self.biases['layer1'], 1)
        self.biases['layer1'] = tf.expand_dims(self.biases['layer1'], 0)

    def initMats(self):
        self.weights['layer0'] = tf.Variable(tf.random_normal([self.d, 2*self.b]))
        self.weights['layer1'] = [tf.Variable(tf.random_normal([self.d, 2*self.d])),
            tf.Variable(tf.random_normal([self.d, 2*self.d]))]
        self.weights['final'] = tf.Variable(tf.random_normal([1, self.d]))

        self.biases['layer0'] = tf.Variable(tf.truncated_normal([1, self.d,1],
            stddev=self.initBias))
        self.biases['layer1'] = tf.Variable(tf.truncated_normal([1, self.d,1],
            stddev=self.initBias))

    def saveMats(self, matDir, sess):
        l0 = sess.run(self.weights['layer0'])
        l1in = sess.run(self.weights['layer1'][0])
        l2in = sess.run(self.weights['layer1'][1])
        lf = sess.run(self.weights['final'])
        l0b = sess.run(self.biases['layer0'])[0]
        l1b = sess.run(self.biases['layer1'])[0]
        np.savetxt(matDir + "/l0.weights", l0, delimiter=',')
        np.savetxt(matDir + "/l1in.weights", l1in, delimiter=',')
        np.savetxt(matDir + "/l1out.weights", l2in, delimiter=',')
        np.savetxt(matDir + "/final.weights", lf, delimiter=',')
        np.savetxt(matDir + "/l0.biases", l0b, delimiter=',')
        np.savetxt(matDir + "/l1.biases", l1b, delimiter=',')
            

    @lazy_property
    def layer0(self):
        # print(<thing>.get_shape().as_list()
        total = tf.concat([tf.einsum('ijk,kl->ijl',self.data,self.expand), 
            tf.tile(self.data,[1,1,self.n])], 1)
        return tf.nn.relu(tf.add(tf.einsum('ij,kjl->kil', self.weights['layer0'], total),
                tf.tile(self.biases['layer0'], 
                [self.batchSize,1,self.n*self.n])))

    @lazy_property
    def layer1(self):
        data = self.layer0
        a_total = tf.concat([tf.einsum('ijk,kl->ijl', 
            tf.einsum('ijk,kl->ijl', data, tf.transpose(self.expand)), 
            self.expand), data], 1)
        a_total = tf.einsum('ij,ljk->lik', self.weights['layer1'][0], a_total)
        b_total = tf.concat([self.layer0, tf.einsum('ijk,kl->ijl', 
            tf.einsum('ijk,kl->ijl', data, tf.transpose(self.tile)), self.tile)], 1)
        b_total = tf.einsum('ij,ljk->lik', self.weights['layer1'][1], b_total)
        return tf.nn.relu(tf.add(tf.add(a_total, b_total),
                tf.tile(self.biases['layer1'], 
                [self.batchSize,1,self.n*self.n])))
    
    @lazy_property
    def layerFinal(self):
        return tf.einsum('ij,ljk->lik', self.weights['final'], self.layer1)

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
        return tf.nn.tanh(self.layerFinal)

    @lazy_property
    def loss(self):
        return tf.reduce_mean(tf.losses.mean_squared_error(self.labels, 
            self.prediction, reduction=tf.losses.Reduction.NONE))

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        #return optimizer.minimize(self.loss, global_step=global_step)
        return optimizer.minimize(self.loss)

#correct = tf.equal(tf.argmax(pred, 1), tf.argmax(_labels, 1))
