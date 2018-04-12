import numpy as np
import signal, sys, math, shutil, os, functools
import tensorflow as tf

def loadMats(self, matDir, tflow = True, trainable=[True, True, True]):
    # this is a halfassed attempt at matrix loading abstraction
    weights = dict()
    biases = dict()
    print("Loading matrices from", matDir)
    weights['layer0'] = np.loadtxt(matDir + "/l0.weights", delimiter=',', 
        dtype="float32")
    weights['layer1'] = [np.loadtxt(matDir + "/l1in.weights", delimiter=',',
        dtype="float32"),
        np.loadtxt(matDir + "/l1out.weights", delimiter=',', dtype='float32')]
    weights['final'] = np.loadtxt(matDir + "/final.weights", delimiter=',', 
        dtype = "float32")

    
    biases['layer0'] = np.loadtxt(matDir + "/l0.biases", delimiter=',', 
        dtype="float32")

    biases['layer1'] = np.loadtxt(matDir + "/l1.biases", delimiter=',', 
        dtype="float32")

    if tflow:
        import tensorflow as tf
        weights['layer0'] = tf.Variable(weights['layer0'], trainable=trainable[0])
        weights['layer1'] = [tf.Variable(weights['layer1'][0], 
            trainable=trainable[1]),
            tf.Variable(weights['layer1'][1], trainable=trainable[1])]
        weights['final'] = tf.Variable(weights['final'], trainable=trainable[2])
        weights['final'] = tf.expand_dims(weights['final'], 0)

        biases['layer0'] = tf.Variable(biases['layer0'], trainable=trainable[0])
        biases['layer1'] = tf.Variable(biases['layer1'], trainable=trainable[1])

        biases['layer0'] = tf.expand_dims(biases['layer0'], 1)
        biases['layer0'] = tf.expand_dims(biases['layer0'], 0)
        biases['layer1'] = tf.expand_dims(biases['layer1'], 1)
        biases['layer1'] = tf.expand_dims(biases['layer1'], 0)
    return weights, biases


def initMats(self):
    weights['layer0'] = tf.Variable(tf.random_normal([self.d, 2*self.b]))
    weights['layer1'] = [tf.Variable(tf.random_normal([self.d, 2*self.d])),
        tf.Variable(tf.random_normal([self.d, 2*self.d]))]
    weights['final'] = tf.Variable(tf.random_normal([1, self.d]))

    biases['layer0'] = tf.Variable(tf.truncated_normal([1, self.d,1],
        stddev=self.initBias))
    biases['layer1'] = tf.Variable(tf.truncated_normal([1, self.d,1],
        stddev=self.initBias))

def saveMats(self, matDir, sess):
    import tensorflow as tf
    l0 = sess.run(weights['layer0'])
    l1in = sess.run(weights['layer1'][0])
    l2in = sess.run(weights['layer1'][1])
    lf = sess.run(weights['final'])
    l0b = sess.run(biases['layer0'])[0]
    l1b = sess.run(biases['layer1'])[0]
    np.savetxt(matDir + "/l0.weights", l0, delimiter=',')
    np.savetxt(matDir + "/l1in.weights", l1in, delimiter=',')
    np.savetxt(matDir + "/l1out.weights", l2in, delimiter=',')
    np.savetxt(matDir + "/final.weights", lf, delimiter=',')
    np.savetxt(matDir + "/l0.biases", l0b, delimiter=',')
    np.savetxt(matDir + "/l1.biases", l1b, delimiter=',')
        
