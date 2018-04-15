import numpy as np
import math, shutil, os
import tensorflow as tf

def loadMats(matDir, trainable):
    print("Loading matrices from", matDir)
    weights = dict()
    biases = dict()
    weights['layer0'] = tf.Variable(np.loadtxt(matDir + "/l0.weights", 
        delimiter=','), trainable = trainable[0], dtype="float32")

    weights['layer1'] = [tf.Variable(np.loadtxt(matDir + "/l1in.weights", 
        delimiter=','), trainable = trainable[1], dtype="float32"),
        tf.Variable(np.loadtxt(matDir + "/l1out.weights", delimiter=','),
            trainable = trainable[1], dtype = "float32")]

    weights['layer2'] = [tf.Variable(np.loadtxt(matDir + "/l2in.weights", 
        delimiter=','), trainable = trainable[1], dtype="float32"),
        tf.Variable(np.loadtxt(matDir + "/l2out.weights", delimiter=','),
            trainable = trainable[1], dtype = "float32")]

    weights['final'] = tf.Variable(np.loadtxt(matDir + "/final.weights", 
        delimiter=','), trainable = trainable[2], dtype = "float32")
    weights['final'] = tf.expand_dims(weights['final'], 0)

    
    biases['layer0'] = tf.Variable(np.loadtxt(matDir + "/l0.biases",
        delimiter=','), trainable = trainable[0], dtype="float32")
    biases['layer0'] = tf.expand_dims(biases['layer0'], 1)
    biases['layer0'] = tf.expand_dims(biases['layer0'], 0)

    biases['layer1'] = tf.Variable(np.loadtxt(matDir + "/l1.biases",
        delimiter=','), trainable = trainable[1], dtype="float32")
    biases['layer1'] = tf.expand_dims(biases['layer1'], 1)
    biases['layer1'] = tf.expand_dims(biases['layer1'], 0)

    biases['layer2'] = tf.Variable(np.loadtxt(matDir + "/l2.biases",
        delimiter=','), trainable = trainable[1], dtype="float32")
    biases['layer2'] = tf.expand_dims(biases['layer2'], 1)
    biases['layer2'] = tf.expand_dims(biases['layer2'], 0)
    return weights, biases

def initMats(weights_stddev, biases_stddev, d, b):
    weights = dict()
    biases = dict()
    weights['layer0'] = tf.Variable(tf.random_normal([d, 2*b], stddev=weights_stddev))

    weights['layer1'] = [tf.Variable(tf.random_normal([d, d], stddev=weights_stddev)),
        tf.Variable(tf.random_normal([d, d], stddev=weights_stddev)),
        tf.Variable(tf.random_normal([d, 2*d], stddev=.5))]

    weights['layer2'] = [tf.Variable(tf.random_normal([d, d], stddev=weights_stddev)),
        tf.Variable(tf.random_normal([d, d], stddev=weights_stddev)),
        tf.Variable(tf.random_normal([d, 2*d], stddev=.5))]
    
    weights['layer3'] = tf.Variable(tf.random_normal([d,2*d], stddev=weights_stddev))

    weights['final'] = tf.Variable(tf.random_normal([1, d], stddev=weights_stddev))

    biases['layer0'] = tf.Variable(tf.truncated_normal([1, d,1],
        stddev=biases_stddev))
    biases['layer1'] = tf.Variable(tf.truncated_normal([1, d,1],
        stddev=biases_stddev))
    biases['layer2'] = tf.Variable(tf.truncated_normal([1, d,1],
        stddev=biases_stddev))
    biases['layer3'] = tf.Variable(tf.truncated_normal([1, d,1],
        stddev=biases_stddev))
    return weights, biases

def saveMats(weights, biases, matDir, sess):
    l0 = sess.run(weights['layer0'])
    l1in = sess.run(weights['layer1'][0])
    l1out = sess.run(weights['layer1'][1])
    l1f = sess.run(weights['layer1'][2])
    lf = sess.run(weights['final'])
    l0b = sess.run(biases['layer0'])[0]
    l1b = sess.run(biases['layer1'])[0]
    np.savetxt(matDir + "/l0.weights", l0, delimiter=',')
    np.savetxt(matDir + "/l1in.weights", l1in, delimiter=',')
    np.savetxt(matDir + "/l1out.weights", l1out, delimiter=',')
    np.savetxt(matDir + "/l1f.weights", l1f, delimiter=',')
    np.savetxt(matDir + "/final.weights", lf, delimiter=',')
    np.savetxt(matDir + "/l0.biases", l0b, delimiter=',')
    np.savetxt(matDir + "/l2.biases", l2b, delimiter=',')
