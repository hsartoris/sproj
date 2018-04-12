#!/usr/bin/python3
import numpy as np
import signal
import sys
import tensorflow as tf
import model.scripts.Prettify as Prettify
import math
import time
import shutil, os
from model.scripts.SeqData import seqData
from model.model import Model

if len(sys.argv) < 2:
    print("Usage: extractData.py outputDir
    Loads first spike train in dataSource")

SAVE_CKPT = True
# if you set this to False it will break
TBOARD_LOG = True

batchSize = 1
prefix = "dataStaging/3neur16k/"
ckptDir = "model/checkpoints/"
pretty = Prettify.pretty()

# timesteps
b = 8
# metalayers
d = 8
# number of neurons
n = 3

testing = seqData(0, 1, prefix, b)

_data = tf.placeholder(tf.float32, [None, b, n])
_labels = tf.placeholder(tf.float32, [None, 1, n*n])
m = Model(b, d, n, _data, _labels, batchSize, learnRate = initLearnRate)

init = tf.global_variables_initializer()

#-----------------------------MODEL EXECUTION------------------------------------

sess = tf.Session()
sess.run(init)

dataX, dataY, _ = testing.next(batchSize)
out0, out1, outf, pred = sess.run([m.layer0, m.layer1, m.layerFinal, m.pred],
        feed_dict={_data:dataX])

print(out0[0])
print(out1[0])
print(outf[0])
print(pred[0])
