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

batchSize = 1
prefix = "dataStaging/3neur16k/"
ckptDir = "model/checkpoints/"

if len(sys.argv) < 3:
    print("Usage: extractData.py <runId> <outdir>")
    sys.exit()

# timesteps
b = 8
# metalayers
d = 8
# number of neurons
n = 3

testing = seqData(0, 1, prefix, b)

_data = tf.placeholder(tf.float32, [None, b, n])
_labels = tf.placeholder(tf.float32, [None, 1, n*n])
m = Model(b, d, n, _data, _labels, batchSize, matDir = ckptDir + sys.argv[1] + "/")

init = tf.global_variables_initializer()

#-----------------------------MODEL EXECUTION------------------------------------

sess = tf.Session()
sess.run(init)

dataX, dataY, _ = testing.next(batchSize)
out0, out1, outf, pred = sess.run([m.layer0, m.layer1, m.layerFinal, m.prediction],
        feed_dict={_data:dataX})

outDir = sys.argv[2] + "/" + sys.argv[1] + "/"
if not os.path.exists(outDir): os.mkdir(outDir)
np.savetxt(outDir + "out0", out0[0], delimiter=',')
np.savetxt(outDir + "out1", out1[0], delimiter=',')
np.savetxt(outDir + "outf", outf[0], delimiter=',')
np.savetxt(outDir + "pred", pred[0], delimiter=',')
