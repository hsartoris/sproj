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

if len(sys.argv) < 4:
    print("Usage: extractData.py runId prefix outdir")
    sys.exit()

runId = sys.argv[1]
prefix = sys.argv[2] + "/"
outDir = sys.argv[3] + "/" + runId + "/"
if not os.path.exists(outDir): os.mkdir(outDir)
struct = np.loadtxt(prefix + "struct.csv", delimiter=',')
n = struct.shape[0]

# timesteps
b = 10
d = 6

dataIdx = 5

testing = seqData(dataIdx, dataIdx+1, prefix, b)

_data = tf.placeholder(tf.float32, [None, b, n])
_labels = tf.placeholder(tf.float32, [None, 1, n*n])
m = Model(b, d, n, _data, _labels, batchSize, matDir = ckptDir + runId + "/")

init = tf.global_variables_initializer()

#-----------------------------MODEL EXECUTION------------------------------------

sess = tf.Session()
sess.run(init)

dataX, dataY, _ = testing.next(batchSize)
out0, out1, outf, pred = sess.run([m.layer0, m.layer1, m.layerFinal, m.prediction],
        feed_dict={_data:dataX})

data = np.loadtxt(prefix + "spikes/" + str(dataIdx) + ".csv", 
    delimiter=',')[:,:b].transpose()
print("Writing output data to", outDir)
np.savetxt(outDir + "input", data, delimiter=',')
np.savetxt(outDir + "out0", out0[0], delimiter=',')
np.savetxt(outDir + "out1", out1[0], delimiter=',')
np.savetxt(outDir + "outf", outf[0], delimiter=',')
np.savetxt(outDir + "pred", pred[0], delimiter=',')
sess.close()
