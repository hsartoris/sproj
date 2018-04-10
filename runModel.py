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

EXIT = False
SAVE_CKPT = True
# if you set this to False it will break
TBOARD_LOG = True

batchSize = 64
baseRate = .0001
initLearnRate = .0025
#initLearningRate = 0.01 - baseRate
trainingSteps = 10000
prefix = "dataStaging/3neur16k/"
ckptDir = "model/checkpoints/"
pretty = Prettify.pretty()
logPath = "/home/hsartoris/tflowlogs/"
testMaxIdx = 16000
validMaxIdx = int(testMaxIdx * .8)
trainMaxIdx = int(validMaxIdx * .8)

# timesteps
b = 10
# metalayers. let's try restricting to 1
d = 10
# number of neurons
n = 3

localtime = time.localtime()
runId = str(localtime.tm_mday) + "_" + str(localtime.tm_hour) + str(localtime.tm_min)

training = None
validation = None
testing = None

def loadData():
    global training
    global validation
    global testing
    # 5120, 6400, 8000
    # 1280, 1600, 2000
    training = seqData(0, trainMaxIdx, prefix, b)
    validation = seqData(trainMaxIdx, validMaxIdx, prefix, b)
    testing = seqData(validMaxIdx, testMaxIdx, prefix, b)

def signal_handler(signal, frame):
    makePred()
    if (input("Exit? [Y/n]") or "Y") == "Y":
        cleanup()
        sess.close()
        sys.exit()
signal.signal(signal.SIGINT, signal_handler)

def cleanup():
    clean = input("Clean up logs and checkpoints? [Y/n]") or "Y"
    if clean == "Y":
        shutil.rmtree(logPath + runId)
        shutil.rmtree(ckptDir + runId)

def makePred(checkDir=None):
    global testing
    global sess
    if checkDir is not None:
        saver = tf.train.Saver()
        f = open(checkDir + "latest")
        saver.restore(sess, checkDir + f.readline() + ".ckpt")
        f.close()
        testing = seqData(0, 5, prefix, b)
        testData = testing.data
        testLabels = testing.labels
    testX, testY, _ = testing.next(1)
    print(sess.run(m.prediction, 
        feed_dict={_data: testX, _labels: testY})[0].reshape(n,n))

# don't load data if given args
if len(sys.argv) == 1: loadData()

_data = tf.placeholder(tf.float32, [None, b, n])
_labels = tf.placeholder(tf.float32, [None, 1, n*n])
m = Model(b, d, n, _data, _labels, batchSize, learnRate = initLearnRate)

init = tf.global_variables_initializer()

if TBOARD_LOG:
    lossSum = tf.summary.scalar("train_loss", m.loss)

if SAVE_CKPT and not len(sys.argv) > 1:
    saver = tf.train.Saver()
    saveDir = (ckptDir + runId + "/")
    os.mkdir(saveDir)

#-----------------------------MODEL TRAINING------------------------------------

sess = tf.Session()
sess.run(init)

if TBOARD_LOG:
    # initialize log writers
    summWriter = tf.summary.FileWriter(logPath + runId + "/train")
    validWriter = tf.summary.FileWriter(logPath + runId + "/validation")

if len(sys.argv) > 1 and sys.argv[1] == "pred":
    makePred(ckptDir + sys.argv[2] + "/")
    sys.exit()

for step in range(trainingSteps):
    batchX, batchY, batchId = training.next(batchSize)
    sess.run(m.optimize, feed_dict={_data: batchX, _labels: batchY})
    if batchId == trainMaxIdx or step == 1:
        # end of epoch
        # calculate current loss on training data
        tLoss, loss= sess.run([lossSum, m.loss], 
                feed_dict={_data: batchX, _labels: batchY})
        # calculate validation loss
        validX, validY, _ = validation.next(batchSize)
        vloss, loss = sess.run([lossSum, m.loss], 
                feed_dict={_data: validX, _labels: validY})

        if TBOARD_LOG:
            # log various data
             validWriter.add_summary(vloss, step)
             summWriter.add_summary(tLoss, step)

        print("Step " + str(step) + ", batch loss = " + "{:.4f}".format(loss))

    pretty.arrow(batchId, trainMaxIdx)

    if step % 500 == 0 and SAVE_CKPT:
        save = saver.save(sess, saveDir + "/" + str(step) + ".ckpt")
        f = open(saveDir + "latest", "w+")
        f.write(str(step))
        f.close()
        print("Saved checkpoint " + str(step))

if SAVE_CKPT:
    save = saver.save(sess, saveDir + "final.ckpt")
    f = open(saveDir + "latest", "w+")
    f.write("final")
    f.close()
    print("Training complete; model saved in file %s" % save)
sess.close()
