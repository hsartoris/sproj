#!/usr/bin/python3
import numpy as np
import signal
import sys
import tensorflow as tf
import model.scripts.prettify
import math
import time
import shutil
from SeqData import seqData
from model.model import Model

EXIT = False
SAVE_CKPT = True
# if you set this to False it will break
TBOARD_LOG = True

runNumber = 2
batchSize = 64
baseRate = .0001
initLearningRate = .0025
#initLearningRate = 0.01 - baseRate
trainingSteps = 10000
prefix = "dataStaging/3neur16k"
ckptDir = "model/checkpoints/"
pretty = prettify.pretty()
logPath = "/home/hsartoris/tflowlogs/"


# timesteps
b = 10
# metalayers. let's try restricting to 1
d = 10
# number of neurons
n = 3

localtime = time.localtime()
runId = str(localtime().tm_mday) + "_" + str(localtime().tm_hour 
	+ str(localtime().tm_min)

training = None
validation = None
testing = None

def loadData():
	global training
	global validation
	global testing
	# 5120, 6400, 8000
	# 1280, 1600, 2000
	trainMaxIdx = 10240
	validMaxIdx = 12800
	testMaxIdx  = 16000
	training = seqData2(0, trainMaxIdx, prefix, b)
	validation = seqData2(trainMaxIdx, validMaxIdx, prefix, b)
	testing = seqData2(validMaxIdx, testMaxIdx, prefix, b)

def signal_handler(signal, frame):
	EXIT = True
signal.signal(signal.SIGINT, signal_handler)

def cleanup():
	clean = input("Clean up log dir? [Y/n]") or "Y"
	if clean == "Y":
        shutil.rmtree(logPath + "/train" + runId)
        shutil.rmtree(logPath + "/validation" + runId)

def makePred(checkDir=None):
	if checkDir is not None:
    	if not SAVE_CKPT: saver = tf.train.Saver()
		f = open(checkDir + "latest")
    	saver.restore(sess, checkDir + f.readline + ".ckpt")
		f.close()
    	testing = seqData2(0, 5, prefix, b)
    	testData = testing.data
    	testLabels = testing.labels
	testX, testY, _ = testing.next(1)
	print(sess.run(model.prediction, 
		feed_dict={data: testX, _labels: testY}).reshape(n,n))
    #dumpData(logPath + "/checkpoints" + runId)
    sys.exit()

# don't load data if given args
if len(sys.argv) == 1: loadData()

m = model.Model(b, d, n)

init = tf.global_variables_initializer()

if TBOARD_LOG:
    lossSum = tf.summary.scalar("train_loss", m.loss)

if SAVE_CKPT:
    saver = tf.train.Saver()
	localtime = time.localtime()
	saveDir = cpktDir + str(localtime.tm_mday) + "_" + str(localtime.tm_hour) 
		+ str(localtime.tm_min) + "/"
	os.mkdir(saveDir)

with tf.Session() as sess:
    sess.run(init)

    if TBOARD_LOG:
        # initialize log writers
        summWriter = tf.summary.FileWriter(logPath + "/train" + runId)
        validWriter = tf.summary.FileWriter(logPath + "/validation" + runId)

    if len(sys.argv) > 1 and sys.argv[1] == "pred":
		makePred(cpktDir + sys.argv[2])
		sys.exit()

    for step in range(trainingSteps):
		if EXIT:
			makePred()
			cancelRun = input("Stop training? [Y/n]") or "Y"
			if cancelRun == "Y":
				sys.exit()
			EXIT = False

        batchX, batchY, batchId = training.next(batchSize)
        sess.run(m.optimize, feed_dict={data: batchX, labels: batchY})
        if batchId == trainMaxIdx or step == 1:
			# end of epoch
            # calculate current loss on training data
            tLoss, tAcc, loss= sess.run([lossSum, accSum, m.loss], 
                    feed_dict={data: batchX, labels: batchY})
            # calculate validation loss
            validX, validY, _ = validation.next(batchSize*2)
            vloss, vacc, loss = sess.run([lossSum, accSum, m.loss], 
                    feed_dict={data: validX, labels: validY})

            if TBOARD_LOG:
                # log various data
                 validWriter.add_summary(vloss, step)
                 summWriter.add_summary(tLoss, step)

            print("Step " + str(step) + ", batch loss = " + "{:.4f}".format(loss))

        pretty.arrow(batchId, trainMaxIdx)

        if step % 500 == 0 and SAVE_CKPT:
            save = saver.save(sess, saveDir + "/" + str(step) + ".ckpt")
			f = open(saveDir + "latest", "w+")
			print(step, file=f)
			f.close()
            print("Saved checkpoint " + str(step))

    if SAVE_CKPT:
        save = saver.save(sess, saveDir + "final.ckpt")
		f = open(saveDir + "latest", "w+")
		print("final", file=f)
		f.close()
        print("Training complete; model saved in file %s" % save)
