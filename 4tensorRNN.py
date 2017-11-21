# I know naming conventions; I have the best naming conventions. I have the best, but there is no better convention than stupid.

import numpy as np
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import subprocess

rows, columns = subprocess.check_output(['stty', 'size']).split()

batchSize = 64
numClasses = 2
numInput = 2
timesteps = 0 # needs to be initialized to maxLen after data is loaded
numHidden = 128
learningRate = .001
trainingSteps = 10000
prefix = "classifiertest2"
maxLen = 0
epochLen = 200

def stdOut(string):
	sys.stdout.write(string)
	sys.stdout.flush()

class seqData(object):
	def __init__(self, minIdx = 0, maxIdx=500):
		global prefix
		global maxLen
		global columns
		self.data = []
		self.labels = []
		randIdx = minIdx
		simpIdx = minIdx
		while randIdx < maxIdx and simpIdx < maxIdx:
			if np.random.rand() < .5:
				temp = np.loadtxt(prefix + "/simplicial/spikes/" + str(simpIdx) + ".csv", delimiter=',')
				self.labels.append([1,0])
				simpIdx += 1
			else:
				temp = np.loadtxt(prefix + "/random/spikes/" + str(randIdx) + ".csv", delimiter=',')
				self.labels.append([0,1])
				randIdx += 1
			if len(temp) > maxLen: maxLen = len(temp)
			self.data.append(temp)
			stdOut("-"*int(float(randIdx + simpIdx - (minIdx * 2))*(float(columns)-1)/((maxIdx-minIdx) * 2)) + ">\r")
		while randIdx < maxIdx:
			temp = np.loadtxt(prefix + "/random/spikes/" + str(randIdx) + ".csv", delimiter=',')
			if len(temp) > maxLen: maxLen = len(temp)
			self.data.append(temp)
			self.labels.append([0,1])
			randIdx += 1
			stdOut("-"*int(float(randIdx + simpIdx - (minIdx * 2))*(float(columns)-1)/((maxIdx-minIdx) * 2)) + ">\r")
		while simpIdx < maxIdx:
			temp = np.loadtxt(prefix + "/simplicial/spikes/" + str(simpIdx) + ".csv", delimiter=',')
			if len(temp) > maxLen: maxLen = len(temp)
			self.data.append(temp)
			self.labels.append([1,0])
			simpIdx += 1
			stdOut("-"*int(float(randIdx + simpIdx - (minIdx * 2))*(float(columns)-1)/((maxIdx-minIdx) * 2)) + ">\r")
		self.batchId = 0
		stdOut("\n")

	def crop(self):
		global maxLen
		for i in range(len(self.data)):
			self.data[i] = np.resize(self.data[i], (maxLen, 2))

	def pad(self):
		global maxLen
		for i in range(len(self.data)):
			self.data[i] = np.append(self.data[i], np.zeros((2,maxLen-len(self.data[i]))), axis=1)

	def next(self, batchSize):
		if self.batchId == len(self.data):
			self.batchId = 0
		batchData = self.data[self.batchId:min(self.batchId + batchSize, len(self.data))]
		batchLabels = self.labels[self.batchId:min(self.batchId + batchSize, len(self.data))]
		self.batchId = min(self.batchId + batchSize, len(self.data))
		return batchData, batchLabels
			
training = seqData(0, 750)
testing = seqData(750,1000)
maxLen = int(maxLen * 1.2)
#maxLen = 2000
training.pad()
testing.pad()
timesteps = maxLen

#_data = tf.placeholder("float", [None, numInput, timesteps])
_data = tf.placeholder("float", [None, timesteps, numInput])
_labels = tf.placeholder("float", [None, numClasses])


weights = { 'out': tf.Variable(tf.random_normal([numHidden, numClasses])) }
biases = { 'out': tf.Variable(tf.random_normal([numClasses])) }

def RNN(x, weights, biases):
	global maxLen
	x = tf.unstack(x, axis=1)
	lstm_cell = rnn.BasicLSTMCell(numHidden, forget_bias=1.0)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

#X, Y = iterator.get_next()

logits = RNN(_data, weights, biases)
pred = tf.nn.softmax(logits)

lossOp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=_labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
trainOp = optimizer.minimize(lossOp)

correct = tf.equal(tf.argmax(pred, 1), tf.argmax(_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
	if len(sys.argv) > 1:
		saver.restore(sess, sys.argv[1])
		testData = testing.data
		testLabels = testing.labels
		print("Accuracy on validation data:", sess.run(accuracy, feed_dict={_data: testData, _labels: testLabels}))
		sys.exit()

	for step in range(1, trainingSteps+1):
		batchX, batchY = training.next(batchSize)
		sess.run(trainOp, feed_dict={_data: batchX,_labels: batchY})
		if step % epochLen == 0 or step == 1:
			loss, acc = sess.run([lossOp, accuracy], feed_dict={_data: batchX, _labels: batchY})
			print("Step " + str(step) + ", batch loss = " + "{:.4f}".format(loss) + ", accuracy = " + "{:.3f}".format(acc))
		stdOut("-"*int(float(step%epochLen)*(float(columns)-1)/epochLen) + ">\r")
		if step % 500 == 0:
			save = saver.save(sess, "./trained" + str(step) + ".ckpt")
			print("Saved checkpoint " + str(step))
	save = saver.save(sess, "./trained.ckpt")
	print("Training complete; model saved in file %s" % save)
	testData = testing.data
	testLabels = testing.labels
	print("Accuracy on validation data:", sess.run(accuracy, feed_dict={_data: testData, _labels: testLabels}))
