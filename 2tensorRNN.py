import numpy as np
import scripts.GraphKit as gk
import sys
import tensorflow as tf
from tensorflow.contrib import rnn

batchSize = 128
numClasses = 2
numInput = 1000
timesteps = 1000
numHidden = 256
learningRate = .01
trainingSteps = 5000
prefix = "classifiertest"

class seqData(object):
	def __init__(self, minIdx = 0, maxIdx=499):
		global prefix
		self.data = []
		self.labels = []
		randIdx = minIdx
		simpIdx = minIdx
		while randIdx < maxIdx and simpIdx < maxIdx:
			if np.random.rand() < .5:
				self.data.append(gk.spikeTimeMatrix(np.loadtxt(prefix + "/simplicial/spikes/" + str(simpIdx) + ".csv", delimiter=','), 1000, 1000))
				self.labels.append([1,0])
				simpIdx += 1
			else:
				self.data.append(gk.spikeTimeMatrix(np.loadtxt(prefix + "/random/spikes/" + str(randIdx) + ".csv", delimiter=','), 1000, 1000))
				self.labels.append([0,1])
				randIdx += 1
			print(str(randIdx + simpIdx) + "\r")
		while randIdx < maxIdx:
			self.data.append(gk.spikeTimeMatrix(np.loadtxt(prefix + "/random/spikes/" + str(randIdx) + ".csv", delimiter=','), 1000, 1000))
			self.labels.append([0,1])
			randIdx += 1
			print(str(randIdx + simpIdx) + "\r")
		while simpIdx < maxIdx:
			self.data.append(gk.spikeTimeMatrix(np.loadtxt(prefix + "/simplicial/spikes/" + str(simpIdx) + ".csv", delimiter=','), 1000, 1000))
			self.labels.append([1,0])
			simpIdx += 1
			print(str(randIdx + simpIdx) + "\r")
		self.batchId = 0

	def next(self, batchSize):
		if self.batchId == len(self.data):
			self.batchId = 0
		batchData = self.data[self.batchId:min(self.batchId + batchSize, len(self.data))]
		batchLabels = self.labels[self.batchId:min(self.batchId + batchSize, len(self.data))]
		self.batchId = min(self.batchId + batchSize, len(self.data))
		return batchData, batchLabels
			
if len(sys.argv) == 1 : training = seqData(0, 349)
testing = seqData(350, 499)

_data = tf.placeholder("float", [None, timesteps, numInput])
_labels = tf.placeholder("float", [None, numClasses])


weights = { 'out': tf.Variable(tf.random_normal([numHidden, numClasses])) }
biases = { 'out': tf.Variable(tf.random_normal([numClasses])) }

def RNN(x, weights, biases):
	x = tf.unstack(x, timesteps, 1)
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
		if step % 150 == 0 or step == 1:
			loss, acc = sess.run([lossOp, accuracy], feed_dict={_data: batchX, _labels: batchY})
			print("Step " + str(step) + ", batch loss = " + "{:.4f}".format(loss) + ", accuracy = " + "{:.3f}".format(acc))
			if acc == 1.0:
				print("probably overtrained; exiting. you did this to yourself.")
				saver.save(sess, "./trained" + str(step) + ".ckpt")
				break
		print(str(step) + "\r")
		if step % 500 == 0:
			save = saver.save(sess, "./trained" + str(step) + ".ckpt")
			print("Saved checkpoint " + str(step))
	save = saver.save(sess, "./trained.ckpt")
	print("Training complete; model save in file %s" % save)
	
	testData = testing.data
	testLabels = testing.labels
	print("Accuracy on validation data:", sess.run(accuracy, feed_dict={_data: testData, _labels: testLabels}))
