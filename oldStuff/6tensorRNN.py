import numpy as np
import scripts.GraphKit as gk
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import prettify

batchSize = 128
numClasses = 2
numInput = 1000
timesteps = 1000
numHidden = 128
learningRate = .001
trainingSteps = 10000
numLayers = 3
epochLen = 150
prefix = "200neur"
pretty = prettify.pretty()

class seqData(object):
	def __init__(self, minIdx = 0, maxIdx=499, numNeurons=1000, timesteps=1000, blockLen=250):
		global prefix
		self.data = []
		self.labels = []
		rand = []
		simp = []
		for i in range(minIdx, maxIdx):
			tRand = gk.spikeTimeArray(np.loadtxt(prefix + "/random/spikes/" + str(i) + ".csv", delimiter=','), timesteps)
			tSimp = gk.spikeTimeArray(np.loadtxt(prefix + "/simplicial/spikes/" + str(i) + ".csv", delimiter=','), timesteps)
			for j in range(0, int(timesteps/blockLen)):
				rand.append(tRand[(j*blockLen):((j+1)*blockLen)])
				simp.append(tSimp[(j*blockLen):((j+1)*blockLen)])
		while len(rand) > 0 and len(simp) > 0:
			if np.random.rand() < .5:
				self.data.append(rand.pop())
				self.labels.append([0,1])
			else:
				self.data.append(simp.pop())
				self.labels.append([1,0])
		while len(rand) > 0:
			self.data.append(rand.pop())
			self.labels.append([0,1])
		while len(simp) > 0:
			self.data.append(simp.pop())
			self.labels.append([1,0])
		self.batchId = 0

	def crop(self, cropLen):
		print("Not implemented lol")
#		for i in range(len(self.data)):
#			self.data[i] = self.data[i][:cropLen]

	def next(self, batchSize):
		if self.batchId == len(self.data):
			self.batchId = 0
		batchData = self.data[self.batchId:min(self.batchId + batchSize, len(self.data))]
		batchLabels = self.labels[self.batchId:min(self.batchId + batchSize, len(self.data))]
		self.batchId = min(self.batchId + batchSize, len(self.data))
		return batchData, batchLabels
			
training = seqData(0, 749, blockLen=timesteps)
testing = seqData(750, 999, blockLen=timesteps)

_data = tf.placeholder("float", [None, timesteps, 1])
_labels = tf.placeholder("float", [None, numClasses])


weights = { 'out': tf.Variable(tf.random_normal([numHidden, numClasses])) }
biases = { 'out': tf.Variable(tf.random_normal([numClasses])) }

def RNN(x, weights, biases):
	x = tf.unstack(x, timesteps, 1)
	#lstm_cell = rnn.BasicLSTMCell(numHidden, forget_bias=1.0)
	#dropout = tf.layers.dropout(inputs=lstm_cell, rate=0.4)
	#lstm2 = rnn.BasicLSTMCell(inputs=dropout, forget_bias=1.0)
	lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(numHidden, forget_bias=1.0) for _ in range(numLayers)])
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
		pretty.arrow(step%epochLen, epochLen)
		if step % 500 == 0:
			save = saver.save(sess, "./trained" + str(step) + ".ckpt")
			print("Saved checkpoint " + str(step))
	save = saver.save(sess, "./trained.ckpt")
	print("Training complete; model save in file %s" % save)
#	testing = seqData(750, 999)
	testData = testing.data
	testLabels = testing.labels
	print("Accuracy on validation data:", sess.run(accuracy, feed_dict={_data: testData, _labels: testLabels}))