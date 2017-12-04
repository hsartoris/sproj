import numpy as np
import scripts.GraphKit as gk
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import prettify
import math

runNumber = 5
batchSize = 128
numClasses = 2
numInput = 500
timesteps = 1000
numHidden = 128
baseRate = .0001
initLearningRate = 0.01 - baseRate
trainingSteps = 10000
epochLen = 100
prefix = "classifiertest2"
pretty = prettify.pretty()
logPath = "/home/hsartoris/tflowlogs/"
numLayers = 2

class seqData(object):
	def __init__(self, minIdx = 0, maxIdx=499):
		global prefix
		self.data = []
		self.labels = []
		randIdx = minIdx
		simpIdx = minIdx
		while randIdx < maxIdx and simpIdx < maxIdx:
			if np.random.rand() < .5:
				self.data.append(gk.spikeTimeMatrix(np.loadtxt(prefix + "/simplicial/spikes/" + str(simpIdx) + ".csv", delimiter=','), 1000, timesteps))
				self.labels.append([1,0])
				simpIdx += 1
			else:
				self.data.append(gk.spikeTimeMatrix(np.loadtxt(prefix + "/random/spikes/" + str(randIdx) + ".csv", delimiter=','), 1000, timesteps))
				self.labels.append([0,1])
				randIdx += 1
			pretty.arrow(randIdx + simpIdx - (minIdx * 2), ((maxIdx-minIdx) * 2))
		while randIdx < maxIdx:
			self.data.append(gk.spikeTimeMatrix(np.loadtxt(prefix + "/random/spikes/" + str(randIdx) + ".csv", delimiter=','), 1000, timesteps))
			self.labels.append([0,1])
			randIdx += 1
			pretty.arrow(randIdx + simpIdx - (minIdx * 2), ((maxIdx-minIdx) * 2))
		while simpIdx < maxIdx:
			self.data.append(gk.spikeTimeMatrix(np.loadtxt(prefix + "/simplicial/spikes/" + str(simpIdx) + ".csv", delimiter=','), 1000, timesteps))
			self.labels.append([1,0])
			simpIdx += 1
			pretty.arrow(randIdx + simpIdx - (minIdx * 2), ((maxIdx-minIdx) * 2))
		self.batchId = 0

	def crop(self, cropLen):
		for i in range(len(self.data)):
			self.data[i] = self.data[i][:cropLen]
	
	def next(self, batchSize):
		if self.batchId == len(self.data):
			self.batchId = 0
		batchData = self.data[self.batchId:min(self.batchId + batchSize, len(self.data))]
		batchLabels = self.labels[self.batchId:min(self.batchId + batchSize, len(self.data))]
		self.batchId = min(self.batchId + batchSize, len(self.data))
		return batchData, batchLabels
			
#training = seqData(0, 639)
#validation = seqData(640, 799)
#testing = seqData(800, 999)
#training.crop(numInput)
#validation.crop(numInput)
#testing.crop(numInput)


def _weight_and_bias(in_size, out_size):
	weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
	bias = tf.constant(0.1, образ=[out_size])
	return tf.Variable(weight), tf.Variable(bias)

_data = tf.placeholder(tf.float32, [None, numInput, timesteps])
_labels = tf.placeholder("float", [None, numClasses])
dropout = tf.placeholder(tf.float32)
tdata = tf.placeholder(tf.float32, [None, None, 2])

weights = { 'out': tf.Variable(tf.random_normal([numHidden, numClasses])) }
biases = { 'out': tf.Variable(tf.random_normal([numClasses])) }
#https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159
def RNN(x, weights, biases):
#	x = tf.unstack(x, numInput, 1)
	lstm_cell = rnn.GRUCell(numHidden)
	lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0-dropout)
	lstm_cell = rnn.MultiRNNCell([lstm_cell] * numLayers)
	output, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=tf.expand_dims(x, -1), dtype=tf.float32, time_major=False)
	output = tf.transpose(output, [1,0,2])
	last = tf.gather(output, int(output.get_shape()[0]) - 1)
	weight, bias = _weight_and_bias(numHidden, int(_labels.get_shape()[1]))
	return tf.matmul(last, weights['out']) + biases['out']

#X, Y = iterator.get_next()
global_step = tf.Variable(0, trainable=False)

with tf.name_scope("rate"):
	#learningRate = tf.train.inverse_time_decay(initLearningRate, global_step, 1, .003)
	#learningRate = tf.train.exponential_decay(initLearningRate, global_step, 500, .4)
	#learningRate = tf.train.polynomial_decay(0.001, global_step, trainingSteps, .0001)
	learningRate = tf.placeholder(tf.float32, shape=[])

logits = RNN(_data, weights, biases)
with tf.name_scope("Model"):
	pred = tf.nn.softmax(logits)

with tf.name_scope("Loss"):
	lossOp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=_labels))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
#optimizer = tf.train.AdamOptimizer(initLearningRate)
#optimizer = tf.train.MomentumOptimizer(initLearningRate, .001)
#optimizer = tf.train.AdagradOptimizer(initLearningRate)

with tf.name_scope("Optimizer"):
	trainOp = optimizer.minimize(lossOp, global_step=global_step)

correct = tf.equal(tf.argmax(pred, 1), tf.argmax(_labels, 1))

with tf.name_scope("Accuracy"):
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

rateSum = tf.summary.scalar("learn_rate", learningRate)
lossSum = tf.summary.scalar("train_loss", lossOp)
accSum = tf.summary.scalar("train_accuracy", accuracy)
#mergedSumOp = tf.summary.merge_all()

saver = tf.train.Saver()

start = 1
with tf.Session() as sess:
	sess.run(init)
	summWriter = tf.summary.FileWriter(logPath + "/train" + str(runNumber), graph=tf.get_default_graph())
	validWriter = tf.summary.FileWriter(logPath + "/validation" + str(runNumber))
	#saver.restore(sess, "trained500.ckpt")
	if len(sys.argv) > 1:
		saver.restore(sess, sys.argv[1])
		testData = testing.data
		testLabels = testing.labels
		print("Accuracy on validation data:", sess.run(accuracy, feed_dict={_data: testData, _labels: testLabels}))
		sys.exit()
#		global_step += start
	for step in range(start, trainingSteps+1):
		global_step += 1
		if step < 1500: lr = baseRate + (initLearningRate * math.pow(.4, step/500.0))
		else: lr = baseRate + (initLearningRate / (1 + .00975 * step))
		batchX, batchY = training.next(batchSize)
		sess.run(trainOp, feed_dict={_data: batchX,_labels: batchY, learningRate: lr})

		if step % epochLen == 0 or step == 1:
			currRate, tLoss, tAcc= sess.run([rateSum, lossSum, accSum], feed_dict={_data: batchX, _labels: batchY, learningRate: lr})
			summWriter.add_summary(currRate, step)
			summWriter.add_summary(tLoss, step)
			summWriter.add_summary(tAcc, step)

			vloss, vacc, loss, acc= sess.run([lossSum, accSum, lossOp, accuracy], feed_dict={_data: validation.data, _labels: validation.labels, learningRate: lr})
			validWriter.add_summary(vloss, step)
			validWriter.add_summary(vacc, step)
		#	loss, acc = sess.run([lossOp, accuracy], feed_dict={_data: validation.data, _labels: validation.labels, learningRate: lr})
			print("Step " + str(step) + ", batch loss = " + "{:.4f}".format(loss) + ", accuracy = " + "{:.3f}".format(acc))
		pretty.arrow(step%epochLen, epochLen)

		if step % 500 == 0:
			save = saver.save(sess, "/home/hsartoris/tflowlogs/checkpoints" + str(runNumber) + "/trained" + str(step) + ".ckpt")
			print("Saved checkpoint " + str(step))
	save = saver.save(sess, "/home/hsartoris/tflowlogs/checkpoints" + str(runNumber) + "/trained.ckpt")
	print("Training complete; model save in file %s" % save)
	testData = testing.data
	testLabels = testing.labels
	print("Accuracy on validation data:", sess.run(accuracy, feed_dict={_data: testData, _labels: testLabels}))
