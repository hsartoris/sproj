import numpy as np
import scripts.GraphKit as gk
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import prettify
import math

runNumber = 6
batchSize = 64
numClasses = 2
numInput = 3
timesteps = 20
baseRate = .0001
initLearningRate = .05
#initLearningRate = 0.01 - baseRate
trainingSteps = 10000
epochLen = 100
prefix = "classifiertest2"
pretty = prettify.pretty()
logPath = "/home/hsartoris/tflowlogs/"
numLayers = 2



b = 20
d = 4
n = 3

_data = tf.placeholder(tf.float32, [None, b, numInput])
_labels = tf.placeholder(tf.float32, [None, d, numInput * numInput])
dropout = tf.placeholder(tf.float32)

weights = { 'layer0': tf.Variable(tf.random_normal([d, 2*b])), 'layer1': tf.Variable(tf.random_normal([d,d])), 'final' : tf.Variable(tf.random_normal([1,d])) }
#weights = [tf.Variable(tf.random_normal([2*b, d])), tf.Variable(tf.random_normal([d, d])),  tf.Variable(tf.random_normal([d, 1]))]
#biases = [tf.Variable(tf.random_normal([d])), tf.Variable(tf.random_normal([1]))]
biases = { 'layer0' : tf.Variable(tf.random_normal([d])), 'final' : tf.Variable(tf.random_normal([1])) } # not currently in use

expand = np.array([[1]*n + [0]*n*(n-1)])
for i in range(1, n):
	expand = np.append(expand, [[0]*n*i + [1]*n + [0]*n*(n-1-i)], 0)

expand = tf.constant(expand, tf.float32)

def batchModel(x, weights, biases):
	# this is a party
	total = tf.concat([tf.einsum('ijk,kl->ijl',x,expand), tf.tile(x,[1,1,n])], 1)		# 0 axis is now batches??
	layer1 = tf.nn.relu(tf.einsum('ij,kjl->kil', weights['layer0'], total))
	layer2 = tf.nn.relu(tf.einsum('ij,kjl->kil', weights['layer1'], layer1))
	print("Compiled first layer set")
#	out = tf.einsum('ij,kjl->kil', weights['final'], layer1)
	print(out.get_shape().as_list())
	return layer2

def model(x, weights, biases):
	upper1 = tf.matmul(x, expand)
	lower1 = tf.tile(x, [1,n])
	total = tf.concat([upper1, lower1], 0)
	layer1 = tf.nn.relu(tf.matmul(weights['layer0'], total))
#	layer1 = [tf.nn.relu(tf.matmul(tf.expand_dims(tf.concat([x[int(i/n)], x[i%n]], 0), 0), weights['layer0']) + biases['layer0']) for i in range(n*n)]
	print("Compiled first layer")
#	out = [(tf.matmul(layer1[i], weights['final']) + biases['final']) for i in range(len(layer1))]
	out = tf.matmul(weights['final'], layer1)
	return out	

#X, Y = iterator.get_next()
global_step = tf.Variable(0, trainable=False)

with tf.name_scope("rate"):
	#learningRate = tf.train.inverse_time_decay(initLearningRate, global_step, 1, .003)
	#learningRate = tf.train.exponential_decay(initLearningRate, global_step, 500, .4)
	#learningRate = tf.train.polynomial_decay(0.001, global_step, trainingSteps, .0001)
	learningRate = tf.placeholder(tf.float32, shape=[])

logits = batchModel(_data, weights, biases)
with tf.name_scope("Model"):
	pred = tf.nn.softmax(logits)

with tf.name_scope("Loss"):
#	lossOp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=_labels))
	#lossOp = tf.reduce_sum(tf.losses.absolute_difference(_labels, pred, reduction=tf.losses.Reduction.SUM))
	lossOp = tf.reduce_sum(tf.losses.absolute_difference(_labels, pred, reduction=tf.losses.Reduction.NONE))

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
class SeqData(object):
	def __init__(self, num, train=.6, validate=.2, test=.2):
		self.data = []
		self.labels = []
		for i in range(num):
			self.data.append(np.loadtxt(str(i) + "/data.csv", delimiter=','))
			self.labels.append(np.loadtxt(str(i) + "/label.csv", delimiter=','))
		p = np.random.permutation(len(self.data))
		self.data = self.data[p]
		self.labels = self.labels[p]
		self.maxTrain = len(self.data) * train
		self.maxValid = len(self.data) * (train + validate)
		self.maxTest = len(self.data)
		self.batchId = 0
	def next(self, batchSize):
		if self.batchId == len(self.data): self.batchId = 0
		n = self.batchId + batchSize
		batchData = self.data[self.batchId:min(n, maxTrain)]
		batchLabels = self.labels[self.batchId:min(n, maxTrain)]
		self.batchId = min(n, maxTrain)
		return batchData, batchLabels
	def validation(self):
		return self.data[self.maxTrain:self.maxValid], self.labels[self.maxTrain:self.maxValid]
	def testing(self):
		return self.data[self.maxValid:len(self.data)], self.labels[self.maxTrain:len(self.data)]

data = SeqData(0, 639)
data.validation()
data.testing()
start = 1

with tf.Session() as sess:
	sess.run(init)
	summWriter = tf.summary.FileWriter(logPath + "/train" + str(runNumber), graph=tf.get_default_graph())
	validWriter = tf.summary.FileWriter(logPath + "/validation" + str(runNumber))
	#saver.restore(sess, "trained500.ckpt")
	if len(sys.argv) > 1:
		saver.restore(sess, sys.argv[1])
		testData, testLabels = data.testing()
		print("Accuracy on validation data:", sess.run(accuracy, feed_dict={_data: testData, _labels: testLabels}))
		sys.exit()
#		global_step += start
	for step in range(start, trainingSteps+1):
		global_step += 1
		#print(weights['final'].eval())
		#if step < 1500: lr = baseRate + (initLearningRate * math.pow(.4, step/500.0))
		#else: lr = baseRate + (initLearningRate / (1 + .00975 * step))
		lr = initLearningRate
		batchX, batchY = training.next(batchSize)
	#	batchY = batchY.transpose()
		sess.run(trainOp, feed_dict={_data: batchX,_labels: batchY, learningRate: lr})

		if step % epochLen == 0 or step == 1:
			currRate, tLoss, tAcc, loss= sess.run([rateSum, lossSum, accSum, lossOp], feed_dict={_data: batchX, _labels: batchY, learningRate: lr})
			summWriter.add_summary(currRate, step)
			summWriter.add_summary(tLoss, step)
			summWriter.add_summary(tAcc, step)
#			validX, validY = validation.data, validation.labels
			validX, validY = data.validation()
			vloss, vacc, loss, acc= sess.run([lossSum, accSum, lossOp, accuracy], feed_dict={_data: validX, _labels: validY, learningRate: lr})
			validWriter.add_summary(vloss, step)
			validWriter.add_summary(vacc, step)
			#loss, acc = sess.run([lossOp], feed_dict={_data: batchX, _labels: batchY, learningRate: lr})
			print("Step " + str(step) + ", batch loss = " + "{:.4f}".format(loss) + ", accuracy = " + "{:.3f}".format(acc))
			print(weights['final'].eval())
		pretty.arrow(step%epochLen, epochLen)

		if step % 500 == 0:
			save = saver.save(sess, "/home/hsartoris/tflowlogs/checkpoints" + str(runNumber) + "/trained" + str(step) + ".ckpt")
			print("Saved checkpoint " + str(step))
	save = saver.save(sess, "/home/hsartoris/tflowlogs/checkpoints" + str(runNumber) + "/trained.ckpt")
	print("Training complete; model save in file %s" % save)
	testData, testLabels = data.testing()
	print("Immediate OOM:", sess.run(accuracy, feed_dict={_data: testData, _labels: testLabels}))
