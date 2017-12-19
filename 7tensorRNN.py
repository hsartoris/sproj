import numpy as np
import scripts.GraphKit as gk
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import prettify
import math
from SeqData import seqData

runNumber = 6
batchSize = 64
numClasses = 2
numInput = 100
timesteps = 1000
numHidden = 128
baseRate = .0001
initLearningRate = .05
#initLearningRate = 0.01 - baseRate
trainingSteps = 10000
epochLen = 100
prefix = "classifiertest2"
pretty = prettify.pretty()
logPath = "/home/hsartoris/tflowlogs/"
numLayers = 2



b = 200
d = 25
n = numInput

_data = tf.placeholder(tf.float32, [None, b, numInput])
_labels = tf.placeholder(tf.float32, [None, 1, numInput * numInput])
dropout = tf.placeholder(tf.float32)

#weights = { 'layer0': tf.Variable(tf.random_normal([d, 2*b])), 'final' : tf.Variable(tf.random_normal([1,d])) }
weights = [tf.Variable(tf.random_normal([2*b, d])), tf.Variable(tf.random_normal([d, d])),  tf.Variable(tf.random_normal([d, 1]))]
#biases = [tf.Variable(tf.random_normal([d])), tf.Variable(tf.random_normal([1]))]
biases = { 'layer0' : tf.Variable(tf.random_normal([d])), 'final' : tf.Variable(tf.random_normal([1])) }

expand = np.array([[1]*n + [0]*n*(n-1)])
for i in range(1, n):
	expand = np.append(expand, [[0]*n*i + [1]*n + [0]*n*(n-1-i)], 0)

expand = tf.constant(expand, tf.float32)

def batchModel(x, weights, biases):
	# this is a party
#	upper1 = tf.einsum('ijk,kl->ijl', x, expand) # this may well be the most simple part of this
#	print(upper1.get_shape().as_list())
#	lower1 = tf.tile(x, [1,1,n]) 				# uhhh
#	print(lower1.get_shape().as_list())
	total = tf.concat([tf.einsum('ijk,kl->ijl',x,expand), tf.tile(x,[1,1,n])], 1)		# 0 axis is now batches??
	print(total.get_shape().as_list())
	layer1 = tf.nn.relu(tf.einsum('ij,kjl->kil', weights['layer0'], total))
	print(layer1.get_shape().as_list())
#	layer1 = [tf.nn.relu(tf.matmul(tf.expand_dims(tf.concat([x[int(i/n)], x[i%n]], 0), 0), weights['layer0']) + biases['layer0']) for i in range(n*n)]
	print("Compiled first layer")
	out = tf.einsum('ij,kjl->kil', weights['final'], layer1)
	print(out.get_shape().as_list())
	return out

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

logits = batchModel(_data, weights, biases)
with tf.name_scope("Model"):
	pred = tf.nn.softmax(logits)

with tf.name_scope("Loss"):
#	lossOp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=_labels))
	#lossOp = tf.reduce_sum(tf.losses.absolute_difference(_labels, pred, reduction=tf.losses.Reduction.SUM))
	lossOp = tf.reduce_sum(tf.losses.absolute_difference(_labels, pred, reduction=None))

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

training = seqData(0, 639)
validation = seqData(640, 799)
testing = seqData(800, 999)
training.crop(numInput)
validation.crop(numInput)
testing.crop(numInput)
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
			validX, validY = validation.next(batchSize*2)
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
	testData = testing.data
	testLabels = testing.labels
	print("Immediate OOM:", sess.run(accuracy, feed_dict={_data: testData, _labels: testLabels}))
