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
numInput = 3 # number of neurons
timesteps = 200
numHidden = 128
baseRate = .0001
initLearningRate = .05
#initLearningRate = 0.01 - baseRate
trainingSteps = 10000
epochLen = 100
prefix = "dataSmall"
pretty = prettify.pretty()
logPath = "/home/hsartoris/tflowlogs/"
numLayers = 2



b = timesteps
d = 9
n = numInput

_data = tf.placeholder(tf.float32, [None, b, numInput])
#_data = tf.placeholder(tf.float32, [b, numInput])
_labels = tf.placeholder(tf.float32, [None, 1, numInput * numInput])
#_labels = tf.placeholder(tf.float32, [1, numInput * numInput])
dropout = tf.placeholder(tf.float32)

weights = { 'layer0': tf.Variable(tf.random_normal([d, 2*b])), 'layer2_in': tf.Variable(tf.random_normal([d, 2*d])), 'layer2_out': tf.Variable(tf.random_normal([d, 2*d])), 'final' : tf.Variable(tf.random_normal([1,d])) }
#weights = [tf.Variable(tf.random_normal([2*b, d])), tf.Variable(tf.random_normal([d, d])),  tf.Variable(tf.random_normal([d, 1]))]
#biases = [tf.Variable(tf.random_normal([d])), tf.Variable(tf.random_normal([1]))]
biases = { 'layer0' : tf.Variable(tf.random_normal([d])), 'final' : tf.Variable(tf.random_normal([1])) }

expand = np.array([[1]*n + [0]*n*(n-1)])
for i in range(1, n):
	expand = np.append(expand, [[0]*n*i + [1]*n + [0]*n*(n-1-i)], 0)

expand = tf.constant(expand, tf.float32)

tile = np.array([([1] + [0]*(n-1))*n])
for i in range(1, n):
	tile = np.append(tile, [([0]*i + [1] + [0]*(n-1-i))*n], 0)

tile = tf.constant(tile, tf.float32);

def batchModel(x, weights):
	# effectively layer 0
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
#	out = tf.einsum('ij,kjl->kil', weights['final'], layer1)
	out = layer1
	print("first layer output size:",out.get_shape().as_list())
	return out

def layer2batch(x, weights):
	a_total = tf.concat([tf.einsum('ijk,kl->ijl', tf.einsum('ijk,kl->ijl', x, tf.transpose(expand)), expand), x], 1)
	print("A mat size", a_total.get_shape().as_list())
	a_total = tf.einsum('ij,ljk->lik', weights['layer2_out'], a_total)
	b_total = tf.concat([x, tf.einsum('ijk,kl->ijl', tf.einsum('ijk,kl->ijl', x, tf.transpose(tile)), tile)], 1)
	print("B mat size", b_total.get_shape().as_list())
	b_total = tf.einsum('ij,ljk->lik', weights['layer2_in'], b_total)
	return tf.nn.relu(tf.add(a_total, b_total))
	

def layer2(x, weights):
	# non-batch model
	a = tf.matmul(tf.matmul(x, tf.transpose(expand)), expand)
	b = tf.matmul(tf.matmul(x, tf.transpose(tile)), tile)
	a_total = tf.concat([a, x], 0)
	b_total = tf.concat([x, b], 0)
	total = tf.nn.relu(tf.add(tf.matmul(weights['layer2_out'], a_total), tf.matmul(weights['layer2_in'], b_total)))
	return total;

def finalBatch(x, weights):
	return tf.einsum('ij,ljk->lik', weights['final'], x)
	

def model(x, weights):
	upper1 = tf.matmul(x, expand)
	lower1 = tf.tile(x, [1,n])
	total = tf.concat([upper1, lower1], 0)
	layer1 = tf.nn.relu(tf.matmul(weights['layer0'], total))
#	layer1 = [tf.nn.relu(tf.matmul(tf.expand_dims(tf.concat([x[int(i/n)], x[i%n]], 0), 0), weights['layer0']) + biases['layer0']) for i in range(n*n)]
	print("Compiled first layer")
#	out = [(tf.matmul(layer1[i], weights['final']) + biases['final']) for i in range(len(layer1))]
	return layer1

def final(x, weights):
	# non-batch model final layer
	return tf.matmul(weights['final'], x)

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

logits = finalBatch(layer2batch(batchModel(_data, weights), weights), weights)
#logits = final(layer2(model(_data, weights), weights), weights);
with tf.name_scope("Model"):
#	pred = tf.nn.softmax(logits)
	pred = tf.nn.tanh(logits)

with tf.name_scope("Loss"):
	lossOp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=_labels))
	lossOp = tf.reduce_mean(tf.losses.mean_squared_error(_labels, pred, reduction=tf.losses.Reduction.NONE))
	#lossOp = tf.reduce_sum(tf.losses.absolute_difference(_labels, pred, reduction=tf.losses.Reduction.SUM))
#	lossOp = tf.reduce_sum(tf.losses.absolute_difference(_labels, pred, reduction=tf.losses.Reduction.NONE))

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

training = seqData2(0, 1280)
validation = seqData(1280, 1600)
testing = seqData(1600, 2000)
#training.crop(numInput)
#validation.crop(numInput)
#testing.crop(numInput)
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
		batchX, batchY = training.next(1)
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
