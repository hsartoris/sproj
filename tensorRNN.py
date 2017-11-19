import numpy as np
import scripts.GraphKit as gk
import sys
import tensorflow as tf
from tensorflow.contrib import rnn

batchSize = 64
numClasses = 2
numInput = 1000
timesteps = 1000
numHidden = 128
learningRate = .001
trainingSteps = 10000

sess = tf.Session()

def parser(x, y):
	return x, tf.one_hot(y, numClasses)

def loadData(prefix, num):
	x_train_rand = np.array([gk.spikeTimeMatrix(np.loadtxt(prefix + "/simplicial/spikes/0.csv", delimiter=','), 1000, 1000).flatten()])
	x_train_simp = np.array([gk.spikeTimeMatrix(np.loadtxt(prefix + "/random/spikes/0.csv", delimiter=','), 1000, 1000).flatten()])
	for i in range(1, num):
		print(i)
		x_train_simp = np.append(x_train_simp, [gk.spikeTimeMatrix(np.loadtxt(prefix + "/simplicial/spikes/" + str(i) + ".csv", delimiter=','), 1000, 1000).flatten()], axis=0)
		x_train_rand = np.append(x_train_rand, [gk.spikeTimeMatrix(np.loadtxt(prefix + "/random/spikes/" + str(i) + ".csv", delimiter=','), 1000, 1000).flatten()], axis=0)
		#x_train_simp.append(tf.constant(gk.spikeTimeMatrix(np.loadtxt(prefix + "/simplicial/spikes/" + str(i) + ".csv", delimiter=','), 1000, 1000))
		#x_train_rand.append(tf.constant(gk.spikeTimeMatrix(np.loadtxt(prefix + "/random/spikes/" + str(i) + ".csv", delimiter=','), 1000, 1000)))
	
	x_train = np.append(x_train_simp, x_train_rand, axis=0)
	y_train = tf.constant(np.append(np.repeat([[1,0]], num, axis=0), np.repeat([[0,1]], num, axis=0), axis=0))
	#y_train = ([1]*num) + ([0]*num)
	return x_train, y_train

def getBatch(batchSize):
	# this is so fucking dumb it's unreal
	# yes, this functionality is done through tensorflow as well but it doesn't work
	print("fuck you")

x,y = loadData("classifiertest", 128)
print(x[0])
dataset = tf.data.Dataset.from_tensor_slices((x,y))
#dataset = dataset.map(parser)
dataset = dataset.batch(batchSize)

iterator = dataset.make_initializable_iterator()

_data = tf.placeholder(tf.float32, [None, timesteps, numInput])
_labels = tf.placeholder(tf.float32, [None, numClasses])

#sess.run(iterator.initializer, feed_dict={_data:x, _labels:y})

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

with tf.Session() as sess:
	sess.run(init)
	sess.run(iterator.initializer)
	for step in range(1, trainingSteps+1):
		try:
			batchX, batchY = iterator.get_next()
			batchX = batchX.eval().reshape((batchSize, timesteps, numInput))
			sess.run(trainOp, feed_dict={_data: batchX,_labels: batchY.eval()})
#			sess.run(trainOp)
			if step % 200 == 0 or step == 1:
#				loss, acc = sesss.run([lossOp, accuracy])
				loss, acc = sess.run([lossOp, accuracy], feed_dict={_data: batchX, _labels: batchY.eval()})
#				print("Step " + str(step) + ", batch loss = " + "{:.4f}".format(loss) + ", accuracy = " + "{:.3f}".format(acc))
		except tf.errors.OutOfRangeError:
			sess.run(iterator.initializer, feed_dict={_data: x.reshape((256, 1000, 1000)), _labels: y})
			sess.run(trainOp)
