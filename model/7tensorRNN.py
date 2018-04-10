import numpy as np
import signal, sys, math, shutil, os, functools
import tensorflow as tf
from scripts.SeqData import seqData

def lazy_property(function):
	attribute = '_cache_' + function.__name__

	@property
	@functools.wraps(function_)
	def decorator(self):
		if not hasattr(self, attribute):
			setattr(self, attribute, function(self))
		return getattr(self, attribute)

	return decorator

class Model():
	def __init__(self, b, d, n, structure=[1,1], learnRate=.0025, matDir=None):
		# Reference:
		# b: timesteps of input slices
		# d: metalayer size
		# n: num neurons
		# lr: learning rate
		# structure[0]: num of layer 1s (can this be >1?)
		# structure[1]: num of layer 2s (not implemented >1)
		# TODO: use structure at all

		self.b = b
		seld.d = d
		self.n = n
		self.lr = learnRate
		self.data = tf.placeholder(tf.float32, [None, b, n])
		self.labels = tf.placeholder(tf.float32, [None, 1, n*n])
		self.weights = dict()
		self.biases = dict()
		self.prediction
		self.optimize

		self.initTiles()
		
		if matDir is not None:
			# attempt to load matrices from previous run
			if os.path.exists(matDir):
				loadMats(matDir)
		else:
			initMats()

	def initTiles(self):
		expand = np.array([[1]*n + [0]*n*(n-1)])
		for i in range(1, n):
		    expand = np.append(expand, [[0]*n*i + [1]*n + [0]*n*(n-1-i)], 0)
		self.expand = tf.constant(expand, tf.float32)
		
		tile = np.array([([1] + [0]*(n-1))*n])
		for i in range(1, n):
		    tile = np.append(tile, [([0]*i + [1] + [0]*(n-1-i))*n], 0)
		self.tile = tf.constant(tile, tf.float32);

	def loadMats(self, matDir):
		print("Loading matrices from", matDir)
		print("Incomplete method!")
		# TODO: write this, and refactor weights 
		# 		make sure to include freezing top layers and unfreezing bottom 1
		sys.exit(-1)

	def initMats(self):
		self.weights['layer0'] = tf.Variable(tf.random_normal([self.d, 2*self.b]))
		self.weights['layer1'] = [tf.Variable(tf.random_normal([self.d, 2*self.d])),
			tf.Variable(tf.random_normal([self.d, 2*self.d]))]
		self.weights['final'] = tf.Variable(tf.random_normal([1, self.d]))

		self.biases['layer0'] = tf.Variable(tf.constant(.1, shape=[self.d,1]))
		self.biases['layer1'] = tf.Variable(tf.constant(.1, shape=[self.d,1]))
	
	@lazy_property
	def layer0(self, x):
		# print(<thing>.get_shape().as_list()
	    total = tf.concat([tf.einsum('ijk,kl->ijl',x,self.expand), 
			tf.tile(x,[1,1,self.n])], 1)
		# TODO: biases
		return tf.nn.relu(tf.add(tf.einsum('ij,kjl->kil', self.weights['layer0'], total),
				tf.tile(self.biases['layer0'], [1,1,self.n*self.n)))

	@lazy_property
	def layer1(self, x):
	    a_total = tf.concat([tf.einsum('ijk,kl->ijl', 
			tf.einsum('ijk,kl->ijl', x, tf.transpose(expand)), expand), x], 1)
	    a_total = tf.einsum('ij,ljk->lik', weights['layer2_out'], a_total)
	    b_total = tf.concat([x, tf.einsum('ijk,kl->ijl', 
			tf.einsum('ijk,kl->ijl', x, tf.transpose(tile)), tile)], 1)
	    b_total = tf.einsum('ij,ljk->lik', weights['layer2_in'], b_total)
		#TODO: biases
	    return tf.nn.relu(tf.add(a_total, b_total))
	
	@lazy_property
	def layerFinal(self, x):
    	return tf.einsum('ij,ljk->lik', weights['final'], x)

	@lazy_property
	def prediction(self):
		return tf.nn.tanh(self.layerFinal(self.layer1(self.layer0(self.data))))

	@lazy_property
	def loss(self):
    	return tf.reduce_mean(tf.losses.mean_squared_error(self.labels, 
			self.prediction, reduction=tf.losses.Reduction.NONE))

	@lazy_property
	def optimize(self):
		optimizer = tf.train.AdamOptimizer(self.lr)
		#return optimizer.minimize(self.loss, global_step=global_step)
		return optimizer.minimize(self.loss)

#correct = tf.equal(tf.argmax(pred, 1), tf.argmax(_labels, 1))
