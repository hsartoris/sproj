import numpy as np
import scripts.GraphKit as gk
import sys
import prettify
import math

timesteps = 1000
prefix = "classifiertest2"
pretty = prettify.pretty()

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
				self.labels.append(gk.loadSparse(prefix + "/simplicial/" + str(simpIdx) + ".npz")[:100,:100].flatten())
				simpIdx += 1
			else:
				self.data.append(gk.spikeTimeMatrix(np.loadtxt(prefix + "/random/spikes/" + str(randIdx) + ".csv", delimiter=','), 1000, timesteps))
				self.labels.append(gk.loadSparse(prefix + "/random/" + str(simpIdx) + ".npz")[:100,:100].flatten())
				randIdx += 1
			pretty.arrow(randIdx + simpIdx - (minIdx * 2), ((maxIdx-minIdx) * 2))
		while randIdx < maxIdx:
			self.data.append(gk.spikeTimeMatrix(np.loadtxt(prefix + "/random/spikes/" + str(randIdx) + ".csv", delimiter=','), 1000, timesteps))
			self.labels.append(gk.loadSparse(prefix + "/random/" + str(simpIdx) + ".npz")[:100,:100].flatten())
			randIdx += 1
			pretty.arrow(randIdx + simpIdx - (minIdx * 2), ((maxIdx-minIdx) * 2))
		while simpIdx < maxIdx:
			self.data.append(gk.spikeTimeMatrix(np.loadtxt(prefix + "/simplicial/spikes/" + str(simpIdx) + ".csv", delimiter=','), 1000, timesteps))
			self.labels.append(gk.loadSparse(prefix + "/simplicial/" + str(simpIdx) + ".npz")[:100,:100].flatten())
			simpIdx += 1
			pretty.arrow(randIdx + simpIdx - (minIdx * 2), ((maxIdx-minIdx) * 2))
		self.batchId = 0

	def crop(self, cropLen):
		for i in range(len(self.data)):
			self.data[i] = self.data[i][:cropLen,:200].transpose()
	def nextSample(self):
		self.batchId += 1
		if self.batchId == len(self.data):
			self.batchId = 0
		return self.data[self.batchId], self.labels[self.batchId]
	
	def next(self, batchSize):
		if self.batchId == len(self.data):
			self.batchId = 0
		batchData = self.data[self.batchId:min(self.batchId + batchSize, len(self.data))]
		batchLabels = self.labels[self.batchId:min(self.batchId + batchSize, len(self.data))]
		self.batchId = min(self.batchId + batchSize, len(self.data))
		return batchData, batchLabels
