import numpy as np
import scripts.GraphKit as gk
import sys
import prettify
import math

timesteps = 1000
prefix = "classifiertest2"
pretty = prettify.pretty()

class seqData2(object):
    def __init__(self, minIdx, maxIdx, dataDir):
        self.data = []
        self.labels = []
        # this is fucking stupid
        label = np.expand_dims(np.loadtxt(dataDir + "/struct.csv", delimiter=',').flatten(), axis=0)
        for i in range(minIdx, maxIdx):
            self.data.append(np.loadtxt(dataDir + "/spikes/" + str(i) + ".csv", delimiter=',').transpose())
            self.labels.append(label)
        pretty.arrow(i, maxIdx - minIdx)
    
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
        # returns data and label arrays, along with current batchId
        if self.batchId == len(self.data):
            self.batchId = 0
        batchData = self.data[self.batchId:min(self.batchId + batchSize, len(self.data))]
        batchLabels = self.labels[self.batchId:min(self.batchId + batchSize, len(self.data))]
        self.batchId = min(self.batchId + batchSize, len(self.data))
        return batchData, batchLabels, self.batchId

    def epochProgress(self):
        return self.batchId, len(self.data)
