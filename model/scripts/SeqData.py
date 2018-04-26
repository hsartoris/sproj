import numpy as np
import sys
from . import Prettify
import math

timesteps = 1000
prefix = "classifiertest2"
pretty = Prettify.pretty()

class seqData(object):
    def __init__(self, minIdx, maxIdx, dataDir, steps):
        self.data = []
        self.labels = []
        # this is fucking stupid
        label = np.expand_dims(np.loadtxt(dataDir + "/struct", delimiter=',').flatten(), 
            axis=0)
        for i in range(minIdx, maxIdx):
            self.data.append(np.loadtxt(dataDir + "/spikes/" + str(i) + ".csv", 
                delimiter=',').transpose()[:steps])
            self.labels.append(label)
            pretty.arrow(i - minIdx, maxIdx - minIdx) 
        self.batchId = 0
        print("Successfully loaded", maxIdx-minIdx, "samples")

    def resetCount(self):
        self.batchId = 0

    def crop(self, batchSize):
        cropLen = len(self.data) % batchSize
        self.data = self.data[:len(self.data) - cropLen]
        print("Cropped", cropLen, "samples to fit batch size")
        return cropLen
        
    def nextSample(self):
        self.batchId += 1
        if self.batchId == len(self.data):
            self.batchId = 0
        return self.data[self.batchId], self.labels[self.batchId]
	
    def next(self, batchSize, verbose=False):
        # returns data and label arrays, along with current batchId
        if self.batchId == len(self.data):
            self.batchId = 0
        batchData = self.data[self.batchId:min(self.batchId + batchSize, len(self.data))]
        batchLabels = self.labels[self.batchId:min(self.batchId + batchSize, len(self.data))]
        self.batchId = min(self.batchId + batchSize, len(self.data))
        if verbose: print("Returning batch of size", len(batchData))
        return batchData, batchLabels, self.batchId

    def epochProgress(self):
        return self.batchId, len(self.data)
