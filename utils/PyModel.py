#!/usr/bin/python3
import numpy as np
import sys
import os
from Visualize import matVis

class Model():
    weightStrs = { 'layer0':"l0.weights", 'layer1':["l1in.weights", 
        "l1out.weights", "l1f.weights"], 'final':"final.weights" }
    biasStrs = { 'layer0':"l0.biases", 'layer1':"l1.biases" }
    
    def __init__(self, matDir):
        self.weightsDir = matDir + "/"
        self.weights = dict()
        self.biases = dict()
        self.loadMatrices()
        self.reluItem = lambda v: (0 if v < 0 else v)
        self.relu = np.vectorize(self.reluItem)

    def lWMat(self, fname):
        return np.loadtxt(self.weightsDir + fname, delimiter=',')

    def loadMatrices(self):
        for name, item in self.weightStrs.items():
            print(name + ":", item)
            if type(item) is str:
                self.weights[name] = self.lWMat(item)
            else: # list
                self.weights[name] = [self.lWMat(m) for m in item]
        for name, fname in self.biasStrs.items():
            self.biases[name] = np.expand_dims(self.lWMat(fname), 1)

    def makeTiles(self, n):
        # needs to be called at runtime
        expand = np.array([[1]*n + [0]*n*(n-1)])
        for i in range(1, n):
            expand = np.append(expand, [[0]*n*i + [1]*n + [0]*n*(n-1-i)], 0)
        self.expand = expand
        
        tile = np.array([([1] + [0]*(n-1))*n])
        for i in range(1, n):
            tile = np.append(tile, [([0]*i + [1] + [0]*(n-1-i))*n], 0)
        self.tile = tile
        return expand, tile

    def layer0(self, data, n):
        layer = dict()
        layer['0top'] = top = np.matmul(data, self.expand)
        layer['0bot'] = bot = np.matmul(data, self.tile)
        layer['1stack'] = stack = np.concatenate((top,bot), 0)
        layer['weights'] = weights = self.weights['layer0']
        layer['2prebias'] = prebias = np.matmul(weights, stack)
        layer['bias'] = bias = self.biases['layer0']
        layer['bias_block'] = bias_block = np.tile(bias, n*n)
        layer['3prerelu'] = prerelu = prebias + bias_block
        layer['4out'] = out = self.relu(prerelu)
        return layer

    def layer1(self, data, n):
        layer = dict()
        layer['0hAvg'] = hAvg = np.matmul(data, self.expand.transpose())/n
        layer['1inputs'] = inputs = np.matmul(hAvg, self.tile)
        layer['2in-data'] = in_tot = inputs * data
        layer['weights_in'] = win = self.weights['layer1'][0]
        layer['3in_proc'] = in_proc = np.matmul(win, in_tot)

        layer['0vAvg'] = vAvg = np.matmul(data, self.tile.transpose())
        layer['1outputs'] = outputs = np.matmul(vAvg, self.expand)
        layer['2out-data'] = out_tot = outputs * data
        layer['weights_out'] = wout = self.weights['layer1'][1]
        layer['3out_proc'] = out_proc = np.matmul(wout, out_tot)

        layer['weights_f'] = wf = self.weights['layer1'][2]
        layer['4stack'] = stack = np.concatenate((in_proc, out_proc), 0)
        layer['5prebias'] = prebias = np.matmul(wf, stack)
        layer['bias'] = bias = self.biases['layer1']
        layer['bias_block'] = bias_block = np.tile(bias, n*n)
        layer['6prerelu'] = prerelu = prebias + bias_block
        layer['7out'] = self.relu(prerelu)
        return layer

    def layerFinal(self, data):
        layer = dict()
        layer['weights'] = weights = self.weights['final']
        layer['0out'] = np.matmul(weights, data)
        return layer

    def prediction(self, data):
        return np.tanh(data)

    def savenp(self, mat, path):
        # assumes path exists
        np.savetxt(path, mat, delimiter=',')

    def exportLayerDict(self, layer, layerPath):
        csvPath = layerPath + "csvs/"
        if not os.path.exists(layerPath):
            os.mkdir(layerPath)
            os.mkdir(csvPath)
        elif not os.path.exists(csvPath):
            os.mkdir(csvPath)

        for name, mat in layer.items():
            matVis(mat, layerPath + name + ".png")
            self.savenp(mat, csvPath + name)

    def run(self, data, saveDir):
        saveDir += "/"
        n = data.shape[1]
        self.makeTiles(n)
        layer0dict = self.layer0(data, n)
        layer1dict = self.layer1(layer0dict['4out'], n)
        layerfdict = self.layerFinal(layer1dict['7out'])
        pred = self.prediction(layerfdict['0out'])
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)

        layerfdict['weights'] = np.expand_dims(layerfdict['weights'], 0)
        layerfdict['0out'] = layerfdict['0out'].reshape((n,n))
        pred = pred.reshape((n,n))

        self.savenp(data, saveDir + "input")
        matVis(data, saveDir + "input.png")
        self.exportLayerDict(layer0dict, saveDir + "layer0/")
        self.exportLayerDict(layer1dict, saveDir + "layer1/")
        self.exportLayerDict(layerfdict, saveDir + "layerf/")
        self.savenp(pred, saveDir + "pred")
        matVis(pred, saveDir + "pred.png")
        


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: PyModel matDir outDir dataFile")
        sys.exit()
    matDir = sys.argv[1] + "/"
    outDir = sys.argv[2] + "/"
    data = np.loadtxt(sys.argv[3], delimiter=',')
    m = Model(matDir)
    m.run(data, outDir)
