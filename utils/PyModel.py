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
        self.relu = np.vectorize(self.reluItem, otypes=[np.float])

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
        print("Layer 0 prerelu max value:", np.max(prerelu))
        return layer

    def layer1(self, data, n):
        layer = dict()

        # labeled as in TFlow model
        # load weights & biases
        layer['weights_in'] = win = self.weights['layer1'][0]
        layer['weights_out'] = wout = self.weights['layer1'][1]
        layer['weights_f'] = wf = self.weights['layer1'][2]
        layer['bias'] = bias = self.biases['layer1']
        # B!: tf.tile(self.biases['layer1'], [self.batchSize,1,self.n*self.n])))
        # of course without batching
        layer['bias_block'] = bias_block = np.tile(bias, n*n)

        #horizCompress
        layer['0hAvg'] = hAvg = np.matmul(data, self.expand.transpose())/n
        #horiz
        layer['1inputs'] = inputs = np.matmul(hAvg, self.tile)
        # in_total
        layer['2in-data'] = in_tot = inputs * data
        # in_total
        layer['3in_proc'] = in_proc = np.matmul(win, in_tot)

        # vertCompress
        layer['0vAvg'] = vAvg = np.matmul(data, self.tile.transpose())/n
        # vert
        layer['1outputs'] = outputs = np.matmul(vAvg, self.expand)
        # out_total
        layer['2out-data'] = out_tot = outputs * data
        # out_total
        layer['3out_proc'] = out_proc = np.matmul(wout, out_tot)

        # output
        layer['4stack'] = stack = np.concatenate((in_proc, out_proc), 0)
        # output
        layer['5prebias'] = prebias = np.matmul(wf, stack)
        # tf.add(output, B!)
        layer['6prerelu'] = prerelu = prebias + bias_block
        # self.activation()
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

    def exportLayerDict(self, layer, layerPath, pn=False, n=None):
        csvPath = layerPath + "csvs/"
        if not os.path.exists(layerPath):
            os.mkdir(layerPath)
            os.mkdir(csvPath)
        elif not os.path.exists(csvPath):
            os.mkdir(csvPath)

        for name, mat in layer.items():
            matVis(mat, layerPath + name + ".png", connections=pn, drawText=pn, 
                    n=n)
            self.savenp(mat, csvPath + name)

    def run(self, data, saveDir, printNumbers=False):
        n = data.shape[1]
        self.makeTiles(n)
        layer0dict = self.layer0(data, n)
        layer1dict = self.layer1(layer0dict['4out'], n)
        layerfdict = self.layerFinal(layer1dict['7out'])
        pred = self.prediction(layerfdict['0out'])
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)

        layerfdict['weights'] = np.expand_dims(layerfdict['weights'], 0)
        layerfdict['0out'] = np.expand_dims(layerfdict['0out'], 0)
        pred = pred.reshape((n,n))

        self.savenp(data, saveDir + "input")
        matVis(data, saveDir + "input.png", drawText=printNumbers)
        self.exportLayerDict(layer0dict, saveDir + "layer0/", pn=printNumbers, 
                n=None)
        self.exportLayerDict(layer1dict, saveDir + "layer1/", pn=printNumbers, 
                n=None)
        self.exportLayerDict(layerfdict, saveDir + "layerf/", pn=printNumbers,
                n=None)
        self.savenp(pred, saveDir + "pred")
        matVis(pred, saveDir + "pred.png", connections=printNumbers, 
                drawText=printNumbers)
        


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: PyModel matDir outDir dataFile [pn]\n\tpn: print numbers")
        sys.exit()
    if len(sys.argv) == 5 and sys.argv[4] == 'pn':
        printNumbers = True
    else:
        printNumbers = False

    if printNumbers: print("Printing numbers")
    matDir = sys.argv[1] + "/"
    outDir = sys.argv[2] + "/"
    data = np.loadtxt(sys.argv[3], delimiter=',')
    m = Model(matDir)
    m.run(data, outDir, printNumbers=printNumbers)
