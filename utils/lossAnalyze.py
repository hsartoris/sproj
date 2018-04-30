#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
minIdx = 1
maxIdx = 99
saveDir = "model/checkpoints/bench/3neur/"

def loadLosses():
    convLosses = []
    dirs = os.listdir(saveDir)
    for d in dirs:
        if d == "15k": continue
        try:
            convLoss = np.loadtxt(saveDir + d + "/conv/vlosses", delimiter=',')
            if not min(convLoss) == 1.0:
                convLosses.append(convLoss)
        except OSError:
            pass
    return convLosses

def minLoss(losses):
    temp = min(enumerate(losses), key = lambda i: np.sum(i[1]))
    return temp[0], temp[1]

def minFinalLoss(losses):
    temp = min(enumerate(losses), key = lambda i: i[1][-1])
    return temp[0], temp[1]
    
def maxLoss(losses):
    temp =  max(enumerate(losses), key = lambda i: np.sum(i[1]))
    return temp[0], temp[1]

def avgLoss(losses):
    lossSum = losses[0]
    for i in range(1, len(losses)):
        lossSum += losses[i]

    lossSum /= len(losses)
    return lossSum

if len(sys.argv) > 1:
    saveDir = sys.argv[1] + "/"

dirs = os.listdir(saveDir)
i = 0
while dirs[i] == "15k": i += 1

steps = np.loadtxt(saveDir + dirs[i] + "/steps", delimiter=',')
plotall = False

convLosses = loadLosses()
#convLosses = np.array(sorted(convLosses, key=lambda i: np.sum(i[:,1])))
#simpleLosses = np.array(sorted(simpleLosses, key=lambda i: np.sum(i[:,1])))
#print(convLosses[0])
#print(avgLoss(np.array(convLosses)))
if plotall:
    #plt.subplot(211)
    for data in convLosses:
        plt.semilogy(steps, data)
    
    #plt.subplot(212)
    #for data in simpleLosses:
    #    plt.semilogy(data[:,0], data[:,1])
    plt.show()

else:
    #minSimpleIdx, minSimpleLoss = minLoss(simpleLosses)
    #minConvIdx, minConvLoss = minLoss(convLosses)
    
    #minSimpleIdx, minSimpleLoss = minFinalLoss(simpleLosses)
    minFinalIdx, minFinalLoss = minFinalLoss(convLosses)
    minTotalIdx, minTotalLoss = minLoss(convLosses)
    minFinalIdx = dirs[minFinalIdx]
    minTotalIdx = dirs[minTotalIdx]
    print("minFinalIdx:", minFinalIdx)
    print("minTotalIdx:", minTotalIdx)
    avgConvLossY = avgLoss(convLosses)
    #avgSimpleLossX, avgSimpleLossY = avgLoss(simpleLosses)
    
    #simpleGraph, = plt.semilogy(minSimpleLoss[:,0], minSimpleLoss[:,1], label="simple " + str(minSimpleIdx))
    avgConv, = plt.semilogy(steps, avgConvLossY, label="avg")
    #avgSimple, = plt.semilogy(avgSimpleLossX, avgSimpleLossY, label="simple_avg")
    minFinGraph, = plt.semilogy(steps, minFinalLoss, label="min final" + 
        minFinalIdx)
    minTotGraph, = plt.semilogy(steps, minTotalLoss, label="min total" + 
        minTotalIdx)
    #diffGraph, = plt.semilogy(avgConvLossX, avgConvLossY - avgSimpleLossY, label="diff")
    plt.legend(handles=[avgConv, minFinGraph, minTotGraph])
    plt.show()
    
    #print(minConvLoss)
    #print(avgConvLoss)
    #print(maxConvLoss)
