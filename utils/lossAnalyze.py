#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import os
minIdx = 1
maxIdx = 99
saveDir = "model/checkpoints/comp/3sproj/"

def loadLosses():
    convLosses = []
    dirs = os.listdir()
    for d in dirs:
        convLoss = np.loadtxt(saveDir + d + "/conv/vlosses", delimiter=',')
        if not min(convLoss[:,1]) == 1.0:
            convLosses.append(convLoss)
    return convLosses

def minLoss(losses):
    temp = min(enumerate(losses), key = lambda i: np.sum(i[1][:,1]))
    return temp[0], temp[1]

def minFinalLoss(losses):
    temp = min(enumerate(losses), key = lambda i: i[1][-1,1])
    return temp[0], temp[1]
    
def maxLoss(losses):
    temp =  max(enumerate(losses), key = lambda i: np.sum(i[1][:,1]))
    return temp[0], temp[1]

def avgLoss(losses):
    lossSum = losses[0][:,1]
    for i in range(1, len(losses)):
        lossSum += losses[i][:,1]

    lossSum /= len(losses)
    idxs = np.array([losses[0][:,0], lossSum])
    return losses[0][:,0], lossSum

convLosses = loadLosses()
#convLosses = np.array(sorted(convLosses, key=lambda i: np.sum(i[:,1])))
#simpleLosses = np.array(sorted(simpleLosses, key=lambda i: np.sum(i[:,1])))
#print(convLosses[0])
print(avgLoss(np.array(convLosses)))
'''
plt.subplot(211)
for data in convLosses:
    plt.semilogy(data[:,0], data[:,1])

plt.subplot(212)
for data in simpleLosses:
    plt.semilogy(data[:,0], data[:,1])
plt.show()

'''
#minSimpleIdx, minSimpleLoss = minLoss(simpleLosses)
#minConvIdx, minConvLoss = minLoss(convLosses)

#minSimpleIdx, minSimpleLoss = minFinalLoss(simpleLosses)
minConvIdx, minConvLoss = minFinalLoss(convLosses)
avgConvLossX, avgConvLossY = avgLoss(convLosses)
#avgSimpleLossX, avgSimpleLossY = avgLoss(simpleLosses)

#simpleGraph, = plt.semilogy(minSimpleLoss[:,0], minSimpleLoss[:,1], label="simple " + str(minSimpleIdx))
avgConv, = plt.semilogy(avgConvLossX, avgConvLossY, label="conv_avg")
#avgSimple, = plt.semilogy(avgSimpleLossX, avgSimpleLossY, label="simple_avg")
convGraph, = plt.semilogy(minConvLoss[:,0], minConvLoss[:,1], label="conv" + str(minConvIdx))
#diffGraph, = plt.semilogy(avgConvLossX, avgConvLossY - avgSimpleLossY, label="diff")
plt.legend(handles=[avgConv, convGraph])
plt.show()

#print(minConvLoss)
#print(avgConvLoss)
#print(maxConvLoss)
