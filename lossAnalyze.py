#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
minIdx = 1
maxIdx = 99
saveDir = "model/checkpoints/3neurdata/"

def loadLosses():
    convLosses = []
    simpleLosses = []
    for i in range(minIdx, maxIdx):
        convLoss = np.loadtxt(saveDir + "conv/" + str(i) + "/losses", delimiter=',')
        if not min(convLoss[:,1]) == 1.0:
            convLosses.append(convLoss)
        simpleLosses.append(np.loadtxt(saveDir + "simple/" + str(i) + "/losses",
            delimiter=','))
    return convLosses, simpleLosses


def minLoss(losses):
    temp = min(enumerate(losses), key = lambda i: np.sum(i[1][:,1]))
    return temp[0], temp[1]
    
def maxLoss(losses):
    temp =  max(enumerate(losses), key = lambda i: np.sum(i[1][:,1]))
    return temp[0], temp[1]

def avgLoss(losses):
    print("Note that this method doesn't work")
    idxs = np.array([losses[0,:,0], np.mean(losses[:,:,1], axis=2)])
    return idxs

convLosses, simpleLosses = loadLosses()
#convLosses = np.array(sorted(convLosses, key=lambda i: np.sum(i[:,1])))
#simpleLosses = np.array(sorted(simpleLosses, key=lambda i: np.sum(i[:,1])))
#print(convLosses[0])
'''
print(avgLoss(np.array(convLosses)))
avgLoss(simpleLosses)

plt.subplot(211)
for data in convLosses:
    plt.semilogy(data[:,0], data[:,1])

plt.subplot(212)
for data in simpleLosses:
    plt.semilogy(data[:,0], data[:,1])
plt.show()
'''


minSimpleIdx, minSimpleLoss = minLoss(simpleLosses)
minConvIdx, minConvLoss = minLoss(convLosses)

simpleGraph, = plt.semilogy(minSimpleLoss[:,0], minSimpleLoss[:,1], label="simple")
convGraph, = plt.semilogy(minConvLoss[:,0], minConvLoss[:,1], label="conv")
plt.legend(handles=[simpleGraph, convGraph])
plt.show()

#print(minConvLoss)
#print(avgConvLoss)
#print(maxConvLoss)
