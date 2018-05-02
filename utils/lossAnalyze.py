#!/usr/bin/python3
import numpy as np
plot = True
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plot = False
    pass
import os
import sys
minIdx = 1
maxIdx = 99
saveDir = "model/checkpoints/bench/3neur/"

def loadLosses():
    convLosses = []
    dumbLosses = []
    dirs = os.listdir(saveDir)
    for d in dirs:
        if d == "15k": continue
        try:
            convLoss = np.loadtxt(saveDir + d + "/conv/vlosses", delimiter=',')
            if not min(convLoss) == 1.0:
                convLosses.append(convLoss)
        except OSError:
            pass
        try:
            dumbLoss = np.loadtxt(saveDir + d + "/dumb/vlosses", delimiter=',')
            dumbLosses.append(dumbLoss)
        except OSError:
            pass
    print("Loaded", len(convLosses), "conv losses and", len(dumbLosses), "dumb losses")
    return convLosses, dumbLosses

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

convLosses, dumbLosses = loadLosses()
#convLosses = np.array(sorted(convLosses, key=lambda i: np.sum(i[:,1])))
#simpleLosses = np.array(sorted(simpleLosses, key=lambda i: np.sum(i[:,1])))
#print(convLosses[0])
#print(avgLoss(np.array(convLosses)))

# min final conv idx, min final conv loss
mFCi, mFCL = minFinalLoss(convLosses)
mFCi = dirs[mFCi]
minFinalConv = np.array([steps, mFCL]).transpose()

# min final dumb idx, min final dumb loss
mFDi, mFDL = minFinalLoss(dumbLosses)
mFDi = dirs[mFDi]
minFinalDumb = np.array([steps, mFDL]).transpose()

outDir = input("Where to save this shit to: ")
if not outDir == '':
    np.savetxt(outDir + "/minFinalConv_" + mFCi, minFinalConv, fmt='%i, %.4f')
    np.savetxt(outDir + "/minFinalDumb_" + mFDi, minFinalDumb, fmt='%i, %.4f')

print("Minimum final conv loss from run" + mFCi + ": " + str(mFCL[-1]))
print("Minimum final dumb loss from run" + mFDi + ": " + str(mFDL[-1]))

if plot:
    if plotall:
        plt.subplot(211)
        for data in convLosses:
            plt.semilogy(steps, data)
        
        plt.subplot(212)
        for data in dumbLosses:
            plt.semilogy(steps, data)
        plt.show()
    
    else:
        minTotalDumbIdx, minTotalDumb = minLoss(dumbLosses)
        minTotalDumbIdx = dirs[minTotalDumbIdx]

        minTotalConvIdx, minTotalConv = minLoss(convLosses)
        minTotalConvIdx = dirs[minTotalConvIdx]

        print("minFinalIdx:", mFCi)
        print("minTotalIdx:", minTotalConvIdx)
        avgConvLoss = avgLoss(convLosses)
        avgDumbLoss = avgLoss(dumbLosses)

        avgConv, = plt.semilogy(steps, avgConvLoss, label="avg_conv")
        avgDumb, = plt.semilogy(steps, avgDumbLoss, label="avg_simple")
                
        totalConv, = plt.semilogy(steps, minTotalConv, label=("min total" + 
            "conv: " + minTotalConvIdx))
        
        totalDumb, = plt.semilogy(steps, minTotalDumb, label=("min total" + 
            "dumb: " + minTotalDumbIdx))

        minConv, = plt.semilogy(steps, mFCL, label=("min final conv loss: " + 
            mFCi))
        minDumb, = plt.semilogy(steps, mFDL, label=("min final dumb loss: " + 
            mFDi))
        plt.legend(handles=[avgConv, avgDumb, totalConv, totalDumb, minConv, 
            minDumb])
        plt.show()
        
        #print(minConvLoss)
        #print(avgConvLoss)
        #print(maxConvLoss)
