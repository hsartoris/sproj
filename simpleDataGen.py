#!/usr/bin/python3
from smallNetworkGen import genSimplex, simulate, optimizeSpike
import sys
import numpy as np
from multiprocessing import Process, Queue
import subprocess
import os
import model.scripts.Prettify
from model.scripts.GraphKit import diGraph, Simplex, SimplicialComplex, perturb
from utils.termVis import termVis

def simplicialNet():
    minPlex = 2
    maxPlex = 5
    target = 64
    sc = SimplicialComplex()
    while target > 0:
        stype = np.random.randint(minPlex, maxPlex)
        sc.addSimplex(Simplex(stype))
        target -= stype
    return np.pad(sc.blueprint, 4, 'constant', constant_values=0)

def tenNeurNet2():
    mat = np.matrix(np.zeros((10,10)))
    mat[1:3,0] = 1
    mat[2,1] = 1
    mat[5:8,4] = 1
    mat[6,5] = 1
    mat[7,5] = 1
    mat[7,6] = 1
    mat[8,2] = 1
    mat[9,8] = 1
    mat[6,9] = 1
    return mat

def tenNeurNet():
    # optimal rate is probably .18
    # 2-simplex at 0-2
    # no connection at 3
    # 4->5
    # 6->7->8
    # no connection at 9
    mat = np.matrix(np.zeros((10,10)))
    mat[1:3,0] = 1
    mat[2,1] = 1

    mat[5,4] = 1

    mat[7,6] = 1
    mat[8,7] = 1
    return mat, .18

def complexMat(spikeLevel):
    # .15 is probably a good spike rate for this net
    mat = np.matrix(np.zeros((6,6)))
    # 2-simplex
    mat[1,0] = spikeLevel
    mat[2,0] = spikeLevel
    mat[2,1] = spikeLevel
    # 3-cycle
    #mat[7,6] = spikeLevel
    #mat[6,8] = spikeLevel
    #mat[8,7] = spikeLevel
    #single connection
    mat[5,4] = spikeLevel
    return mat

def smallNetVar1():
    mat = np.matrix(np.zeros((3,3)))
    mat[0,2] = 1
    mat[1,2] = 1
    mat[0,1] = 1
    return mat

def smallNetVar3():
    mat = np.matrix(np.zeros((3,3)))
    mat[2,0] = 1
    mat[1,2] = 1
    mat[0,1] = 1
    return mat

def smallNetVar2():
    # loop
    mat = np.matrix(np.zeros((5,5)))
    mat[1,4] = 1
    mat[1,2] = 1
    mat[2,4] = 1
    return mat

def smallNetVar3():
    # only two connections
    mat = np.matrix(np.zeros((3,3)))
    mat[0,1] = 1
    mat[1,2] = 1
    return mat

def runData():
    out = "Neurons: " + str(neurons) + "\n"
    out += "Timesteps: " + str(steps) + "\n"
    out += "Run count: " + str(runs) + "\n"
    out += "Spike prob:" + str(spikeProb) + "\n"
    return out

def level1mat():
    mat = np.matrix(np.zeros((3,3)))

def watchCount(spikeDir):
    subprocess.call('watch -n 1 "ls ' + spikeDir + ' | wc -l"', shell=True)

def randomData(n, timesteps):
    return np.random.randint(2, size=(n, timesteps))

def saveRandomData(n, params, dataDir, q):
    startIdx = params['startIdx']
    for run in range(params['runs']):
        q.put(run + startIdx)
        np.savetxt(dataDir + "/spikes/" + str(run+startIdx) + ".csv",
                randomData(n, params['timesteps']), delimiter=',', fmt='%i')


if __name__=="__main__":
    mat, spikeProb = tenNeurNet()
    newMat = np.zeros(mat.shape*2)
    newMat[:mat.shape[0],:mat.shape[1]] = mat
    newMat[mat.shape[0]:,mat.shape[1]:] = mat
    mat = newMat
    print(mat)
    print(np.sum(mat))
    #mat = perturb(np.pad(mat, 3, mode='constant'), .05)
    print(mat)
    print(np.sum(mat))
    #mat = genSimplex(3)
    #mat = simplicialNet()
    #mat = smallNetVar3()
    if not (input("neurons: " + str(mat.shape[0]) + ", continue?[Y/n]") or "Y").lower() == 'y': exit()
    #spikeProb = .005
    spikeProb = .15
    if len(sys.argv) == 2 and sys.argv[1] == "optimize":
        spikeProb = optimizeSpike(mat, 0.0)
        print("Optimized rate:", spikeProb)
        exit()

    if len(sys.argv) < 4:
        print("Usage: simpleDataGen.py outDir timesteps runs [spikeProb]\n\tNote: assumes dataStaging")
        sys.exit()
    
    outDir = "dataStaging/" + sys.argv[1] + "/"
    #neurons = int(sys.argv[2])
    neurons = mat.shape[0]
    steps = int(sys.argv[2])
    runs = int(sys.argv[3])
    spikeDir = outDir + "/spikes"
    
    #spikeProb = (.05 if len(sys.argv) < 5 else float(sys.argv[4]))
    runBlocks = 8
    runBlockSize = int(runs/runBlocks)

    if not os.path.exists(outDir + "/spikes"):
        os.makedirs(outDir + "/spikes")

    # pretty much unrelated code for making random or empty matrices

    empty = False
    random = False
    if empty:
        mat = genSimplex(3)
        spikes = np.zeros((3, steps))
        for i in range(runs):
            np.savetxt(outDir + "spikes/" + str(i) + ".csv", spikes,
                    delimiter=',', fmt='%i')
        np.savetxt(outDir + "struct", mat, delimiter=',')
        exit()
    elif random:
        mat = genSimplex(3)


    threads = []
    q = Queue()
    print("Starting simulation of " + str(runs) + " runs of " + str(steps) 
        + "timesteps, with a spike probability of " + str(spikeProb))

    for i in range(runBlocks):
        start = i * runBlockSize
        if i < runBlocks - 1: runCount = runBlockSize
        else: runCount = runs - start
        params = { 'startIdx':start, 'runs':runCount, 'timesteps':steps, 
            'spikeProb':spikeProb}
        if not random:
            threads.append(Process(target=simulate, args=(mat, params, outDir, q)))
        else:
            threads.append(Process(target=saveRandomData, args=(3, params, outDir, q)))
    # mat now defined above, look at it dumbass
    # for 2-simplex, .25 is probably a good rate
    #mat = genSimplex(3)
    #mat = smallNetVar3()
    #mat = tenNeurNet()
    #mat = complexMat(1)

    for i in range(len(threads)):
        threads[i].start()

    arrow = model.scripts.Prettify.pretty()
    count = 0
    while count < runs :
        try:
            q.get()
            count += 1
            arrow.arrow(count, runs)
            sleep(.5)
        except Exception:
            pass
    for i in range(len(threads)):
        threads[i].join()
    print(runData())
    f = open(outDir + "runData", "w+")
    f.write(runData())
    f.close()
    np.savetxt(outDir + "struct", mat, delimiter=',')
