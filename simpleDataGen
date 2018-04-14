#!/usr/bin/python3
from smallNetworkGen import genSimplex, simulate
import sys
import numpy as np

def tenNeurNet():
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

def complexMat(spikeLevel):
    mat = np.matrix(np.zeros((9,9)))
    # 2-simplex
    mat[1,0] = spikeLevel
    mat[2,0] = spikeLevel
    mat[2,1] = spikeLevel
    # 3-cycle
    mat[7,6] = spikeLevel
    mat[6,8] = spikeLevel
    mat[8,7] = spikeLevel
    #single connection
    mat[5,4] = spikeLevel
    return mat

def level1mat():
    mat = np.matrix(np.zeros((3,3)))

if __name__=="__main__":
    if len(sys.argv) < 5:
        print("Usage: simpleDataGen.py outDir neurons timesteps runs [spikeProb]")
        sys.exit()
    
    outDir = sys.argv[1] + "/"
    #neurons = int(sys.argv[2])
    neurons = 9
    steps = int(sys.argv[3])
    runs = int(sys.argv[4])
    
    spikeProb = (.1 if len(sys.argv) < 6 else float(sys.argv[5]))
    
    params = { 'runs':runs, 'timesteps':steps, 'spikeProb':spikeProb}
    
    #mat = genSimplex(neurons)
    #mat = tenNeurNet()
    mat = complexMat(.9)
    simulate(mat, params, outDir)
    np.savetxt(outDir + "struct.csv", mat, delimiter=',')
