#!/usr/bin/python3
from smallNetworkGen import genSimplex, simulate
import sys
import numpy as np

if len(sys.argv) < 5:
    print("Usage: simpleDataGen.py outDir neurons timesteps runs [spikeProb]")
    sys.exit()

outDir = sys.argv[1] + "/"
neurons = int(sys.argv[2])
steps = int(sys.argv[3])
runs = int(sys.argv[4])

spikeProb = (.15 if len(sys.argv) < 6 else float(sys.argv[5]))

params = { 'runs':runs, 'timesteps':steps, 'spikeProb':spikeProb}

mat = genSimplex(neurons)
simulate(mat, params, outDir)
np.savetxt(outDir + "struct.csv", mat, delimiter=',', fmt='%i')
