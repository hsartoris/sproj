#!/usr/bin/python3
"""
Small (3 neuron) network generator. Produces and saves network structure in <dataDir> with <runs> number of outputs of <timesteps> length. If <dataDir> exists, attempts to load existing network structure.

Without specifying <runs> or <timesteps>, attempts to load from saved parameters.

Usage:
    smallNetworkGen.py <dataDir> [<runs>] [<timesteps>] 
        [-o|-O] [-c|--complex] [-q|--quiet] [-t]
    smallNetworkGen.py -h | --help

Options:
    -h --help       Show this screen
    -c --complex    Use weighted connections instead of 1
    -o              Overwrites existing outputs; otherwise appends
    -O              Overwrites network structure and output
    -q --quiet      Suppresses debug output
    -t              Testing mode; delete all created files on exit
"""

from docopt import docopt
import os, shutil, prettify
import numpy as np

structName = "struct.csv"
paramsName = "params"
spikeDir = "spikes/"
params = dict()
spikeProb = .3
params['spikeProb'] = spikeProb
NUM_NEUR = 3

def genMatrix(simple=True):
    global NUM_NEUR
    mat = np.matrix(np.zeros(shape=(NUM_NEUR,NUM_NEUR)))
    # 0 -> 1 -> 2
    # 0    ->   2
    mat[1,0] = (1 if simple else np.random.random())
    mat[2,0] = (1 if simple else np.random.random())
    mat[2,1] = (1 if simple else np.random.random())
    return mat


def loadParams(dataDir):
    f = open(dataDir + paramsName)
    p = f.readlines()
    params = { 'runs':int(p[0]), 'timesteps':int(p[1]) } 
    f.close()
    return params

def randSpikeCol(spikeProb):
    global NUM_NEUR
    # returns column matrix of 1/0 with spikeProb chance of 1
    return np.matrix(np.random.choice(2, NUM_NEUR, p=[1-spikeProb, spikeProb])).transpose()

def simulate(matrix, params, dataDir, simple=True):
    arrow = prettify.pretty()
    global NUM_NEUR
    global verbose
    # for now this just overwrites, and assumes simple
    if not os.path.exists(dataDir + spikeDir): 
        if verbose: print("creating spike directory" + dataDir + spikeDir)
        os.makedirs(dataDir + spikeDir)
    for run in range(params['runs']):
        arrow.arrow(run, params['runs'])
        data = np.matrix(np.zeros((NUM_NEUR,params['timesteps'])))
        data[:,0] = randSpikeCol(params['spikeProb'])
        for step in range(1,params['timesteps']):
            data[:,step] = np.clip((matrix * data[:,step-1]) + 
                                    randSpikeCol(params['spikeProb']), 0, 1)
        np.savetxt(dataDir + spikeDir + str(run) + ".csv", data, delimiter=',', fmt='%i')
    

if __name__ == "__main__":
    arguments = docopt(__doc__)
    verbose = not arguments['--quiet']
    if verbose: print(arguments)

    dataDir = arguments['<dataDir>'] + "/"
    dirExists = os.path.isdir(dataDir)
    overwriteData = arguments['-o']
    overwriteAll = arguments['-O']
    if dirExists and verbose: print("dataDir exists; locating existing network structure")

    if dirExists:
        if os.path.exists(dataDir + structName) and not overwriteAll:
            # structure already defined
            if verbose: print("Found network structure")
            matrix = np.loadtxt(dataDir + structName, delimiter=',')
        else:
            if verbose: print("Generating new network structure")
            matrix = genMatrix()
        
        if os.path.exists(dataDir + spikeDir) and (overwriteData or overwriteAll):
            if verbose: print("Deleting old spike directory")
            shutil.rmtree(dataDir + spikeDir)
    
        if os.path.exists(dataDir + paramsName):
            # parameters already defined
            params = loadParams(dataDir)
            if arguments['<runs>'] is not None or arguments['<timesteps>'] is not None:
                # new params found
                print("Warning! Found existing parameters file. Backing up to " + paramsName + ".bak")
                if not (overwriteData or overwriteAll):
                    # overwrite not enabled; check with user
                    cont = input("Continue or revert? [C/n] ")
                    if cont == 'n': exit()

                os.rename(dataDir + paramsName, dataDir + paramsName + ".bak")
                if arguments['<runs>'] is not None: 
                    params['runs'] = int(arguments['<runs>'])
                if arguments['<timesteps>'] is not None:
                    params['timesteps'] = int(arguments['<timesteps>'])
        # at this point, params are loaded and matrix is generated or loaded
        # TODO: save params
        params['spikeProb'] = spikeProb
        simulate(matrix, params, dataDir)
    else:
        # from scratch
        if arguments['<runs>'] is None or arguments['<timesteps>'] is None:
            print("No existing parameters found. Please supple <runs> and <timesteps>.")
            exit()
        os.makedirs(dataDir)
        matrix = genMatrix(not arguments['--complex'])
        np.savetxt(dataDir + structName, matrix, delimiter=',')
        params['runs'] = int(arguments['<runs>'])
        params['timesteps'] = int(arguments['<timesteps>'])
        params['spikeProb'] = spikeProb
        p = open(dataDir + paramsName, 'w+')
        p.write("{}\n{}\n".format(params['runs'], params['timesteps']))
        p.close()
        simulate(matrix, params, dataDir)


    if arguments['-t']:
        # delete directory
        if verbose: print("deleting " + dataDir)
        shutil.rmtree(dataDir)

