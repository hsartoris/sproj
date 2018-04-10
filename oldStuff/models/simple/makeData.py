import simple
import numpy as np
import sys
import os
from backports.shutil_get_terminal_size import get_terminal_size

def correlation(prev, curr):
    prev = np.array(prev.transpose())[0]
    curr = np.array(curr.transpose())[0]
    m = np.zeros(shape=(len(prev), len(prev)))
    for i in range(len(prev)):
        if prev[i] == 0: continue
        for j in range(len(curr)):
            if i == j or curr[j] == 0: continue
            m[i][j] = (curr[j] - prev[j])/prev[i]
    return np.array([m])

if __name__ == "__main__":
    loading = ["|", "/", "-", "\\", "*"]
    prefix = "data1"
    iterations = 1000
    networks = 100000
    neurons = 6
    cols = get_terminal_size().columns
    spikeChance = .3
    threshold = .5
    initSpikeChance = .5
    print("Generating matrices...")
    for i in range(0, networks):
        w = simple.randWeightMatrix(neurons,0,False)
        os.makedirs(prefix + "/" + str(i))
        np.savetxt(prefix + "/" + str(i) + "/w.csv", w, delimiter=',')
        sys.stdout.write("\r{0}>".format("="*(cols*i/networks)))
        sys.stdout.flush()
    sys.stdout.write("\r{0}".format("="*cols))
    sys.stdout.flush()
    print("Simulating...")
    pFloor = np.vectorize(simple.probFloor)
    logistic = np.vectorize(simple.logisticf)
    count = 0

    for i in range(0, networks):
        w = np.matrix(np.genfromtxt(prefix + "/" + str(i) + "/w.csv", delimiter=','))
        s = np.matrix(pFloor(np.random.rand(len(w), 1), initSpikeChance))
        out = s
        totalCorr = np.zeros(shape=(len(w), len(w)))
        correlations = np.array([np.zeros(shape=(len(w), len(w)))])
        for j in range(0, iterations):
            prev = s
            s = w * s
            s = s + np.matrix(pFloor(np.random.rand(len(w),1), spikeChance))
            s = logistic(s, threshold)
            out = np.append(out, s, 1)
            corr = correlation(prev, s)
            correlations = np.append(correlations, corr, 0)
            totalCorr += corr[0]
            sys.stdout.write("\r{0}>".format(loading[(count/600)%len(loading)] + ("="*((cols-3)*i/networks))))
            sys.stdout.flush()
            count += 1
        np.savetxt(prefix + "/" + str(i) + "/raw.csv", out, delimiter=',')
        np.savetxt(prefix + "/" + str(i) + "/correlations.csv", correlations.reshape(-1, correlations.shape[-1]), delimiter=',')
        #print(totalCorr)
	np.savetxt(prefix + "/" + str(i) + "/total.csv", totalCorr, delimiter=',')
    sys.stdout.write("\r{0}".format("="*cols))
    sys.stdout.flush()
    print("Done simulating")
