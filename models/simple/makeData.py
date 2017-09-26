import simple
import numpy as np
import sys
import os
from backports.shutil_get_terminal_size import get_terminal_size

if __name__ == "__main__":
    loading = ["|", "/", "-", "\\", "*"]
    prefix = "data2"
    iterations = 1000
    networks = 1000
    cols = get_terminal_size().columns
    spikeChance = .3
    threshold = .5
    initSpikeChance = .5
    print("Generating matrices...")
    for i in range(0, networks):
        w = simple.randWeightMatrix(6,0,False)
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
        for j in range(0, iterations):
            s = w * s
            s = s + np.matrix(pFloor(np.random.rand(len(w),1), spikeChance))
            s = logistic(s, threshold)
            out = np.append(out, s, 1)
            sys.stdout.write("\r{0}>".format(loading[(count/600)%len(loading)] + ("="*((cols-3)*i/networks))))
            sys.stdout.flush()
            count += 1
        np.savetxt(prefix + "/" + str(i) + "/raw.csv", out, delimiter=',')
    sys.stdout.write("\r{0}".format("="*cols))
    sys.stdout.flush()
    print("Done simulating")
