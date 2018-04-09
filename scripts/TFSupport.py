import numpy as np
import tensorflow as tf
import time

def dumpData(fdir, printout=True):
    testX, testY, _ = testing.next(1)
    print(sess.run(pred, feed_dict={_data: testX, _labels: testY}))
    layer0w = weights['layer0'].eval()
    layer2in = weights['layer2_in'].eval()
    layer2out = weights['layer2_out'].eval()
    finalw = weights['final'].eval()
    if printout:
        print("Layer 0 weights:")
        print(layer0w)
        print("Layer 2 in weights:")
        print(layer2in)
        print("Layer 2 out weights:")
        print(layer2out)
        print("Final layer weights:")
        print(finalw)
    
    f = open(fdir + "/dump",  "w+")
    print("Model trained on:", prefix, file=f)
    t = time.localtime(time.time())
    f.write(str(t.tm_hour) + ":" + ("0"+str(t.tm_min) if t.tm_min < 10 else str(t.tm_min))
            + ":" + ("0"+str(t.tm_sec) if t.tm_sec < 10 else str(t.tm_sec))
            + " " + str(t.tm_mon) + "/" + str(t.tm_mday) + "/" + str(t.tm_year) + "\n")
    print("batchSize: " + str(batchSize), file=f)
    print("timesteps: " + str(timesteps), file=f)
    print("trainingSteps: " + str(trainingSteps), file=f)
    print("runNumber: " + str(runNumber), file=f)
    print("initLearningRate: " + str(initLearningRate), file=f)
    f.close()

    np.savetxt(fdir + "/w_0.csv", layer0w, delimiter=',')
    np.savetxt(fdir + "/w_2_in.csv", layer2in, delimiter=',')
    np.savetxt(fdir + "/w_2_out.csv", layer2out, delimiter=',')
    np.savetxt(fdir + "/w_f.csv", finalw, delimiter=',')
