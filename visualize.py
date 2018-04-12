#!/usr/bin/python3
import sys, os
from model.scripts import matViz
import numpy as np
if len(sys.argv) < 2:
    print("Usage: visualize.py <runId>")
    sys.exit()

runId = sys.argv[1]
if not os.path.exists("visData/" + runId):
    print("data not found")
    sys.exit()

visDir = "visData/" + runId + "/"

out0 = np.loadtxt(visDir + "out0", delimiter=',')
out1 = np.loadtxt(visDir + "out1", delimiter=',')
outf = np.expand_dims(np.loadtxt(visDir + "outf", delimiter=','), 0)
pred = np.expand_dims(np.loadtxt(visDir + "pred", delimiter=','), 0)

matViz(out0, visDir + "out0.png")
matViz(out1, visDir + "out1.png")
matViz(outf, visDir + "outf.png")
matViz(pred, visDir + "pred.png")
