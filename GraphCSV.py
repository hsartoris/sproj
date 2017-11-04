#!/usr/bin/python3

import sys
import numpy as np
import networkx as nx

if sys.version_info[0] < 3:
	raise Exception("Requires Python 3, for stupid reasons")

if __name__ == "__main__" and len(sys.argv[1]) < 0:
	raise Exception("Takes a file as input")

def loadMatrix(filename):
	return np.loadtxt(filename, delimiter=',')

def graphCSV(filename):
	Gx = nx.drawing.nx_pydot.to_pydot(nx.DiGraph(np.loadtxt(filename, delimiter=',')))
	Gx.write_png(filename.split('.')[0] + ".png", prog='dot')

if __name__ == "__main__":
	graphCSV(sys.argv[1])
