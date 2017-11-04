import sys
import numpy as np
import networkx as nx

if sys.version_info[0] < 3:
	raise Exception("Requires Python 3, for stupid reasons")

def loadMatrix(filename):
	return np.loadtxt(filename, delimiter=',')

def graphCSV(filename):
	Gx = nx.drawing.nx_pydot.to_pydot(nx.DiGraph(np.loadtxt(filename, delimiter=',')))
	Gx.write_png(filename.split('.')[0] + ".png", prog='dot')
