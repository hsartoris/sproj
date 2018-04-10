import sys
from scripts.GraphKit import diGraph, Simplex, SimplicialComplex, saveSparse, loadSparse
import numpy as np

verbosity=1

# modified version of previous script. First generates simplicial graphs, taking into account actual connectivity levels, then generates random graphs, using sample connectivity levels from simplicial graphs
if len(sys.argv) < 5:
	print("Usage: makeData2.py <numGraphs> <numNeurons> <percent> <prefix> [initIdx=0]")
	exit()
numGraphs = int(sys.argv[1]) # will generate numGraphs simplicial AND numGraphs random
numNeurons = int(sys.argv[2])
percent = float(sys.argv[3]) # relevant only for simplicial generation; random is dependent
prefix = sys.argv[4]
maxPlex = 8
minPlex = 2
initIdx = (0 if len(sys.argv) < 6 else int(sys.argv[5]))

connLevels = []
maxEdges = (numNeurons * (numNeurons - 1))/2

for i in range(initIdx, initIdx + numGraphs):
	if verbosity > 0: print("-"*40)
	if verbosity > 0: print("Generating graph " +  str(i))
	matrix = np.zeros((numNeurons, numNeurons), dtype=int)
	complexes = 0
	while np.sum(matrix) < percent * maxEdges:
		curr = float(np.sum(matrix)) / maxEdges
		if verbosity > 1: print(str(curr) + "% connected. Creating new simplicial complex.")
		target = np.random.randint(numNeurons)
		if verbosity > 2: print("Aiming for " + str(target) + " nodes in complex.")
		sc = SimplicialComplex()
		while sc.edges() < target:
			sc.addSimplex(Simplex(np.random.randint(minPlex,maxPlex)))
			if verbosity > 3: print("Added simplex of dimension " + str(sc.simplices[len(sc.simplices)-1].n))
		if verbosity > 2: print("Completed complex with " + str(sc.n) + " nodes.") 
		rules = []
		while len(rules) < sc.n:
			rule = np.random.randint(len(matrix))
			if not rule in rules:
				rules.append(rule)
		sc.map(matrix, rules)
		complexes += 1
	if verbosity > 0: print("Completed simplicial graph " + str(i) + " with " + str(complexes) + " simplicial complexes, and " + str(np.sum(matrix)) + " edges (" + str(float(np.sum(matrix))/maxEdges) + "% connected)")
	saveSparse(prefix + "/simplicial/" + str(i), matrix)
	np.savetxt(prefix + "/simplicial/" + str(i) + ".csv", matrix, delimiter=',', fmt='%i')
	connLevels.append(np.sum(matrix))
#	subprocess.call("pipeline/pipe.py " + prefix + str(i) + " " + str(timesteps) + " " +  prefix, shell=True)

for i in range(initIdx, initIdx + numGraphs):
	# generate random
	matrix = np.zeros((numNeurons, numNeurons), dtype=int)
	connTarget = connLevels.pop(np.random.randint(len(connLevels)))
	while np.sum(matrix) < connTarget:
		n1 = n2 = np.random.randint(len(matrix))
		while n2 == n1:
			n2 = np.random.randint(len(matrix))
		if (not matrix[n1][n2] > 0) and (not matrix[n2][n1] > 0):
			matrix[n1][n2] = 1
	saveSparse(prefix + "/random/" + str(i), matrix)
	np.savetxt(prefix + "/random/" + str(i) + ".csv", matrix, delimiter=',', fmt="%i")

if len(connLevels) > 0: print("probably shouldn't have anything left but there is")
