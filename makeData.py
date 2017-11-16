import sys
from scripts.GraphKit import diGraph, Simplex, SimplicialComplex, saveSparse, loadSparse
import numpy as np
import pipeline.pipe
import subprocess

verbosity=1

if len(sys.argv) < 6:
	print("Syntax: <num graphs> <num neurons> <percent connections> <timesteps> <prefix>")

numGraphs = int(sys.argv[1])
numNeurons = int(sys.argv[2])
percent = float(sys.argv[3])
timesteps = int(sys.argv[4])
prefix = sys.argv[5]
maxPlex = 6
minPlex = 2
simplicial = int(sys.argv[6]) == 1

for i in range(numGraphs):
	if verbosity > 0: print("-"*40)
	if verbosity > 0: print("Generating graph " +  str(i))
	matrix = np.zeros((numNeurons, numNeurons), dtype=int)
	maxEdges = (numNeurons * (numNeurons - 1))/2
	complexes = 0
	while np.sum(matrix) < percent * maxEdges:
		curr = float(np.sum(matrix)) / maxEdges
		if simplicial:
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
		else:
			if verbosity > 1 and complexes%5 == 0: print(str(curr) + "% connected. Adding new random edge.")
			n1 = n2 = np.random.randint(len(matrix))
			while n2 == n1:
				n2 = np.random.randint(len(matrix))
			if (not matrix[n1][n2] > 0) and (not matrix[n2][n1] > 0):
				matrix[n1][n2] = 1
			complexes += 1
	if verbosity > 0: print("Completed graph " + str(i) + " with " + (str(complexes) + " simplicial complexes, and " if simplicial else "") + str(np.sum(matrix)) + " edges (" + str(float(np.sum(matrix))/maxEdges) + "% connected)")
	#np.savetxt(prefix + str(i) + ".csv", matrix, delimiter=',', fmt='%i')
	saveSparse(prefix + str(i), matrix)
	subprocess.call("pipeline/pipe.py " + prefix + str(i) + " " + str(timesteps) + " " +  prefix, shell=True)
