from time import time
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pylab
import warnings
import os
import sys
from math import exp

warnings.filterwarnings("ignore", category=DeprecationWarning)  # lol
plt.axis('off')

options = "generate|simulate|graph"

def genNx(w):
	G = nx.DiGraph()
	for i in range(0, len(w)):
		for j in range(0, len(w[i])):
			if w[i][j] > 0:
				G.add_edges_from([(str(j), str(i))], weight=w[i][j])
	return G


def drawNx(g, drawEdgeLabels=False):
	# draw the graph
	edgeLabels = dict([((u, v), d['weight'])
					   for u, v, d in g.edges(data=True)])
	nodeLabels = {node: node for node in g.nodes()}
	pos = nx.spring_layout(g)
	if drawEdgeLabels:
		nx.draw_networkx_edge_labels(g, pos, edge_labels=edgeLabels)
		nx.draw_networkx(g, pos, labels=nodeLabels)
	else:
		nx.draw_networkx(g, labels=nodeLabels)
	pylab.show()


def randWeightMatrix(numNeurons, cycles, isVerbose, minWeight=1, maxWeight=1):
	# returns a 2d array of weights defining a fully connected, somewhat reasonable network
	out = np.arange(numNeurons)
	mem = np.array([])
	w = np.zeros(shape=(numNeurons, numNeurons))
	mem = np.append(mem, random.randint(0, len(out) - 1))
	out = np.delete(out, mem[0])
	if isVerbose: print("First node: " + str(mem[0]))
	while len(out) > 0:
		# choose a node in the network & a node outside the network
		# add a directed edge from the latter to the former & add note that the other is now in the network
		memberNode = mem[random.randint(0, len(mem) - 1)]  # actual index of the node in the weight matrix
		newMember = random.randint(0, len(out) - 1)  # index of actual index
		if isVerbose: print("Adding node " + str(out[newMember]) + " connected to " + str(int(memberNode)))
		# w[memberNode] is the input vector for memberNode
		w[int(memberNode)][int(out[newMember])] = random.uniform(minWeight, maxWeight)
		mem = np.append(mem, out[newMember])
		out = np.delete(out, newMember)

	if isVerbose:
		print("Pre-cycle matrix: ")
		print(w)

	# insert the random cycles
	while cycles > 0:
		node1 = node2 = mem[random.randint(0, len(mem) - 1)]
		while node1 == node2:
			# make sure we have different nodes
			node2 = mem[random.randint(0, len(mem) - 1)]
		w[int(node1)][int(node2)] = random.uniform(minWeight, maxWeight)
		if isVerbose: print("Adding connection from node " + str(int(node2)) + " to node " + str(int(node1)))
		cycles = cycles - 1
	if isVerbose:
		print("Post-cycle matrix:")
		print(w)
	return w


def writeMatrix(w, ID="w" + str(int(time()))[-5:]):
	np.set_printoptions(threshold=np.inf, linewidth=np.inf)
	if not os.path.exists(ID):
		os.makedirs(ID)  # this is perhaps not entirely safe but no one else should be in this dir
	np.savetxt(ID + "/w.csv", w, delimiter=',')


def generate():
	# incoming ternary operators uh oh
	num = raw_input("Generate how many networks? [1] ")
	num = (int(num) if len(num) > 0 else 1)

	neur = raw_input("How many neurons per network? [20] ")
	neur = (int(neur) if len(neur) > 0 else  20)

	cycles = raw_input("How many cycles? (if c > 1, c cycles will be introduced. if 0 <= c < 1, treated as percentage of neuron input) [.25] ")
	cycles = (float(cycles) if len(cycles) > 0 else .25)
	if cycles >= 1: cycles = int(cycles)

	verbosity = raw_input("Verbose mode? [N/y]").lower() == 'y'

	minWeight = raw_input("Minimum weight [1] ")
	minWeight = (float(minWeight) if len(minWeight) > 0 else 1)
	maxWeight = raw_input("Maximum weight [1] ")
	maxWeight = (float(maxWeight) if len(maxWeight) > 0 else 1)

	prefix = raw_input("Output prefix: ")
	if not os.path.exists(prefix):
			os.makedirs(prefix)
	else:
		print("Prefix exists, exiting.")
		exit()
	
	for i in range(0, num):
		w = randWeightMatrix(neur, cycles if cycles >= 1 else int(cycles * neur), verbosity, minWeight, maxWeight)
		writeMatrix(w, prefix + "/" + str(i))

def logisticf(x, th):
	steepness = 20 # arbitrary?
	return 1/(1 + exp((th-x) * steepness))

def simulate():
	prefix = raw_input("Input prefix: ")
	maxIdx = raw_input("Highest folder index: [0] ")
	maxIdx = (int(maxIdx) if len(maxIdx) > 0 else 0)
	
	spikeChance = raw_input("Spike chance: [.1]")
	spikeChance = (float(spikeChance) if len(spikeChance) > 0 else .1)
	
	
	for i in range(0,maxIdx+1):
		w = np.genfromtxt(prefix + "/" + str(i) + "/w.csv", delimiter=',')
		s = np.random.randint(2, size=len(w))
	# TODO
	
	print("todo")


def main():
	if len(sys.argv) > 1:
		if str(sys.argv[1]) == "generate":
			generate()
		elif str(sys.argv[1]) == "simulate":
			simulate()
			#TODO
		elif sys.argv[1] == "graph":
			prefix = raw_input("Enter folder name (assumes network is w.csv): ")
			showEdges = raw_input("Label edges? [N/y]" ).lower() == 'y'
			w = np.genfromtxt(prefix + "/w.csv", delimiter=',')
			drawNx(genNx(w), drawEdgeLabels=showEdges)
		else:
			print("Syntax: " + sys.argv[0] + " [" + options + "]")
		# TODO: handle
		exit()
	else:
		print("Syntax: " + sys.argv[0] + " [" + options + "]")



if __name__ == "__main__":
	main()
