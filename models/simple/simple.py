from time import time
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pylab
import warnings
import os
import sys
import argparse
#import nest		# MUST BE REENABLED TO USE NEST FUNCTIONS
#import nest.voltage_trace
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

def makeNest(w):
	#neurons = []
	#for i in range(0, len(w)):
	#	neurons += [nest.Create("iaf_psc_alpha")]
	# this assumes no nest network has been created before1
	pop = nest.Create("iaf_psc_alpha", len(w))
	for i in range(0, len(w)):
		for j in range(0, len(w[i])):
			if w[i][j] > 0:
				syn_spec={"weight":w[i][j]}
				nest.Connect([j+1], [i+1])
				#nest.Connect(neurons[j],neurons[i])
	nest.PrintNetwork()

	poiss = nest.Create("poisson_generator", len(w))
	volts = nest.Create("voltmeter", len(w))
	conn_dict = {'rule': 'one_to_one'}
	nest.Connect(poiss, pop, conn_dict)
	nest.Connect(volts, pop, conn_dict)
	nest.Simulate(1000)
	nest.PrintNetwork()
	print("Trying to show voltage trace")
	nest.voltage_trace.from_device((volts[0],))
	nest.voltage_trace.show()


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

def addConnection(w, f=-1, t=-1, weight=1, verbose=False):
	if f >= 0 and t >= 0:
		w[int(t)][int(f)] = weight
		if verbose:
			print("Adding connection from " + str(int(f)) + " to " + str(int(t)) + ".")
	else:
		print("Tried to add with bad indices")

def randWeightMatrix(numNeurons, cycles, verbose, minWeight=1, maxWeight=1, makeTris=False, numTris=0):
	# returns a 2d array of weights defining a fully connected, somewhat reasonable network
	if makeTris and numTris == 0:
		soloNeurons = numNeurons % 3
		numTris = int(numNeurons / 3)
	elif numTris > 0:
		if numTris * 3 < numNeurons: soloNeurons = numNeurons - (numTris * 3)
	else:
		if verbose: print("fuck this")
		# fuck this
	#numNeurons = (numPrimaries if not makeTris else (numPrimaries * 3) + soloNeurons)
	out = np.arange(numNeurons) # neurons not yet in network
	mem = np.array([])							# neurons in network
	w = np.zeros(shape=(numNeurons, numNeurons))
	mem = np.append(mem, random.randint(0, len(out) - 1))
	out = np.delete(out, mem[0])
	solo = np.array([])
	if verbose: print("First node: " + str(int(mem[0])))
	if makeTris:
		tris = np.zeros(shape=(numTris, 3))
		tris[0][0] = mem[0]
	#while len(out) > 0:
	for i in range(1, numNeurons):
		newMember = random.randint(0, len(out) - 1) # index of actual index
		if not makeTris:
			# choose a node in the network & a node outside the network
			# add a directed edge from the latter to the former & add note that the other is now in the network
			memberNode = mem[random.randint(0, len(mem) - 1)]  # actual index of the node in the weight matrix
		elif i < numTris * 3:
			tris[int(i/3)][i%3] = int(out[newMember])
			if i%3 != 0:
				# if in the middle of defining a triangle
				memberNode = mem[i-1] # get index in weight matrix of previous
				if i%3 == 2:
					# connect back to first part of triangle
					addConnection(w, f=int(mem[i-2]), t=int(out[newMember]), weight=random.uniform(minWeight,maxWeight), verbose=verbose)
			else:
				# 3rd tri
				memberNode = -1 # signal not to connect this yet
		else:
			memberNode = -1
			#solo = np.append(solo, out[newMember])
			tris = np.concatenate((tris, np.array([[out[memberNode], out[memberNode], out[memberNode]]])), axis=0)
			# ok this is hacky
		if memberNode != -1:
			addConnection(w, f=int(out[newMember]), t=int(memberNode), weight=random.uniform(minWeight,maxWeight), verbose=verbose)

		mem = np.append(mem, out[newMember])
		out = np.delete(out, newMember)
	# at this point all nodes should be connected OR all triangles should be formed and last chunk are unconnected solo neurons
	if verbose and makeTris: print("Triangles assembled; connecting")
	if makeTris:
		# this actually is probably extensible to any dimension shape
		# like 1d......
		addedTris = np.array([[]])
		i = random.randint(0, len(tris)-1)
		addedTris = np.concatenate((addedTris, np.array([tris[i]])), axis=1)
		tris = np.delete(tris, i, 0)
		while len(tris) > 0:
			nextTri = random.randint(0, len(tris)-1)
			currTri = random.randint(0, len(addedTris)-1)
			addConnection(w, f=tris[nextTri][random.randint(0,len(tris[nextTri])-1)], t=addedTris[currTri][random.randint(0,len(addedTris[currTri])-1)], weight=random.uniform(minWeight, maxWeight), verbose=verbose)
			# god damn
			addedTris = np.concatenate((addedTris, np.array([tris[nextTri]])), axis=0)
			tris = np.delete(tris, nextTri, 0)
		if verbose: print("Connected tris")

	# TODO: soloNeurons

	if verbose:
		print("Pre-cycle matrix: ")
		print(w)

	# insert the random cycles
	while cycles > 0:
		#node1 = node2 = mem[random.randint(0, len(mem) - 1)]
		node1 = node2 = random.randint(0, len(w) - 1)
		while node1 == node2:
			# make sure we have different nodes
			#node2 = mem[random.randint(0, len(mem) - 1)]
			node2 = random.randint(0, len(w) -1)
		addConnection(w, f=node1, t=node2, weight=random.uniform(minWeight,maxWeight), verbose=verbose)
		#w[int(node1)][int(node2)] = random.uniform(minWeight, maxWeight)
		#if verbose: print("Adding connection from node " + str(int(node2)) + " to node " + str(int(node1)))
		cycles = cycles - 1
	if verbose:
		print("Post-cycle matrix:")
		print(w)
	return w


def writeMatrix(w, ID="w" + str(int(time()))[-5:]):
	np.set_printoptions(threshold=np.inf, linewidth=np.inf)
	if not os.path.exists(ID):
		os.makedirs(ID)  # this is perhaps not entirely safe but no one else should be in this dir
	np.savetxt(ID + "/w.csv", w, delimiter=',')


def generate(num, neur, cycles, verbosity, minWeight, maxWeight, prefix):
	# incoming ternary operators uh oh
	#num = raw_input("Generate how many networks? [1] ")
	#num = (int(num) if len(num) > 0 else 1)

	#neur = raw_input("How many neurons per network? [20] ")
	#neur = (int(neur) if len(neur) > 0 else  20)

	#cycles = raw_input("How many cycles? (if c > 1, c cycles will be introduced. if 0 <= c < 1, treated as percentage of neuron input) [.25] ")
	#cycles = (float(cycles) if len(cycles) > 0 else .25)
	if cycles >= 1: cycles = int(cycles)

	#verbosity = raw_input("Verbose mode? [N/y]").lower() == 'y'

	#minWeight = raw_input("Minimum weight [1] ")
	#minWeight = (float(minWeight) if len(minWeight) > 0 else 1)
	#maxWeight = raw_input("Maximum weight [1] ")
	#maxWeight = (float(maxWeight) if len(maxWeight) > 0 else 1)

	#prefix = raw_input("Output prefix: ")
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
	#print(str(1 + exp((th-x) * steepness)))
	return float(1)/(1 + exp((th-x) * steepness))

def probFloor(a, prob):
	if a > (1-prob):
		return 1
	return 0

def simulate():
	# yes lotta hardcoding, but hopefully there's just a better way to take arguments like this. TODO
	prefix = raw_input("Input prefix: ")
	maxIdx = raw_input("Highest folder index: [0] ")
	maxIdx = (int(maxIdx) if len(maxIdx) > 0 else 0)
	iterations = raw_input("Number of iterations: [20000] ")
	iterations = (int(iterations) if len(iterations) > 0 else 20000)

	spikeChance = raw_input("Spike chance: [.1]")
	spikeChance = (float(spikeChance) if len(spikeChance) > 0 else .1)

	initSpikeChance = raw_input("Init spike chance: [.5] ")
	initSpikeChance = (float(initSpikeChance) if len(initSpikeChance) > 0 else .5)

	threshold = raw_input("Threshold: [.5]")
	threshold = (float(threshold) if len(threshold) > 0 else .5)

	pFloor = np.vectorize(probFloor)
	logistic = np.vectorize(logisticf)


	for i in range(0,maxIdx+1):
		w = np.matrix(np.genfromtxt(prefix + "/" + str(i) + "/w.csv", delimiter=','))
		#s = np.matrix(pFloor(np.random.rand(len(w), 1), initSpikeChance))
		s = np.matrix(pFloor(np.random.rand(len(w), 1), initSpikeChance))
		out = s
		print(w)
		for j in range(0, iterations):
			print(s)
			s = w * s
			s = s + np.matrix(pFloor(np.random.rand(len(w), 1), spikeChance)) # disabling random spiking for testing purposes
			s = logistic(s, threshold)
			out = np.append(out, s, 1)
		np.savetxt(prefix + "/" + str(i) + "/data.csv", out, delimiter=',')
	# TODO: noisify



def main():
	parser = argparse.ArgumentParser(prog="simple.py", usage="%(prog)s <generate|simulate|graph> [-h] [options]", description="Generate, simulate, and visualize neural models.")
	parser.add_argument("--prefix", "-P", default="unset", help="For generate and simulate, prefix of master folder. For graph, folder containing weighted matrix.")
	if "generate" in sys.argv:
		print("Parsing for generate")
		parser.add_argument("generate", nargs='+') # catch this
		parser.add_argument("--networks", "-nets", type=int, help="Number of neural nets to generate", default=1)
		parser.add_argument("--neurons", "-neur", type=int, help="Number of neurons to generate", default=20)
		parser.add_argument("--cycles", "-c", type=float, help="Number of cycles. If c > 1, c cycles will be introduced. if 0 <= c < 1, treated as percentage of neuron input", default=.25)
		parser.add_argument("--verbose", "-V", action="store_true", help="Verbose flag", default=False)
		parser.add_argument("--minweight", "-mw", type=float, help="Minimum connection weight", default=1)
		parser.add_argument("--maxweight", "-Mw", type=float, help="Maximum connection weight", default=1)
		args = parser.parse_args()
		if len(args.generate) < 2:
			args.prefix = raw_input("Prefix: ")
			if len(args.prefix) == 0: exit()
		else:
			args.prefix = args.generate[1] # sketch
		generate(args.networks, args.neurons, args.cycles, args.verbose, args.minweight, args.maxweight, args.prefix)
	elif "simulate" in sys.argv:
		print("Parsing for simulate")
	elif "graph" in sys.argv:
		print("Parsing for graph")
	else:
		print("too far")

	parser.add_argument("action", nargs='?',  help="[generate|simulate|graph]")

	args = parser.parse_args()

	# switching to argparse
	#if len(sys.argv > 1):
		#if str(sys.argv[1]) == "generate":
			# parse for generate
		#elif str(sys.argv[1]) == "simulate":
			# parse for simulate
		#elif str(sys.argv[1]) == "graph":
	#try:
		#opts, args = getopt.getopt(sys.argv[1:])

	if len(sys.argv) > 1:
		if str(sys.argv[1]) == "generate":
			generate()
		elif str(sys.argv[1]) == "simulate":
			simulate()
			#TODO
		elif sys.argv[1] == "graph":
			if not len(sys.argv) > 2:
				prefix = raw_input("Enter folder name (assumes network is w.csv): ")
			else:
				prefix = sys.argv[2]
			showEdges = raw_input("Label edges? [N/y]" ).lower() == 'y'
			w = np.genfromtxt(prefix + "/w.csv", delimiter=',')
			drawNx(genNx(w), drawEdgeLabels=showEdges)
		else:
			print("Syntax: " + sys.argv[0] + " [" + options + "]")
		# TODO
		exit()
	else:
		print("Syntax: " + sys.argv[0] + " [" + options + "]")



if __name__ == "__main__":
	#main()
	#prefix = "data1"

	#for i in range(0, 1000):
	#	w = randWeightMatrix(6,True,makeTris=False, minWeight=.1, maxWeight=.7)
	#	np.savetxt(w, prefix + "/" + str(i) + "/w.csv", delimiter=',')

	#np.savetxt("out.csv", w, delimiter=',')
	#writeMatrix(w, "test1/0")
	#simulate()
	w = np.genfromtxt("data1/0/w.csv", delimiter=',')
	drawNx(genNx(w), drawEdgeLabels=False)
	#makeNest(w)
