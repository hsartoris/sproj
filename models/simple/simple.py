import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pylab
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # lol
plt.axis('off')

def drawNx(g,drawEdgeLabels=False):
	# draw the graph
	edgeLabels = dict([((u,v),d['weight'])
		for u,v,d in g.edges(data=True)])
	nodeLabels = {node:node for node in g.nodes()}
	pos = nx.spring_layout(g)
	if drawEdgeLabels:
		nx.draw_networkx_edge_labels(g, pos, edge_labels=edgeLabels)
		nx.draw_networkx(g, pos, labels=nodeLabels)
	else :
		nx.draw_networkx(g, labels=nodeLabels)
	pylab.show()


def randWeightMatrix(numNeurons, cycles, isVerbose, minWeight=1, maxWeight=1, genGraph=False):
	# returns a 2d array of weights defining a fully connected, somewhat reasonable network
	out = np.arange(numNeurons)
	mem = np.array([])
	w = np.zeros(shape=(numNeurons, numNeurons))
	node1 = random.randint(0, len(out) - 1)
	out = np.delete(out, node1)
	mem = np.append(mem, node1) # for the first node the index and value are the same
	if isVerbose: print("First node: " + str(node1))
	if genGraph: g = nx.DiGraph(); g.add_node(str(node1))
	while len(out) > 0:
		# choose a node in the network & a node outside the network
		# add a directed edge from the latter to the former & add note that the other is now in the network
		memberNode = mem[random.randint(0, len(mem) - 1)] # actual index of the node in the weight matrix
		newMember = random.randint(0, len(out) - 1) # index of actual index 
		if isVerbose: print("Adding node " + str(out[newMember]) + " connected to " + str(int(memberNode)))
		# w[memberNode] is the input vector for memberNode
		w[int(memberNode)][int(out[newMember])] = random.uniform(minWeight, maxWeight)
		if genGraph: g.add_edges_from([(str(out[newMember]), str(int(memberNode)))], weight=w[int(memberNode)][int(out[newMember])])
		mem = np.append(mem, out[newMember])
		out = np.delete(out, newMember)
	
	if isVerbose: 
		print("Pre-cycle matrix: ")
		print(w) 
	
	# insert the random cycles
	while cycles > 0:
		node1 = node2 =  mem[random.randint(0, len(mem) - 1)]
		while node1 == node2:
			# make sure we have different nodes
			node2 = mem[random.randint(0, len(mem) - 1)]
		w[int(node1)][int(node2)] = random.uniform(minWeight, maxWeight)
		if genGraph: g.add_edges_from([(str(int(node2)), str(int(node1)))], weight=w[int(node1)][int(node2)])
		if isVerbose: print("Adding connection from node " + str(int(node2)) + " to node " + str(int(node1)))
		cycles = cycles - 1
	if isVerbose:
		print("Post-cycle matrix:")
		print(w)
	if genGraph: return w,g
	return w


w,g = randWeightMatrix(20, 5, False, genGraph=True)
#graph = nx.DiGraph()
#graph.add_nodes_from(range(0, len(w))
drawNx(g, True)
w = randWeightMatrix(20, 5, False)

#print(w)
