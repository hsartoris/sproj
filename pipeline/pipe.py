#!/usr/bin/env python
#. stolen from Derek
import nest
import pylab
import nest.topology as topp
import nest.raster_plot as raster
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys

#DEFINE FUNCTIONS
'''
Take a NEST network as a parameter and generates a networkX graph
'''
def drawNetwork(pop):
	pop_connect_dict = nest.GetConnections(pop)
	G = nx.DiGraph()
	for i in pop:
		G.add_node(i)
	netXEdges = []
	for j in pop_connect_dict:
		x = j[0]
		y = j[1]
		netXEdges.append((x,y))
	G.add_edges_from(netXEdges)
	nx.draw(G, with_labels=True)
	return G

'''
Reads from a csv file for storing weights, connects corresponding
nest neurons, outputs a np matrix
'''
def readAndConnect(file, population):
	matrix = np.loadtxt(open(file, "rb"), delimiter=",")
	row_pos = 0
	#adjMatrix = []
#	for i_neuron_array in matrix:
#		col_pos = 0
#		for j_connection in i_neuron_array:
#			if j_connection == 1.0:
#				#adjMatrix.append([row_pos,col_pos])
#				nest.Connect([population[row_pos]],[population[col_pos]])
#			col_pos = col_pos + 1		
#		row_pos = row_pos +1
	for row_pos in range(len(matrix)):
		for col_pos in range(len(matrix[row_pos])):
			if matrix[row_pos][col_pos] == 1.0:
				nest.Connect([population[row_pos]],[population[col_pos]])
	return matrix

def readAndCreate(file):
	'''Reads from a csv file for storing weights, creates the population indicated,
	returns population'''
	matrix = np.loadtxt(open(file, "rb"), delimiter=",")

	#Create the neurons for the network
	pop = nest.Create("izhikevich", len(matrix))
	ratio = .2 # inhib to excite
	#Connect the neurons
	for row_pos in range(len(matrix)):
		for col_pos in range(len(matrix[row_pos])):
			if matrix[row_pos][col_pos] == 1.0:
				if np.random.random() <= ratio:
					nest.Connect([pop[row_pos]],[pop[col_pos]],syn_spec = {"model":"stdp_synapse","weight":-1.0})
				else:
					nest.Connect([pop[row_pos]],[pop[col_pos]],syn_spec={"model":"stdp_synapse","weight":1.0})
	return pop, matrix

def create(matrix):
	#Create the neurons for the network
	pop = nest.Create("izhikevich", len(matrix))
	ratio = .2 # inhib to excite
	#Connect the neurons
	for row_pos in range(len(matrix)):
		for col_pos in range(len(matrix[row_pos])):
			if matrix[row_pos][col_pos] > 0:
				if np.random.random() <= ratio:
					nest.Connect([pop[row_pos]],[pop[col_pos]],syn_spec = {"model":"stdp_synapse","weight":-1.0})
				else:
					nest.Connect([pop[row_pos]],[pop[col_pos]],syn_spec={"model":"stdp_synapse","weight":1.0})
	return pop

def rasterGenerator(pop):
	spikes = nest.Create("spike_detector",len(pop))
	nest.Connect(pop, spikes)
	plot = nest.raster_plot.from_device(spikes, hist=True)
	return plot


######################################################################################

#######
#
# 
#####
#   
#  
#

#########################################
######################################################################################


def spikeTimeMatrix(spikes, num_neurons, timesteps):
	n = nest.GetStatus(spikes, "events")[0]
	output = np.matrix(np.zeros((num_neurons,timesteps)))
	for i in range(len(n['times'])):
		output[n['senders'][i]-1,int(round(n['times'][i]))] = 1
	return output

if __name__ == "__main__":
	precise = True
	#SET PARAMETERS
	numNeurons = 5
	poisson_rate = 3000.0
	if len(sys.argv) < 3:
		print("Bad arguments. no info for you")
		exit()
	neuronPop, matrix = readAndCreate(sys.argv[1])
	simtime = float(sys.argv[2])

	#CREATE NODES
	noise = nest.Create("poisson_generator",1,{'rate':poisson_rate})
	spikes = nest.Create("spike_detector", 1)
	
	Ex = 1
	d = 1.0
	wEx = 1.0
	wIn = -1.0
	
	#SPECIFY CONNECTION DICTIONARIES
	conn_dict = {"rule": "fixed_indegree", "indegree": Ex,
				"autapses":False,"multapses":False} #connection dictionary
	syn_dict_ex = {"delay": d, "weight": wEx}
	syn_dict_in = {"delay": d, "weight": wIn}
	
	#SPECIFY CONNECTIONS
	nest.Connect(noise, neuronPop,syn_spec = syn_dict_ex)
	nest.Connect(neuronPop,spikes)
	
	nest.Simulate(simtime)
	
#	drawNetwork(neuronPop)
#	plot = nest.raster_plot.from_device(spikes, hist=True)
#	plt.show()
	split = sys.argv[1].split("/")
	name = split[len(split)-1]
	'''
	The exact neuron spikes and corresponding timings can be obtained by viewing the events
	dictionary of GetStatus(spikesEx, "events")
	'''
	#print nest.GetStatus(spikes, "events")
	#print(dir(nest.GetStatus(spikes, "events")))
#	np.savetxt(sys.argv[3] + "spikes/" + name, spikeTimeMatrix(spikes, len(matrix), int(simtime)), delimiter=',', fmt='%i')
	n = nest.GetStatus(spikes, "events")[0]
	temp = np.array([n['senders'], n['times']])
	np.savetxt(sys.argv[3] + "spikes/" + name, temp, delimiter=',', fmt=('%f' if precise else '%i'))
	print(temp)

def pipe(matrix, simtime):
	reload(nest)
	#SET PARAMETERS
	poisson_rate = 3000.0
	neuronPop = create(matrix)

	#CREATE NODES
	noise = nest.Create("poisson_generator",1,{'rate':poisson_rate})
	spikes = nest.Create("spike_detector", 1)
	
	Ex = 1
	d = 1.0
	wEx = 1.0
	wIn = -1.0
	
	#SPECIFY CONNECTION DICTIONARIES
	conn_dict = {"rule": "fixed_indegree", "indegree": Ex,
				"autapses":False,"multapses":False} #connection dictionary
	syn_dict_ex = {"delay": d, "weight": wEx}
	syn_dict_in = {"delay": d, "weight": wIn}
	
	#SPECIFY CONNECTIONS
	nest.Connect(noise, neuronPop,syn_spec = syn_dict_ex)
	nest.Connect(neuronPop,spikes)
	
	nest.Simulate(float(simtime))
	
#	drawNetwork(neuronPop)
#	plot = nest.raster_plot.from_device(spikes, hist=True)
#	plt.show()
	'''
	The exact neuron spikes and corresponding timings can be obtained by viewing the events
	dictionary of GetStatus(spikesEx, "events")
	'''
	#print nest.GetStatus(spikes, "events")
	#print(dir(nest.GetStatus(spikes, "events")))
	#np.savetxt("spikes/"+str(index) + ".csv", spikeTimeMatrix(spikes, len(matrix), int(simtime)), delimiter=',', fmt='%i')
	return spikeTimeMatrix(spikes, len(matrix), simtime)

