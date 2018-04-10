import numpy as np

class Simple(object):
	## This network is a three-neuron model, with neuron 0 connected
	#  to both neurons 1 and 2.
	## This network is entirely matrix-based.
	
	def __init__(self, w1=.8, w2=.2):
		self.w = np.matrix(np.zeros((3,3)))
		self.w[1,0] = w1
		self.w[2,0] = w2
		self.s = np.matrix(np.zeros((3,1)))
	
	def run(self, timesteps, chance=.2):
		# returns matrix where first axis is timestep, ie (10x3)
		out = np.matrix(np.zeros((timesteps, 3)))
		for i in range(timesteps):
			self.s = self.w * self.s # advance a step, then add random spiking
			for j in range(len(self.s)):
				if np.random.random() < chance:
					self.s[j] = 1
			out[i] = self.s.transpose()
		return out

def correlation(a1, a2, delay):
	# from a1 to a2
	# equivalent for delay=0, but not above
	t1 = (a1[:-delay] if delay > 0 else a1)
	t2 = (a2[delay:] if delay > 0 else a2)
	t3 = np.multiply(t1,t2)
	return np.sum(t3)/len(t3)
	
def label(data, depth):
	# takes data from run() and creates correlational labels
	temp = np.array(data.transpose())
	timesteps = len(temp[0])
	n = len(temp)
	depth = min(depth, timesteps)
	out = np.matrix(np.zeros((n*n, depth)))
	for i,a1 in enumerate(temp):
		for j,a0 in enumerate(temp):
			for d in range(depth):
				out[(i*n)+j, d] = correlation(a0, a1, d)
	return out.transpose()

#s = Simple()
#out = s.run(100, .1)
#print(out)
