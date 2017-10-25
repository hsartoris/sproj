import scipy.integrate as integrate
import numpy as np

class ProbDist:
	lookup = np.array([])
	cutoff = 1000000000
	def __init__(self, function, normalization_factor = 1):
		# ProbDist(lambda x: <func>) [ I hope ]
		self.f = np.vectorize(function) 
		self.nf = normalization_factor
	
	def get():
		# find out how normalization actually works lol
		i = 0
		acc = 0
		r = np.random.rand()
		while (True):
			if (i >= len(lookup)): lookup = np.append([nf * integrate(function, i, i+1)])
			acc += lookup[i]
			if acc > r: return i
		# ideally this terminates
			if (i == cutoff): return i
