"""
@author Hayden Sartoris
Intended as a toolkit of sorts for playing with graphs.


"""
import scipy.integrate as integrate
import numpy as np

class SingleVarFunction(object):
	# abstract class to represent functions
	def f(self, x):
		raise NotImplementedError( "Not implemented on this class" )

class PowerLaw(SingleVarFunction):
	def __init__(self, gamma, k_min = 1):
		if (gamma < 0): raise ValueError("Gamma must be positive")
		if not (k_min > 0): raise ValueError("k_min must be greater than 0")
		self.g = gamma
		self.C = (gamma - 1) * (k_min ** (gamma - 1 ))
		self.km = k_min

	def f(self, k):
		if not (k > self.km): raise ValueError("k must be greater than " + str(self.km))
		return self.C * (k ** (-(self.g)))

class ProbDist:
	# not sure how to handle 0 probability
	# update: fuck it
	lookup = np.array([])
	cutoff = 1000000000
	k_min = .1
	def __init__(self, function, normalization_factor = 1):
		# ProbDist(lambda x: <func>)
		# Takes a probability function p(k) that predicts how probable
		# 	continuous k is
		#self.f = np.vectorize(function)
		if isinstance(function, SingleVarFunction):
			self.f = function.f
			self.k_min = function.km
		else:
			self.f = function
		self.nf = normalization_factor
		self.startIdx = int(self.k_min)

	def get(self):
		i = self.startIdx
		acc = 0
		r = np.random.rand()
		while (True):
			if (i-self.startIdx >= len(self.lookup)): self.lookup = np.append(self.lookup, [integrate.quad(self.f, (i if (i > 0) else self.k_min), i+1, epsabs=0)[0] / self.nf])
			acc += self.lookup[i-self.startIdx]
			if acc > r: return i
			# ideally this terminates
			if (i == self.cutoff): return i
			i += 1

class Simplex:
	def __init__(self, n):
		self.n = n
		self.blueprint = np.zeros(shape=(n,n))
		self.source = 0
		self.sink = n-1
		for i in range(n):
			for j in range(n):
				if i == j or self.blueprint[i][j] > 0 or self.blueprint[j][i] > 0:
					# no loops; don't overwrite connections
					continue
				if i == self.source or j == self.sink:
					# always connect from source or to sink
					self.blueprint[i][j] = 1
				elif i == self.sink or j == self.source:
					self.blueprint[j][i] = 1
				elif np.random.rand() > .5:
					# neither node source or sink
					self.blueprint[i][j] = 1
				else:
					self.blueprint[j][i] = 1

	def save(self, filename):
		np.savetxt(filename, self.blueprint, delimiter=',')
