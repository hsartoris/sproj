"""
@author Hayden Sartoris
Intended as a toolkit of sorts for playing with graphs.

28 Oct 2017: provides SingleVarFunction, PowerLaw(SingleVarFunction), and
	ProbDist, in support of genGraph::Matrix for known distribution

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

class GraphGen:
	def __init__(self, n):
		# what am I doing
		self.n = n
		
