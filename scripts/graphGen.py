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

	def f(self, k):
		if not (k > 0): raise ValueError("k must be greater than 0")
		return self.C * (k ** (-(self.g)))

class ProbDist:
	# not sure how to handle 0 probability
	lookup = np.array([])
	cutoff = 1000000000
	k_min = .1
	def __init__(self, function, normalization_factor = 1):
		# ProbDist(lambda x: <func>) [ I hope ]
		#self.f = np.vectorize(function)
		if isinstance(function, SingleVarFunction):
			self.f = function.f
		else:
			self.f = function
		self.nf = normalization_factor

	def get(self):
		i = 0
		acc = 0
		r = np.random.rand()
		while (True):
			if (i >= len(self.lookup)): self.lookup = np.append(self.lookup, [integrate.quad(self.f, (i if (i > 0) else self.k_min), i+1, epsabs=0) / self.nf])
			acc += self.lookup[i]
			if acc > r: return i
			# ideally this terminates
			if (i == cutoff): return i
