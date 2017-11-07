"""
@author Hayden Sartoris
Intended as a toolkit of sorts for playing with graphs.


"""
import scipy.integrate as integrate
import numpy as np

verbose = True

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

#class SimplicialComplex:
	#def __init__(self

class Simplex:
	def __init__(self, n, weight=1, rules=None):
		# weights
		if not rules is None:
			self.rules = rules
		self.weight=weight
		self.n = n
		self.blueprint = np.triu(np.zeros(shape=(n,n))+1, 1)
				
	def save(self, filename, pad=False):
		if pad:
			temp = np.zeros(shape=(len(self.blueprint)+2, len(self.blueprint)+2))
			self.map(temp, range(1,len(self.blueprint)+1))
			np.savetxt(filename, temp, delimiter=',', fmt='%i')
		else: 
			np.savetxt(filename, self.blueprint, delimiter=',', fmt='%i')
	

	def map(self, w, rules=None, weight=0):
		if rules is None: rules = self.rules
		else: self.rules = rules
		if weight == 0: weight = self.weight
		# syntax: pass in matrix to map Simplex into
		# rules[0] ::> index to map blueprint[0] onto
		if not len(rules) ==  len(self.blueprint):
			raise Exception("Bad rules matrix length")

		for i in range(len(self.blueprint)):
			for j in range(len(self.blueprint)):
				if self.blueprint[i][j] > 0:
					# connection
					if not w[rules[i]][rules[j]] > 0:
						w[rules[i]][rules[j]] = weight
	def rank(self):
		return np.linalg.matrix_rank(np.matrix(self.blueprint))

	def face(self, n):
		# returns random RULESET of dimension n
		temp = []
		while len(temp) < n:
			i = np.random.randint(n)
			if not i in temp: temp.append(i)
		list.sort(temp)
		# this means nodes will be in order
		for i in range(len(temp)): temp[i] = self.rules[i]
		return temp
	
	def dim(s):
		return s.n - 1

class SimplicialComplex:
	def __init__(self, n):
		self.simplices=[]
		self.n=n
		self.blueprint = np.zeros(shape=(n,n))
	def addSimplex(self, sim):
		if len(self.simplices) == 0:
			rules = []
			while len(rules) < sim.n:
				r = np.random.randint(len(self.blueprint))
				if not r in rules:
					rules.append(r)
		else:
			# pick random simplex to append to
			target = np.random.randint(len(self.simplices))
			if verbose: print("Chose simplex " + str(target) + " of dimension " + str(Simplex.dim(self.simplices[target])))
			# note that dim here is actually order, dim is usually order-1, oops
			faceDim = np.random.randint(1, min(sim.n, self.simplices[target].n))
			rules = self.simplices[target].face(faceDim)
			if verbose: print("Chose face of order " + str(faceDim) + "; nodes are " + str(rules))
			for i in range(sim.n - faceDim):
				idx = -1
				while idx < 0:
					# find unused node
					idx = np.random.randint(len(self.blueprint))
					if idx in rules or np.sum(self.blueprint[:,idx]) > 0 or np.sum(self.blueprint[idx,:]) > 0:
						idx = -1
				# could print this but then it wouldn't be one line lol
				rules.insert(np.random.randint(len(rules)+1), idx)
			# rules should now be complete
		sim.map(self.blueprint, rules)
		self.simplices.append(sim)

	def __str__(self):
		temp = "Simplicial complex of " + str(self.n) + " nodes (wrong number though lol)\nContains the following simplices:\n"
		for i in range(len(self.simplices)):
			temp += "Simplex " + str(i) + " of dimension " + str(Simplex.dim(self.simplices[i])) + " with nodes " + str(self.simplices[i].rules) 
			temp += "\n"
		return temp

	def save(self, filename, pad=False):
		if pad:
			temp = np.zeros(shape=(len(self.blueprint)+2, len(self.blueprint)+2))
			self.map(temp, range(1,len(self.blueprint)+1))
			np.savetxt(filename, temp, delimiter=',', fmt='%i')
		else: 
			np.savetxt(filename, self.blueprint, delimiter=',', fmt='%i')


				


def linkedSimplices():
	s1 = Simplex(4)
	s2 = Simplex(4)
	faceDim = np.random.randint(min(Simplex.dim(s1), Simplex.dim(s2)))
	#nodes = 
	
def simplicialGraph():
	# 1. choose desired face size
	# 2. choose appropriate number of nodes
	# 3. validate nodes: no cycles; not a simplex; etc
	# 4. construct simplex with additional nodes as necessary
	print("screw off") 	
