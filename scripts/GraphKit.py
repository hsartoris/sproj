"""
@author Hayden Sartoris
Intended as a toolkit of sorts for playing with graphs.


"""
import scipy.integrate as integrate
import numpy as np
import scipy.sparse as sparse

verbose = False

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
class diGraph(object):
    def __init__(self, n):
        self.n = n
        self.blueprint = np.matrix(np.zeros(shape=(n,n)))
        self.rules = None
        self.weight = 1

    def save(self, filename, pad=0):
        if pad > 0:
            temp = np.matrix(np.zeros(shape=(len(self.blueprint)+(2*pad), len(self.blueprint)+(2*pad))))
            self.map(temp, range(pad, len(self.blueprint) + pad))
        else:
            temp = self.blueprint
        np.savetxt(filename, temp, delimiter=',', fmt='%i')

    def map(self, w, rules=None, weight=0):
        # in-place map of diGraph object onto existing matrix
        if rules is None and not self.rules is None: rules = self.rules
        else: self.rules = rules
        if rules is None: raise Exception("No rules set and none provided")
        if weight == 0: weight = self.weight
        if not len(rules) == len(self.blueprint): raise Exception("Bad rules array length")

        for i in range(len(rules)):
            for j in range(len(rules)):
                if self.blueprint[i,j] > 0 and not w[rules[i], rules[j]] > 0:
                    if (rules[i] == rules[j]): print(str(i) + "," + str(j))
                    w[rules[i], rules[j]] = weight

class Simplex(diGraph):
    def __init__(self, n, weight=1, rules=None):
        # adheres to stupid simplex terminology; Simplex(8) is a 9-node simplex
        diGraph.__init__(self, n+1)
        if not rules is None:
            self.rules = rules
        self.weight=weight
        self.blueprint = np.matrix(np.triu(np.zeros(shape=(self.n,self.n))+1, 1))


    def rank(self):
        return np.linalg.matrix_rank(self.blueprint)

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
    def dim(self):
        return self.n - 1

class SimplicialComplex(diGraph):
    def __init__(self):
        diGraph.__init__(self,0)
        self.simplices=[]
    def addSimplex(self, sim):
        if len(self.simplices) == 0:
            rules = range(sim.n)
#			rules = []
#			while len(rules) < sim.n:
#				r = np.random.randint(len(self.blueprint))
#				if not r in rules:
#					rules.append(r)
            self.blueprint = np.zeros((sim.n, sim.n))
        else:
            # pick random simplex to append to
            target = np.random.randint(len(self.simplices))
            if verbose: print("Chose simplex " + str(target) + " of dimension " + str(Simplex.dim(self.simplices[target])))
            faceDim = np.random.randint(1, min(sim.n, self.simplices[target].n))
            rules = self.simplices[target].face(faceDim)
            if verbose: print("Chose face of order " + str(faceDim) + "; nodes are " + str(rules))
            if verbose: print("Adding " + str(sim.n - faceDim) + " nodes to existing blueprint of size " + str(len(self.blueprint)))
            # Expand the blueprint to accomodate the new simplex
            overlap = sim.n - faceDim
            temp = np.zeros((len(self.blueprint)+overlap, len(self.blueprint)+overlap))
            temp[:-overlap,:-overlap] = self.blueprint
            self.blueprint = temp
            # add new indices to the ruleset
            for i in range(sim.n - faceDim):
                idx = len(self.blueprint) - (i + 1)
                # could print this but then it wouldn't be one line lol
                rules.insert(np.random.randint(len(rules)+1), idx)
            # rules should now be complete
        sim.map(self.blueprint, rules=rules)
        self.simplices.append(sim)
        self.n = len(self.blueprint)

    def __str__(self):
        temp = "Simplicial complex of " + str(self.n) + " nodes (wrong number though lol)\nContains the following simplices:\n"
        for i in range(len(self.simplices)):
            temp += "Simplex " + str(i) + " of dimension " + str(Simplex.dim(self.simplices[i])) + " with nodes " + str(self.simplices[i].rules) 
            temp += "\n"
        return temp
    def edges(self):
        return np.sum(self.blueprint)

def saveSparse(filename, matrix):
    sp = sparse.csc_matrix(matrix)
    sparse.save_npz(filename, sp)

def loadSparse(filename):
    sp = sparse.load_npz(filename)
    return sp.todense()

def linkedSimplices():
    s1 = Simplex(4)
    s2 = Simplex(4)
    faceDim = np.random.randint(min(Simplex.dim(s1), Simplex.dim(s2)))
    #nodes = 

def spikeTimeMatrix(spikes, numNeurons, timesteps):
    # takes saved matrix where first row is spiking neuron
    # and second row is time of spike
    # note: loses precision
    output = np.matrix(np.zeros((numNeurons, timesteps)))
    for i in range(len(spikes[0])):
        output[int(spikes[0][i]-1), int(round(spikes[1][i]))] = 1
    return output

def spikeTimeArray(spikes, timesteps):
    # takes saved matrix where first row is spiking neuron
    # and second row is time of spike
    # outputs a list of len 10*timesteps
    output = np.zeros((10*timesteps, 1), dtype=int)
    for i in range(len(spikes[0])):
        output[int(spikes[1][i] * 10)] = np.array([int(spikes[0][i])])
    return output

def perturb(matrix, chance=.1):
    # incomplete
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i == j: continue
            if np.random.random() < chance:
                matrix[i,j] = (0 if matrix[i,j] == 1 else 1)
    return matrix

