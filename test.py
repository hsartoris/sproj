from scripts.GraphKit import *
import numpy as np

simplices = []
total = 0
for i in range(5):
	simplices.append(Simplex(np.random.randint(2,9)))
	total += simplices[i].n
	print("Generated simplex of dimension " + str(Simplex.dim(simplices[i])))
sc = SimplicialComplex()
for i in range(len(simplices)):
	sc.addSimplex(simplices[i])
print(sc)
sc.save("neurotop/runMe.csv")

