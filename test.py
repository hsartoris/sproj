from scripts.GraphKit import *
import numpy as np

simplices = []
total = 0
for i in range(10):
	simplices.append(Simplex(np.random.randint(2,9)))
	total += simplices[i].n
	print("Generated simplex of dimension " + str(Simplex.dim(simplices[i])))
sc = SimplicialComplex(2*total)
for i in range(len(simplices)):
	sc.addSimplex(simplices[i])
print(sc)
sc.save("neurotop/complete.csv")
