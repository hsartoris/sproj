from scripts.graphGen import *

#p = ProbDist(lambda x: x**(-3))
p = ProbDist(PowerLaw(2))
for i in range(100):
	print(p.get())
