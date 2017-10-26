from scripts.graphGen import ProbDist

p = ProbDist(lambda x: x**(-3))
print(p.get())
