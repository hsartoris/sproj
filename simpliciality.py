from math import pow, log
counts = [0, 0, 17361, 23029, 13452, 4995, 1298, 171, 1]
#counts = [0, 0, 8178, 1620, 28] 
weights = [0,0]
maxPlex = 8.0

simpliciality = 0

for o in range(2, len(counts)):
	weights.append(pow(counts[o] * pow(o/(maxPlex-1), 25), 1))
	simpliciality += weights[len(weights)-1]
print(weights)
print(simpliciality)
