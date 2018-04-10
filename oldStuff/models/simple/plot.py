import matplotlib
#matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2 or ".csv" not in sys.argv[1]:
	print("Requires csv to be plotted as argument")
	exit()

data = np.genfromtxt(sys.argv[1], delimiter=',')

for i in range(0, len(data)):
	plt.plot(data[i])
plt.savefig(sys.argv[1][:-4] + ".png", dpi=200)
