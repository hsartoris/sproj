from tiny import *
import numpy as np
import os

networks = 10000
timesteps = 20
depth = 3

for i in range(networks):
	if not os.path.exists(str(i)):
		os.makedirs(str(i))
	s = Simple(np.random.random(), np.random.random())
	data = s.run(timesteps)
	l = label(data, depth)
	np.savetxt(str(i) + "/data.csv", data, delimiter=',')
	np.savetxt(str(i) + "/label.csv", l, delimiter=',')
	print(i," out of ", networks)

