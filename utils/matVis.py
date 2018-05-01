#!/usr/bin/python3
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from simpleDataGen import tenNeurNet
mat = tenNeurNet()[0]

fig, ax = plt.subplots()
ax.invert_yaxis()
ax.xaxis.tick_top()

ax.imshow(mat, interpolation='none', cmap='gray_r')
plt.xticks(np.arange(0, mat.shape[0]))
plt.yticks(np.arange(0, mat.shape[1]))
plt.tick_params(axis='both', which='both', top='off', left='off')
plt.show()
