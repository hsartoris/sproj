import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("data1/0/total.csv")

td = plt.figure().gca(projection="3d")
td.scatter(df.index)
plt.show()
