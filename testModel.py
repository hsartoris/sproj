import tensorflow as tf
import numpy as np
from model.model2 import Model
b = 3
d = 4
n = 5

_data = tf.placeholder(tf.float32, [b,n])
_labels = tf.placeholder(tf.float32, [1, n*n])

m = Model(b, d, n, _data, _labels, 1)
