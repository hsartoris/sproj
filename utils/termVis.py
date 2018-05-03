#!/usr/bin/python3
import numpy as np
import sys

def termVis(struct):
    if len(struct.shape) == 1:
        struct = np.expand_dims(struct, 0)
    w = struct.shape[1]
    print(" " + "-"*(w+2))
    for i in range(struct.shape[0]):
        print(str(i) + "|" + "".join([("#" if x > 0 else " ") for x in struct[i]]) + 
    "|")
    print(" " + "-"*(w+2))
'''
if not len(sys.argv) == 2:
    print("Usage: termVis <struct>")
    exit()

struct = np.loadtxt(sys.argv[1], delimiter=',')
'''
