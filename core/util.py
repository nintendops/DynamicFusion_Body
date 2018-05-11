import numpy as np
from numpy import linalg as la


'''
This script contains the utility functions for certain mathematical computations 
'''


def compose_se3(R,t):
    M = numpy.identity(4)
    M[0] = np.append(R[0],t[0])
    M[1] = np.append(R[1],t[1])
    M[2] = np.append(R[2],t[2])
    return M

def decompose_se3(M):
    pass

# Radius-based spatial subsampling
def uniform_sample(arr,radius):
    candidates = arr.copy()
    result = []
    locations = []
    pointer = 0
    
    while candidates.size > 0:
        remove = []
        rows = candidates.shape[0]
        sample = candidates[0]
        index = np.arange(rows).reshape(-1,1)
        dists = np.column_stack((index,candidates))
        result.append(sample)
        locations.append(pointer)
        for row in dists:
            if la.norm(row[1:] - sample) < radius:
                remove.append(int(row[0]))
                pointer = pointer + 1
        candidates = np.delete(candidates, remove, axis=0)
    return np.array(result), np.array(locations)


def huber_loss(x,c):
    if abs(x) <= c:
        return 0.5 * (x**2)
    else:
        return c * (abs(x) - 0.5*c)
    
def tukey_biweight_loss(x,c):
    if abs(x) > c:
        return 0
    else:
        return x * (1 - (x/c)**2)**2
