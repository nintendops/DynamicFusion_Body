import numpy as np
from numpy import linalg as la


'''
This script contains the utility functions for certain mathematical computations 
'''


'''
A SE(3) matrix is a 4x4 matrix composed by a rotation matrix R and a translation vector t
k SE3 transformations can be interpolated with dual-quaternion blending

'''

def compose_se3(R,t):
    pass

def decompose_se3(M):
    pass



# Radius-based spatial subsampling
def uniform_sample(arr,radius):
    candidates = arr.copy()
    result = []
    while candidates.size > 0:
        remove = []
        rows = candidates.shape[0]
        sample = candidates[0]
        index = np.arange(rows).reshape(-1,1)
        dists = np.column_stack((index,candidates))
        result.append(sample)
        for row in dists:
            # print("comparing {0} with {1} and the norm is {2}".format(row[1:], sample, la.norm(row[1:] - sample)))
            if la.norm(row[1:] - sample) < radius:
                remove.append(int(row[0]))
        candidates = np.delete(candidates, remove, axis=0)
    return np.array(result)




