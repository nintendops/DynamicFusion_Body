import numpy as np
from numpy import linalg as la
from transformation import quaternion_from_matrix, quaternion_matrix, quaternion_multiply, quaternion_conjugate

'''
This script contains the utility functions for certain mathematical computations 
'''

# Compose a 4x4 SE3 matrix
def compose_se3(R,t):
    M = np.identity(4)
    M[0] = np.append(R[0],t[0])
    M[1] = np.append(R[1],t[1])
    M[2] = np.append(R[2],t[2])
    return M

# Decompose M into R and t
def decompose_se3(M):
    return M[np.ix_([0,1,2],[0,1,2])], M[np.ix_([0,1,2],[3])]
    

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

# basis of the 8 vector dual quaternion is (1, i, j, k, e, ei, ej, ek)
def SE3TDQ(M):
    R, t = decompose_se3(M)
    q = quaternion_from_matrix(R)
    q = q / la.norm(q)
    qe = 0.5 * quaternion_multiply([0,t[0],t[1],t[2]],q)
    return np.append(q,qe)

def DQTSE3(q):
    R = quaternion_matrix(q[:4])[np.ix_([0,1,2],[0,1,2])]
    t = quaternion_multiply(2 * q[4:], quaternion_conjugate(q[:4]))
    return compose_se3(R,t[1:])

# Trilinear interpolation of signed distance. Return None if pos is out of the volume.
def interpolate_tsdf(pos, tsdf):
    pass
    
def cal_dist(a,b):
    return la.norm(a-b)
