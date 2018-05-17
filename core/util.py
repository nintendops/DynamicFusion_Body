import math
import numpy as np
from numpy import linalg as la
from .transformation import quaternion_from_matrix, quaternion_matrix, quaternion_multiply, quaternion_conjugate

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
    candidates = np.array(arr).copy()
    locations = np.arange(len(candidates))
    #for idx in range(len(arr)):
    #    candidates.append((arr[idx],idx))

    result = []
    result_idx = []
    
    while candidates.size > 0:
        #print("current candidate size: %d"%(candidates.size))
        remove = []
        rows = len(candidates)
        sample = candidates[0]
        index = np.arange(rows).reshape(-1,1)
        dists = np.column_stack((index,candidates))
        result.append(sample)
        result_idx.append(locations[0])
        for row in dists:
            if la.norm(row[1:] - sample) < radius:
                remove.append(int(row[0]))
        candidates = np.delete(candidates, remove, axis=0)
        locations = np.delete(locations, remove)
    return np.array(result), np.array(result_idx)


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

def InitialSE3():
    M = np.identity(4)
    M[0,0] = 0.9
    M[1,1] = 0.9
    M[2,2] = 0.9
    M[0,3] = 0.1
    M[1,3] = 0.1
    M[2,3] = 0.1
    return M

# Trilinear interpolation of signed distance. Return None if pos is out of the volume.
def interpolate_tsdf(pos, tsdf):
    if tsdf.ndim != 3:
        raise ValueError('Only 3D numpy array is accepted')
    res_x, res_y, res_z = tsdf.shape

    if min(pos) < 0 or pos[0] > res_x - 1 or pos[1] > res_y - 1 or pos[2] > res_z - 1 :
        return None

    x0 = math.floor(pos[0])
    y0 = math.floor(pos[1])
    z0 = math.floor(pos[2])
    x1 = math.ceil(pos[0])
    y1 = math.ceil(pos[1])
    z1 = math.ceil(pos[2])

    xd = pos[0] - x0
    yd = pos[1] - y0
    zd = pos[2] - z0

    c000 = tsdf[(x0,y0,z0)]
    c100 = tsdf[(x1,y0,z0)]
    c001 = tsdf[(x0,y1,z0)]
    c101 = tsdf[(x1,y1,z0)]
    c010 = tsdf[(x0,y0,z1)]
    c110 = tsdf[(x1,y0,z1)]
    c011 = tsdf[(x0,y1,z1)]
    c111 = tsdf[(x1,y1,z1)]

    c00 = c000 * (1-xd) + c100 * xd
    c01 = c001 * (1-xd) + c101 * xd
    c10 = c010 * (1-xd) + c110 * xd
    c11 = c011 * (1-xd) + c111 * xd

    c0 = c00 * (1-yd) + c10 * yd
    c1 = c01 * (1-yd) + c11 * yd
    return c0 * (1-zd) + c1 * zd
    
def cal_dist(a,b):
    return la.norm(a-b)












