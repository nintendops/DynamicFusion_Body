import math
import numpy as np
from numpy import linalg as la

'''
This script contains the utility functions for certain mathematical computations 
'''

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0


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

    result = []
    result_idx = []
    
    while candidates.size > 0:
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

    
'''
Transform a 3D point directly with a dual quaternion. 
A 3-vector (x,y,z) can be expresed by vq = 1 + epsilon(xi + yj + zk), or [1,0,0,0,0,x,y,z]
The transformation of it by a dq is given by dq*v*(quaternion and dual conjugate of dq)
'''
def dqb_warp(dq, pos):
    vq = np.array([1,0,0,0,0,pos[0],pos[1],pos[2]], dtype=np.float32)
    dqv = dual_quaternion_multiply(dq,vq)
    dqvdqc = dual_quaternion_multiply(dqv,dual_quaternion_conjugate(dq)) 
    return dqvdqc[-3:]

def dqb_warp_normal(dq,pos):
    rq = np.append(dq[:4],[0,0,0,0])
    return dqb_warp(rq,pos)

# basis of the 8 vector dual quaternion is (1, i, j, k, e, ei, ej, ek)
def SE3TDQ(M):
    R, t = decompose_se3(M)
    q = quaternion_from_matrix(M)
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


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.

    >>> q = quaternion_multiply([4, 1, -2, 3], [8, -5, 6, 7])
    >>> np.allclose(q, [28, -44, -14, 48])
    True

    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([
        -x1*x0 - y1*y0 - z1*z0 + w1*w0,
        x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
        x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)

'''
Let q = qr + qd*eps
q1*q2 = qr1*qr2 + (qr1*qd2 + qd1*qr2)epsilon
'''
def dual_quaternion_multiply(q1,q2):
    qr1 = q1[:4]
    qd1 = q1[4:]
    qr2 = q2[:4]
    qd2 = q2[4:]
    qr = quaternion_multiply(qr1,qr2)
    qd = quaternion_multiply(qr1,qd2) + quaternion_multiply(qd1,qr2)
    return np.append(qr,qd)


def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_conjugate(q0)
    >>> q1[0] == q0[0] and all(q1[1:] == -q0[1:])
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q


def dual_quaternion_conjugate(dquaternion):
    dq = np.array(dquaternion, dtype=np.float64, copy=True)
    np.negative(dq[4:],dq[4:])
    np.negative(dq[5:],dq[5:])
    np.negative(dq[1:4],dq[1:4])
    return dq


'''
lw: 3x4 camera extrinsic
K: 3x3 camera intrinsic 
pos: (x,y,z)
'''
def project_to_pixel(K, pos, lw = None):
    p = []
    if lw is None:
        if len(pos) == 4:
            pos = pos[:-1]
        p = np.matmul(K,pos)
        if p[2] == 0:
            return (None, None)
        return (p[0]/p[2], p[1]/p[2])
    else:
        if len(pos) == 3:
            pos = np.append(pos,1)
        p = np.matmul(K, np.matmul(lw,pos))
        if p[2] == 0:
            return (None, None)
        else:
            return (p[0]/p[2],p[1]/p[2])

def read_proj_matrix(fpath):
    f = open(fpath,'r')
    arr = []
    for line in f:
        arr.append(line[:-1].split(' '))
    return np.array(arr,dtype='float')

# find the inverse of a 3x4 rigid transformation matrix
def inverse_rigid_matrix(A):
    R,t = decompose_se3(A)
    R_inv = la.inv(R)
    t_inv = np.matmul(R_inv,t) * -1
    M = np.zeros((3,4))
    M[0] = np.append(R_inv[0],t_inv[0])
    M[1] = np.append(R_inv[1],t_inv[1])
    M[2] = np.append(R_inv[2],t_inv[2])
    return M
