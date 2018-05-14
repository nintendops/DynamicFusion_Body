import numpy as np
from numpy import linalg as la
from skimage import measure
from skimage.draw import ellipsoid
from util import *
from fusion import Fusion
from transformation import random_rotation_matrix


# helper functions
def cal_dist(a,b):
    return la.norm(a-b)

def weird_function(a, b=None):
    if b is not None:
        return (1,2)
    else:
        return 1


    
if __name__ == "__main__":
    # Generate a level set about zero of two identical ellipsoids in 3D
    ellip_base = ellipsoid(6, 10, 16, levelset=True)
    ellip_double = np.concatenate((ellip_base[:-1, ...], ellip_base[2:, ...]), axis=0)
    print(ellip_double.max())
    print(ellip_double.min())
    fus = Fusion(ellip_double, ellip_double.min())
    
    # Testing DQ functions
    R = random_rotation_matrix()[np.ix_([0,1,2],[0,1,2])]
    t = np.array([0.1,0.4,0.2])
    M = compose_se3(R,t)
    print("Input matrix")
    print(M)
    print('converted dq')
    q = SE3TDQ(M)
    print(q)
    print('converted back to matrix')
    print(DQTSE3(q))

    # Testing weird stuff
    print(weird_function(1,1))
    print(weird_function(1))
