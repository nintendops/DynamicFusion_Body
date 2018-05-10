import numpy as np
from numpy import linalg as la
from skimage import measure
from skimage.draw import ellipsoid
from util import *
from fusion import Fusion

# helper functions
def cal_dist(a,b):
    return la.norm(a-b)


if __name__ == "__main__":
    # Generate a level set about zero of two identical ellipsoids in 3D
    ellip_base = ellipsoid(6, 10, 16, levelset=True)
    ellip_double = np.concatenate((ellip_base[:-1, ...], ellip_base[2:, ...]), axis=0)
    fus = Fusion(ellip_double)

    
    
