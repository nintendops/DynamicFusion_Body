import numpy as np
from numpy import linalg as la
from skimage import measure
from skimage.draw import ellipsoid
from core import *
from core.util import *
from core.transformation import random_rotation_matrix


# helper functions
def cal_dist(a,b):
    return la.norm(a-b)

def weird_function(a, b=None):
    if b is not None:
        return (1,2)
    else:
        return 1

TEST_INPUT = True
TEST_FUSION = True
TEST_UTIL = False
TEST_WEIRD_STUFF = False
    
if __name__ == "__main__":

    ellip_base = ellipsoid(6, 10, 16, levelset=True)
    volume = np.concatenate((ellip_base[:-1, ...], ellip_base[2:, ...]), axis=0)
        
    if TEST_INPUT:
        sdf_filepath = DATA_PATH + '0000.dist'
        b_min, b_max, volume, closest_points = load_sdf(sdf_filepath, verbose=True)
        print("Shape of volume: (%d, %d, %d)" % volume.shape)

    res_x, res_y, res_z = volume.shape
        
    if TEST_FUSION:
        # Generate a level set about zero of two identical ellipsoids in 3D
        fus = Fusion(volume, volume.min(), subsample_rate = 4.0, verbose = True)
        print("Solving for a test iteration")
        fus.solve(fus._vertices)

    if TEST_UTIL:
        # Testing DQ functions
        print('Testing DQ functions')
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

        # Testing interpolation
        print('Testing tsdf trilinear interpolation')
        for i in range(3):
            pos1 = res_x/2 * np.random.rand(3)
            pos2 = res_y/2 * np.random.rand(3)
            pos3 = (10,10,10)
            posb1 = -1 * np.random.rand(3)
            posb2 = np.array(volume.shape) + 1
            print('below should not be None')
            print(interpolate_tsdf(pos1,volume))
            print(interpolate_tsdf(pos2,volume))
            print('ground truth %f' % volume[pos3])
            print(interpolate_tsdf(pos3,volume))
            print('below should be None')
            print(interpolate_tsdf(posb1,volume))
            print(interpolate_tsdf(posb2,volume))
            

    if TEST_WEIRD_STUFF:
        # Testing weird stuff
        print(weird_function(1,1))
        print(weird_function(1))
