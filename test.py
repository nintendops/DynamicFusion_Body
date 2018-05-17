import numpy as np
import cProfile
from sympy import *
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
TEST_FUSION_DUMMY = False
TEST_UTIL = False
TEST_WEIRD_STUFF = False


def test_diff(r11,r12,r13,t1):
    M = np.identity(4)
    M[1] = np.array([r11,r12,r13,t1])
    #M[2] = np.array([r11,r12,r13,t2])
    #M[3] = np.array([r11,r12,r13,t3])
    a = np.array([1,2,3,1])
    return la.norm(np.matmul(M,a))

if __name__ == "__main__":

    ellip_base = ellipsoid(6, 10, 16, levelset=True)
    volume = np.concatenate((ellip_base[:-1, ...], ellip_base[2:, ...]), axis=0)

    
    if TEST_FUSION_DUMMY:
        # Generate a level set about zero of two identical ellipsoids in 3D
        fus = Fusion(volume, volume.min(), subsample_rate = 2, verbose = True)
        print("Solving for a test iteration")
        fus.solve(fus._vertices + 0.01)
        print("skip finding correspondence...")
        print("Updating TSDF...")
        fus.updateTSDF(volume)
        print("Updating deformation graph...")
        fus.update_graph()

    
    if TEST_INPUT:
        sdf_filepath = DATA_PATH + '0000.dist'
        b_min, b_max, volume, closest_points = load_sdf(sdf_filepath, verbose=True)
        sdf_filepath1 = DATA_PATH + '0001.dist'
        b_min1, b_max1, volume1, closest_points1 = load_sdf(sdf_filepath1, verbose=True)
        
        res_x, res_y, res_z = volume.shape
        print("Shape of volume: (%d, %d, %d)" % volume.shape)

        if TEST_FUSION:
            # Generate a level set about zero of two identical ellipsoids in 3D
            fus = Fusion(volume, volume.min(), subsample_rate = 2, verbose = True)
            print("Setting up correspondences...")
            fus.setupCorrespondences(volume1)
            print("Solving for a test iteration")
            fus.solve()
            print("Updating TSDF...")
            fus.updateTSDF()
            print("Updating deformation graph...")
            fus.update_graph()

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
