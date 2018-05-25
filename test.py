import numpy as np
import sys
import os
import cProfile
from numpy import linalg as la
from skimage import measure
from skimage.draw import ellipse, ellipsoid
from core import *
from core.util import *
from core.transformation import random_rotation_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


TEST_FUSION = True
TEST_FUSION_DUMMY = False
TEST_UTIL = False

def visualize(tsdf):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    verts, faces, ns, vs = measure.marching_cubes_lewiner(tsdf,setp_size=1,allow_degenerate=False)
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    ax.set_ylabel("y-axis: b = 10")
    ax.set_zlabel("z-axis: c = 16")
    ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, 20)  # b = 10
    ax.set_zlim(0, 32)  # c = 16
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":

    # Generate a level set about zero of two identical ellipsoids in 3D
    ellip_base = ellipsoid(6, 10, 16, levelset=True)
    e2 = ellipsoid(6,10,16, levelset=True)
    volume = ellip_base[:-1, ...]
    volume2 = e2[:-1, ...]

    output_mesh_name = 'mesh.obj'
    if len(sys.argv) >= 2:
        output_mesh_name = sys.argv[1]
    
    if TEST_FUSION_DUMMY:

        fus = Fusion(volume, volume.max(), marching_cubes_step_size = 3, subsample_rate = 2, verbose = True)
        
        print("Solving for a test iteration")
        fus.setupCorrespondences(volume2, method = 'clpts')
        fus.solve(method = 'clpts', tukey_data_weight = 1, regularization_weight=10)
        print("Updating TSDF...")
        fus.updateTSDF()
        print("Updating deformation graph...")
        fus.update_graph()
        
        # Display resulting triangular mesh using Matplotlib. This can also be done
        # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        verts, faces, ns, vs = measure.marching_cubes_lewiner(fus._tsdf,
                                                  step_size = 1,
                                                  allow_degenerate=False)

        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        ax.set_xlabel("x-axis: a = 6 per ellipsoid")
        ax.set_ylabel("y-axis: b = 10")
        ax.set_zlabel("z-axis: c = 16")
        ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
        ax.set_ylim(0, 20)  # b = 10
        ax.set_zlim(0, 32)  # c = 16
        plt.tight_layout()
        plt.show()

    if TEST_FUSION:
        sdf_filepath = os.path.join(DATA_PATH, '0000.64.dist')
        b_min, b_max, volume, closest_points = load_sdf(sdf_filepath, verbose=True)
        
        if TEST_FUSION:
            # Generate a level set about zero of two identical ellipsoids in 3D
            fus = Fusion(volume, volume.max(), subsample_rate = 1.5, knn = 3, marching_cubes_step_size = 2, verbose = True, use_cnn = False)
            fus.write_canonical_mesh(DATA_PATH, 'original.obj')
            f_iter = 1
            datas = os.listdir(DATA_PATH)
            datas.sort()
            for tfile in datas:
                if f_iter > 10:
                    break
                if tfile.endswith('64.dist') and not tfile.endswith('0000.64.dist'):
                    try:
                        b_min1, b_max1, volume, closest_points1 = load_sdf(os.path.join(DATA_PATH,tfile), verbose=True)
                        print("Processing iteration:", f_iter)
                        print("New shape of volume: (%d, %d, %d)" % volume.shape)
                        print("Setting up correspondences...")
                        fus.setupCorrespondences(volume, method = 'clpts')
                        cProfile.run('fus.solve(regularization_weight=0.5, method = "clpts")', 'profiles/solve_' + str(f_iter))
                        print("Updating TSDF...")
                        cProfile.run('fus.updateTSDF()','profiles/updateTSDF_' + str(f_iter))
                        print("Updating deformation graph...")
                        fus.update_graph()
                        f_iter += 1
                    except ValueError as e:
                        print(str(e))
                        break
                    except KeyboardInterrupt:
                        break
            fus.write_canonical_mesh(DATA_PATH,output_mesh_name)
            
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
            

