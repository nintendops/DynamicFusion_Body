'''
Dynamic fusion from a live frame at t to the canonical space S. 
The steps are: estimation of warp field -> tsdf fusion -> update deformation nodes
The warp field is associated with a set of warp nodes N={dg_v, dg_w, dg_SE}
The warp nodes are simply subsampled from canonical frame vertices

Once the SE(3) for each warp node is computed, the warp field provides a warp function W(x) which transforms a point x in the canonical space to a point in the live frame.

The Fusion object is initialized with a truncated signed distance volumne of the first frame (treated as the canonical frame).
To process a new frame, we need two things: tsdf of the new frame and correponding points. The processing of a new frame is done by something like:

fusion.solve(corr_to_current_frame)
fusion.updateTSDF(tsdf_of_current_frame)
fusion.marching_cubes()
fusion.update_graph()


All input/output datatypes should be be numpy arrays.   

'''

import numpy as np
from numpy import linalg as la
from scipy.spatial import KDTree
from skimage import measure
from util import *

class Fusion:
    def __init__(self, tsdf, subsample_rate = 5.0):
        self._tsdf = tsdf
        self.marching_cubes()
        self.construct_graph(subsample_rate)
        
    # Construct deformation graph from canonical vertices (easy)
    def construct_graph(self, subsample_rate):
        vert_avg = np.average(self._vertices, axis=0)
        average_distances = []
        for f in self._faces:
            average_distances.append(self.average_dist_from_face(f))
        radius = decimation_factor * np.average(np.array(average_distances))
        # uniform sampling
        nodes_v = uniform_sample(self._vertices,radius)

    
    # Perform surface fusion for each voxel center with a tsdf query function for the live frame (medium)
    def updateTSDF(self, curr_tsdf):
        pass

    # Update the deformation graph after new surafce vertices are found (easy)
    def update_graph(self):
        pass

    '''
    Solve for a warp field {dg_SE} with correspondences to the live frame (super hard)
    E = Data_term(Warp Field, surface points, surface normals) + Regularization_term(Warp Field, deformation graph)
    Nonlinear least square problem. Solved by Iterative Gauss-Newton with a Sparse Cholesky Solver solving a linear system at each iteration. 
    Need to compute the Jacobian (or the Hessian of E) for a vector r, where rtr = E
    Paper did not mention criteria for convergence so we just need to figure something out by ourselves
    
    '''
    def solve(self, correspondences):
        pass

    # Warp a point from canonical space to the current live frame, using the wrap field computed from t-1 (easy)
    def warp(self, pos):
        pass

    # Interpolate a se3 matrix from k-nearest nodes to pos (medium)
    def dq_blend(self, pos):
        pass
    
    # Mesh vertices and normal extraction from current tsdf in canonical space (use library)
    def marching_cubes(self):
        self._vertices, self._faces, self._normals, values = measure.marching_cubes_lewiner(self._tsdf, level=0.1,allow_degenerate=False)  

    # Write the current warp field to file (easy)
    def write_warp_field(self, path, filename):
        pass


    # Write the canonical mesh to file (easy)
    def write_canonical_mesh(self,path,filename):
        pass


    # Process a warp field file and write the live frame mesh
    def write_live_frame_mesh(self,path,filename, warpfield_path):
        pass
    
    def average_dist_from_face(self, f):
        v1 = self._vertices[f[0]]
        v2 = self._vertices[f[1]]
        v3 = self._vertices[f[2]]
        return (cal_dist(v1,v2) + cal_dist(v1,v3) + cal_dist(v2,v3))/3

                                    
# helper functions
def cal_dist(a,b):
    return la.norm(a-b)

