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
fusion.update_graph()
fusion.marching_cubes()

All input/output datatypes should be be numpy arrays.   

'''

import numpy as np
from scipy.special import KDTree
from .util import *

class Fusion:
    def __init__(self, tsdf):
        self._tsdf = tsdf
        self.construct_graph()
        self.marching_cubes()

    # Construct deformation graph from canonical vertices (easy)
    def construct_graph(self):
        pass
    
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
        pass

    # Write the current warp field to file (easy)
    def write_warp_field(self, path, filename):
        pass


    # Write the canonical mesh to file (easy)
    def write_canonical_mesh(self,path,filename):
        pass


    # Process a warp field file and write the live frame mesh
    def write_live_frame_mesh(self,path,filename, warpfield_path):
        pass


    


    
