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

import math
import numpy as np
from numpy import linalg as la
from scipy.spatial import KDTree
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from skimage import measure
from util import *

class Fusion:
    def __init__(self, tsdf, subsample_rate = 5.0, knn = 4):
        self._tsdf = tsdf
        self._knn = knn
        self._subsample_rate = subsample_rate
        self._nodes = []
        self._neighbor_look_up = []
        self._correspondences = None
        self._kdtree = None
        self._opt_result = None
        self.marching_cubes()
        self.construct_graph()
        
    # Construct deformation graph from canonical vertices (easy)
    def construct_graph(self):
        vert_avg = np.average(self._vertices, axis=0)
        average_distances = []
        for f in self._faces:
            average_distances.append(self.average_edge_dist_in_face(f))
        radius = self._subsample_rate * np.average(np.array(average_distances))
        # uniform sampling
        nodes_v, nodes_idx = uniform_sample(self._vertices,radius)

        '''
        Each node is a 4-tuple (index of corresponding surface vertex dg_idx, 3D position dg_v, 4x4 Transformation dg_se3, weight dg_w) 
        HackHack:
        Not sure how to determine dgw. Use 1 for now.
        '''
        for i in range(len(nodes_v)):
            self._nodes.append((nodes_idx[i],
                                nodes_v[i],
                                np.identity(4),
                                1.0))

        # construct kd tree
        self._kdtree = KDTree(nodes_v)

        for vert in self._vertices:
            pts, idx = self._kdtree.query(vert, k=self._knn)
            self._neighbor_look_up.append(idx)
            
    # Perform surface fusion for each voxel center with a tsdf query function for the live frame
    def updateTSDF(self, curr_tsdf):
        pass

    # Update the deformation graph after new surafce vertices are found (easy)
    def update_graph(self):
        pass

    # Solve for a warp field {dg_SE} with correspondences to the live frame
    '''
    Correspondences format: A list of 3D corresponding points in live frame. 
                            Length should equal the length of surface vertices (self._vertices). 
                            The indices should also match with the surface vertices array indices. 

    E = Data_term(Warp Field, surface points, surface normals) + Regularization_term(Warp Field, deformation graph)
    Nonlinear least square problem. The paper solved it by Iterative Gauss-Newton with a Sparse Cholesky Solver.
    how about scipy.optimize.least_squares?    
    '''
    def solve(self, correspondences,
              tukey_data_weight =  0.01,
              tukey_regularization_weight = 0.0001,
              regularization_weight = 200):
        self._correspondences = correspondences
        values = np.concatenate([ dg[2] for dg in self._nodes], axis=0).flatten()
        # We may consider using other optimization library
        self._opt_result = least_squares(computef,
                                         values,
                                         jac='3-point',
                                         tr_solver='lsmr',
                                         args=(tuky_data_weight, tukey_regularization_weight, regularization_weight))

    # TODO, Optional: we can compute a sparsity structure to greatly speed up the optimizer
    def computeSparsity(self, n):
        sparsity = lil_matrix((n,n), dtype=np.float32)
        '''
        fill non-zero entries with 1
        '''
        return sparsity
                               
    # Compute residual function. Input is a flattened vector {dg_SE3}
    def computef(self, x, tdw, trw, rw):
        f = []
        matrices = x.reshape(-1,4)
        matrices = np.split(matrices, matrices.shape[0]/4, axis=0)

        # Data Term
        for idx in range(len(self._vertices)):
            locations = self._neighbor_look_up[idx]
            knn_matrices = [matrices[i] for i in locations]
            vert_warped, n_warped = self.warp(self._vertices[idx],knn_matrices, self._normals[idx])
            p2s = np.dot(n_warped, vert_warped - self._correspondences[idx])
            f.append(math.sqrt(tukey_biweight_loss(p2s,tdw)))

        # Regularization Term: Instead of regularization tree, just use the simpler knn nodes for now
        for idx in range(len(self._nodes)):
            dgi_se3 = matrices[idx]
            for nidx in self._neighbor_look_up[self._nodes[0]]:
               dgj_v =  np.append(self._nodes[nidx][1],1)
               dgj_se3 = matrices[nidx]
               diff = np.matmul(dgi_se3,dgj_v) - np.matmul(dgj_se3, dgj_v)
               for i in range(3):
                   f.append(math.sqrt(rw * max(self._nodes[idx][3], self._nodes[nidx][3]) * huber_loss(diff[i], trw)))

        return np.array(f)
    
    # Warp a point from canonical space to the current live frame, using the wrap field computed from t-1. No camera matrix needed.
    # Matrices: {dg_SE3} used for dual quaternion blending
    def warp(self, pos, matrices, normal = None):
        pass

    # Interpolate a se3 matrix from k-nearest nodes to pos.
    def dq_blend(self, pos, matrices):
        pass
    
    # Mesh vertices and normal extraction from current tsdf in canonical space
    def marching_cubes(self):
        self._vertices, self._faces, self._normals, values = measure.marching_cubes_lewiner(self._tsdf,
                                                                                            level=0.1,
                                                                                            allow_degenerate=False)  
    # Write the current warp field to file 
    def write_warp_field(self, path, filename):
        pass


    # Write the canonical mesh to file 
    def write_canonical_mesh(self,path,filename):
        pass


    # Process a warp field file and write the live frame mesh
    def write_live_frame_mesh(self,path,filename, warpfield_path):
        pass
    
    def average_edge_dist_in_face(self, f):
        v1 = self._vertices[f[0]]
        v2 = self._vertices[f[1]]
        v3 = self._vertices[f[2]]
        return (cal_dist(v1,v2) + cal_dist(v1,v3) + cal_dist(v2,v3))/3

    # Trilinear interpolation of signed distance 
    def get_tsdf(self, x, y, z):
        pass
    
# helper functions
def cal_dist(a,b):
    return la.norm(a-b)

