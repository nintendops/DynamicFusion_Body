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
    def __init__(self, tsdf, subsample_rate = 5.0, knn = 4):
        self._tsdf = tsdf
        self._knn = knn
        self._subsample_rate = subsample_rate
        self._nodes = []
        self._neighbor_look_up = []
        self._correspondences = None
        self._kdtree = None

        self.marching_cubes()
        self.construct_graph(subsample_rate, knn)
        
    # Construct deformation graph from canonical vertices (easy)
    def construct_graph(self):
        vert_avg = np.average(self._vertices, axis=0)
        average_distances = []
        for f in self._faces:
            average_distances.append(self.average_edge_dist_in_face(f))
        radius = self._subsample_rate * np.average(np.array(average_distances))
        # uniform sampling
        nodes_v = uniform_sample(self._vertices,radius)

        '''
        HackHack:
        Not sure how to determine dgw. Use 1 for now.
        '''
        for dgv in nodes_v:
            self._nodes.append((dgv,np.identity(4),1.0))

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

    '''
    Solve for a warp field {dg_SE} with correspondences to the live frame
    E = Data_term(Warp Field, surface points, surface normals) + Regularization_term(Warp Field, deformation graph)
    Nonlinear least square problem. The paper solved it by Iterative Gauss-Newton with a Sparse Cholesky Solver solving a linear system at each iteration. 
    how about scipy.optimize.least_squares?    
    '''
    def solve(self, correspondences, tukey_data_weight =  0.01, tukey_regularization_weight = 0.0001, regularization_weight = 200):
        # (ps,pl) = (point in canonical frame, point in live frame)
        self._correspondences = correspondences
        values = np.concatenate([ dg[1] for dg in self._nodes], axis=0).flatten()

        
    # Compute residual function. Inputs are {dg_SE3}
    def computef(self, x):
        # Data Term
        data_energy = []
        idx = 0
        matrices = x.reshape(-1,4)
        matrices = np.split(matrices, matrices.shape[0]/4, axis=0)
        for vert in self._vertices:
            locations = self._neighbor_look_up[idx]
            

    # Warp a point from canonical space to the current live frame, using the wrap field computed from t-1. No camera matrix needed.
    def warp(self, pos):
        pass

    # Interpolate a se3 matrix from k-nearest nodes to pos. If locations are not given, search for k-nearest neighbors. 
    def dq_blend(self, pos, locations=None):
        pass
    
    # Mesh vertices and normal extraction from current tsdf in canonical space
    def marching_cubes(self):
        self._vertices, self._faces, self._normals, values = measure.marching_cubes_lewiner(self._tsdf, level=0.1,allow_degenerate=False)  

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

