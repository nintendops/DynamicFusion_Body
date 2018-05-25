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


All input/output datatypes should be be numpy arrays.   

Important hyperparameters:
- subsample_rate: determine the density of deformation nodes embedded in the canonical mesh
- marching_cubes_step_size: determine the quality of the extracted mesh from marching cube
- knn: affect blending and regularization.
- wmax: maximum weight that can be accumulated in tsdf update. Affect the influences of the later fused frames.
- method for finding correspondences

TODO: 
- optimizer is too slow. Most time spent on Jacobian estimation
- (Optional) provide an option to calculate dmax for DQB weight


'''

import math
import numpy as np
import pickle
import os
from numpy import linalg as la
from scipy.spatial import KDTree
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from skimage import measure
from .util import *
from .sdf import *
from . import *


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')

class Fusion:
    def __init__(self, tsdf, trunc_distance, subsample_rate = 5.0, knn = 4, marching_cubes_step_size = 3, verbose = False, use_cnn = True, write_warpfield = True):
        if type(tsdf) is not np.ndarray or tsdf.ndim != 3:
            raise ValueError('Only 3D numpy array is accepted as tsdf')

        self._itercounter = 0
        self._tsdf = tsdf
        self._curr_tsdf = None
        self._tsdfw = np.zeros(tsdf.shape)
        self._tdist = abs(trunc_distance)
        self._lw = np.array([1,0,0,0,0,0.1,0,0],dtype=np.float32)
        self._knn = knn
        self._marching_cubes_step_size = marching_cubes_step_size
        self._nodes = []
        self._neighbor_look_up = []
        self._correspondences = []
        self._kdtree = None
        self._verbose = verbose
        self._write_warpfield = write_warpfield
        
        if use_cnn:
            self.input, self._feature, self._sess = cnnInitialize()
        else:
            self._sess = None
        if verbose:
            print("Running initial marching cubes")
        self.marching_cubes()
        average_distances = []
        for f in self._faces:
            average_distances.append(self.average_edge_dist_in_face(f))
        self._radius = subsample_rate * np.average(np.array(average_distances))

        if verbose:
            print("Constructing initial graph...")
        self.construct_graph()
        
    # Construct deformation graph from canonical vertices
    def construct_graph(self):
        # uniform sampling
        nodes_v, nodes_idx = uniform_sample(self._vertices,self._radius)
        if self._verbose:
            print("%d deformation nodes sampled, with average radius of %f" % (len(nodes_v), self._radius))
        
        '''
        Each node is a 4-tuple (index of corresponding surface vertex dg_idx, 3D position dg_v, 4x4 Transformation dg_se3, weight dg_w) 
        HackHack:
        Not sure how to determine dgw. Use sample radius for now.
        '''
        for i in range(len(nodes_v)):
            self._nodes.append((nodes_idx[i],
                                nodes_v[i],
                                np.array([1,0.00,0.00,0.00,0.00,0.01,0.01,0.00],dtype=np.float32),
                                2 * self._radius))

        # construct kd tree
        self._kdtree = KDTree(nodes_v)
        self._neighbor_look_up = []
        for vert in self._vertices:
            dists, idx = self._kdtree.query(vert, k=self._knn)
            self._neighbor_look_up.append(idx)

    # Perform surface fusion for each voxel center with a tsdf query function for the live frame.
    def updateTSDF(self, curr_tsdf = None, wmax = 100.0):

        if curr_tsdf is not None:
            self._curr_tsdf = curr_tsdf
        
        if self._curr_tsdf is None:
            raise ValueError('tsdf of live frame has not been loaded')            
        if type(self._curr_tsdf) is not np.ndarray:
            raise ValueError('Only accept 3D np array as tsdf')
        elif self._curr_tsdf.ndim != 3:
            raise ValueError('Only accept 3D np array as tsdf')

        '''
        if self._curr_tsdf.shape != self._tsdf.shape:
            raise ValueError('live frame TSDF should match the size of canonical TSDF')
        '''
        itest = 0
        
        it = np.nditer(self._tsdf, flags=['multi_index'], op_flags = ['readwrite'])
        while not it.finished:
            tsdf_s = np.copy(it[0])
            pos = np.array(it.multi_index, dtype=np.float32)
            dists, kdidx = self._kdtree.query(pos, k=self._knn + 1)
            locations = kdidx[:-1]
            matrices = [self._nodes[i][2] for i in locations]            
            tsdf_l = interpolate_tsdf(self.warp(pos, matrices, locations, m_lw = self._lw), self._curr_tsdf)
            if tsdf_l is not None and tsdf_l > -1 * self._tdist:
                wi = 0
                wi_t = self._tsdfw[it.multi_index]
                for idx in locations:
                    wi += la.norm(self._nodes[idx][1] - pos) / len(locations)
                # Update (v(x),w(x))

                if wi_t == 0:
                    wi_t = wi
                
                it[0] = (it[0] * wi_t + min(self._tdist, tsdf_l)*wi)/(wi + wi_t)
                self._tsdfw[it.multi_index] = min(wi + wi_t, wmax)

                if itest % 250 == 0 and itest < 5000:
                    print('original tsdf and weight: (%f,%f). new tsdf and weight: (%f,%f)'%(tsdf_s, wi_t, it[0], min(wi + wi_t, wmax)))
                    print('interpolated tsdf at warp location:',tsdf_l)
                    print('new weight vs original weight:',wi,wi_t)
                
            itest += 1
            it.iternext()

    # Update the deformation graph after new surafce vertices are found
    def update_graph(self):
        self.marching_cubes()
        # update topology of existing nodes
        vert_kdtree = KDTree(self._vertices)
        for i in range(len(self._nodes)):
            pos = self._nodes[i][1]
            se3 = self._nodes[i][2]
            dist, vidx = vert_kdtree.query(pos)
            self._nodes[i] = (vidx, pos, se3, 2*self._radius)

        # find unsupported surface points
        unsupported_vert = []
        for vert in self._vertices:
            dists, kdidx = self._kdtree.query(vert,k=self._knn)
            if min([ la.norm(self._nodes[idx][1] - vert)/self._nodes[idx][3] for idx in kdidx]) >= 1:
                unsupported_vert.append(vert)

        nodes_new_v, nodes_new_idx = uniform_sample(unsupported_vert, self._radius)
        for i in range(len(nodes_new_v)):
            self._nodes.append((nodes_new_idx[i],
                                nodes_new_v[i],
                                self.dq_blend(nodes_new_v[i]),
                                2 * self._radius))

        if self._verbose:
            print("Inserted %d new deformation nodes. Current number of deformation nodes: %d" % (len(nodes_new_v), len(self._nodes)))
            
        # recompute KDTree and neighbors
        self._kdtree = KDTree(np.array([n[1] for n in self._nodes]))
        self._neighbor_look_up = []
        for vert in self._vertices:
            dists, idx = self._kdtree.query(vert, k=self._knn)
            self._neighbor_look_up.append(idx)
            
        # since fusion is complete at this point, delete current live frame data     
        self._curr_tsdf = None
        self._correspondences = []
        if self._write_warpfield:
            self.write_warp_field(DATA_PATH, 'test')
        


    def setupCorrespondences(self, curr_tsdf, method = 'cnn', prune_result = True, tolerance = 0.2):
        self._curr_tsdf = curr_tsdf
        self._correspondences = []
        idx_pruned = []
        lverts, lfaces, lnormals, lvalues = self.marching_cubes(curr_tsdf, step_size = 1)

        print('lverts shape:',lverts.shape)
        
        if self._sess is None or method == 'clpts':
            if self._verbose:
                print('Using closest pts method for finding correspondences...')
            
            l_kdtree = KDTree(lverts)
            i = 0
            
            for idx in range(len(self._vertices)):
                locations = self._neighbor_look_up[idx]
                knn_dqs = [self._nodes[i][2] for i in locations]
                v_warped, n_warped = self.warp(self._vertices[idx],knn_dqs,locations,self._normals[idx], m_lw = self._lw)
                dists, iidx = l_kdtree.query(v_warped,k=self._knn)
                best_pt = lverts[iidx[0]]
                best_cost = 1
                
                
                for idx in iidx:
                    p = lverts[idx]
                    cost = abs(np.dot(n_warped, v_warped - p))
                    if cost < best_cost:
                        best_cost = cost
                        best_pt = p
                if best_cost > tolerance:
                    idx_pruned.append(idx)
                    i+=1
                self._correspondences.append(best_pt)
        else:
            if self._verbose:
                print('Using cnn method for finding correspondences...')
            s_feats = compute_correspondence(self.input, self._feature, self._sess, self._vertices, self._faces)
            l_feats = compute_correspondence(self.input, self._feature, self._sess, lverts,lfaces)
            l_kdtree = KDTree(np.array(l_feats))        
            for idx in range(len(s_feats)):
                dists, iidx = l_kdtree.query(s_feats[idx])
                self._correspondences.append(lverts[iidx])

        if prune_result:
            if method == 'cnn':
                # Prune out bad correspondences
                for idx in range(len(self._vertices)):
                    locations = self._neighbor_look_up[idx]
                    knn_dqs = [self._nodes[i][2] for i in locations]
                    v_warped, n_warped = self.warp(self._vertices[idx],knn_dqs,locations,self._normals[idx], m_lw = self._lw)
                    cost = abs(np.dot(n_warped,v_warped - self._correspondences[idx]))
                    if cost > tolerance:
                        idx_pruned.append(idx)

            if self._verbose:
                print('ratio of correspondence outlier rejection', float(len(idx_pruned))/float(len(self._vertices)))

            # update data (HACKHACK)
            self._vertices = np.delete(self._vertices, idx_pruned, axis=0)
            self._correspondences = np.delete(self._correspondences, idx_pruned, axis=0)
            self._neighbor_look_up = np.delete(self._neighbor_look_up, idx_pruned, axis=0)
            self._normals = np.delete(self._normals, idx_pruned, axis=0)
            self._faces = None

            vert_kdtree = KDTree(self._vertices)
            for i in range(len(self._nodes)):
                pos = self._nodes[i][1]
                se3 = self._nodes[i][2]
                dist, vidx = vert_kdtree.query(pos)
                self._nodes[i] = (vidx, pos, se3, 2*self._radius)



    # Solve for a warp field {dg_SE} with correspondences to the live frame
    '''
    Correspondences format: A list of 3D corresponding points in live frame. 
                            Length should equal the length of surface vertices (self._vertices). 
                            The indices should also match with the surface vertices array indices. 

    E = Data_term(Warp Field, surface points, surface normals) + Regularization_term(Warp Field, deformation graph)
    Nonlinear least square problem. The paper solved it by Iterative Gauss-Newton with a Sparse Cholesky Solver.
    how about scipy.optimize.least_squares?    
    '''
    def solve(self,
              correspondences = None,
              method = 'cnn',
              precompute_lw = True,
              tukey_data_weight =  0.2,
              huber_regularization_weight = 0.001,
              regularization_weight = 1):

        if correspondences is not None:
            self._correspondences = correspondences
            if len(self._correspondences) != len(self._vertices):
                raise ValueError("Please first call setupCorrespondences to compute point to point correspondences between canonical and live frame vertices!")

        iteration = 1
        if method == 'clpts':
            iteration = 3

        self._itercounter += 1
        solver_verbose_level = 0
        if self._verbose:
            solver_verbose_level = 2

        # We may consider using other optimization library
        if precompute_lw:
            if self._verbose:
                print('estimating global transformation lw...')
                
            for iter in range(1):
                values = self._lw
                opt_result = least_squares(self.computef_lw,
                                           values,
                                           max_nfev = 100,
                                           verbose=solver_verbose_level,
                                           args=(tukey_data_weight, 1))
                print('global transformation found:', DQTSE3(opt_result.x))
                self._lw = opt_result.x
                if method == 'clpts':
                    self.setupCorrespondences(self._curr_tsdf, method = 'clpts')

        if self._verbose:
            print('estimating warp field...')
        for iter in range(iteration):
            values = np.concatenate([ dg[2] for dg in self._nodes], axis=0).flatten()
            n = len(self._vertices) + 3 * self._knn * len(self._nodes)
                    
            if iter > 0 and correspondences is None:
                self.setupCorrespondences(self._curr_tsdf, method = 'clpts')

            f =  self.computef(values,tukey_data_weight,huber_regularization_weight, regularization_weight)
            cost_before = 0.5 * np.inner(f,f)
                
            if self._verbose:
                print("Cost before optimization:",cost_before)
                print('Current regularization weight:',regularization_weight)
            
            opt_result = least_squares(self.computef,
                                       values,
                                       method='trf',
                                       jac='2-point',
                                       ftol=1e-5,
                                       tr_solver='lsmr',
                                       jac_sparsity = self.computeSparsity(n, len(values)),
                                       loss = 'huber',
                                       max_nfev = 20,
                                       verbose = solver_verbose_level,
                                       args=(tukey_data_weight, huber_regularization_weight, regularization_weight))
            # Results: (x, cost, fun, jac, grad, optimality)
            new_values = opt_result.x
            if self._verbose:
                diff = la.norm(new_values - values)
                print("Optimized cost at %d iteration: %f" % (self._itercounter, opt_result.cost))
                print("Norm of displacement (total): %f; sum: %f" % (diff, (new_values - values).sum()))
            
            nw_dqs = np.split(new_values,len(new_values)/8)
            for idx in range(len(self._nodes)):
                nd = self._nodes[idx]
                self._nodes[idx] = (nd[0], nd[1], nw_dqs[idx], nd[3])                  

            reduct_rate = (cost_before - opt_result.cost)/cost_before
            # relax regularization
            if reduct_rate > 0.05 and reduct_rate < 0.9:
                regularization_weight /= 8
                if self._verbose:
                    print('Cost reduction rate:', reduct_rate)
            else:
                break
            
                
    # Optional: we can compute a sparsity structure to speed up the optimizer
    def computeSparsity(self, n, m):
        sparsity = lil_matrix((n,m), dtype=np.float32)
        '''
        fill non-zero entries with 1
        '''
        data_term_length = len(self._vertices)

        for idx in range(data_term_length):
            locations = self._neighbor_look_up[idx]
            for loc in locations:
                for i in range(8):
                    sparsity[idx, 8 * loc + i] = 1
        
        for idx in range(len(self._nodes)):
            for i in range(8):
                sparsity[data_term_length + 3*idx, 8 * (idx) + i] = 1
                sparsity[data_term_length + 3*idx + 1, 8 * (idx) + i] = 1
                sparsity[data_term_length + 3*idx + 2, 8 * (idx) + i] = 1

            for nidx in self._neighbor_look_up[self._nodes[idx][0]]:
                for i in range(8):
                    sparsity[data_term_length + 3*idx, 8 * (nidx) + i] = 1
                    sparsity[data_term_length + 3*idx + 1, 8 * (nidx) + i] = 1
                    sparsity[data_term_length + 3*idx + 2, 8 * (nidx) + i] = 1
        
        
        return sparsity

    def computef_lw(self,x,tdw,trw):
        f = []

        # Data Term only        
        for idx in range(len(self._vertices)):
            locations = self._neighbor_look_up[idx]
            knn_dqs = [self._nodes[i][2] for i in locations]
            vert_warped, n_warped = self.warp(self._vertices[idx], knn_dqs, locations, self._normals[idx], m_lw = x)
            p2s = np.dot(n_warped, vert_warped - self._correspondences[idx])
            #f.append(np.sign(p2s) * math.sqrt(tukey_biweight_loss(abs(p2s),tdw)))
            f.append(p2s)
            
        return np.array(f)
    
    # Compute residual function. Input is a flattened vector {dg_SE3}
    def computef(self, x, tdw, trw, rw):
        f = []
        dqs = np.split(x, len(x)/8)

        dte = 0
        rte = 0
        # Data Term        
        for idx in range(len(self._vertices)):
            locations = self._neighbor_look_up[idx]
            knn_dqs = [dqs[i] for i in locations]
            vert_warped, n_warped = self.warp(self._vertices[idx], knn_dqs, locations, self._normals[idx], m_lw = self._lw)                    
            p2s = np.dot(n_warped, vert_warped - self._correspondences[idx])
            f.append(p2s)
            #f.append(np.sign(p2s) * math.sqrt(tukey_biweight_loss(abs(p2s),tdw)))
            dte += f[-1]**2
        # Regularization Term: Instead of regularization tree, just use the simpler knn nodes for now
        for idx in range(len(self._nodes)):
            dgi_se3 = dqs[idx]
            for nidx in self._neighbor_look_up[self._nodes[idx][0]]:
               dgj_v =  self._nodes[nidx][1]
               dgj_se3 = dqs[nidx]
               diff = dqb_warp(dgi_se3,dgj_v) - dqb_warp(dgj_se3, dgj_v)
               for i in range(3):
                   f.append(rw * max(self._nodes[idx][3], self._nodes[nidx][3]) * diff[i]) 
                   #f.append(np.sign(diff[i]) * math.sqrt(rw * max(self._nodes[idx][3], self._nodes[nidx][3]) * huber_loss(diff[i], trw)))
                   rte += f[-1]**2

        '''
        if self._verbose:
            print("Data term energy:%f; reg term energy:%f"%(dte,rte))
        '''
        
        return np.array(f)

    '''
    Warp a point from canonical space to the current live frame, using the wrap field computed from t-1. No camera matrix needed.
    params: 
    dq: dual quaternions {dg_SE3} used for dual quaternion blending
    locations: indices for corresponding node in the graph.
    normal: if provided, return a warped normal as well.
    dmax: if provided, use the value instead of dgw to calculate weights
    m_lw: if provided, apply global rigid transformation
    '''
    def warp(self, pos, dqs = None, locations = None, normal = None, dmax = None, m_lw = None):
        if dqs is None or locations is None: 
            dists, kdidx = self._kdtree.query(pos, k=self._knn+1)
            locations = kdidx[:-1]
            dqs = [self._nodes[i][2] for i in locations]
        
        se3 = self.dq_blend(pos,dqs,locations,dmax)
                  
        pos_warped = dqb_warp(se3,pos)
        if m_lw is not None:
            pos_warped = dqb_warp(m_lw, pos_warped)
        
        if normal is not None:
            normal_warped = dqb_warp_normal(se3, normal)
            if m_lw is not None:
                normal_warped = dqb_warp_normal(m_lw, normal_warped)
            return (pos_warped,normal_warped)
        else:
            return pos_warped
        
    '''
    Not sure how to determine dgw, so following idea from [Sumner 07] to calculate weights in DQB. 
    The idea is to use the knn + 1 node as a denominator. 
    '''
    # Interpolate a se3 matrix from k-nearest nodes to pos.
    def dq_blend(self, pos, dqs = None, locations = None, dmax = None):
        if dqs is None or locations is None: 
            dists, locations = self._kdtree.query(pos, k=self._knn)
            dqs = [self._nodes[i][2] for i in locations]

        dqb = np.zeros(8)
        for idx in range(len(dqs)):
            dg_idx, dg_v, dg_se3, dg_w = self._nodes[locations[idx]]
            dg_dq = dqs[idx]
            if dmax is None:
                w = math.exp( -1.0 * (la.norm(pos - dg_v)/(2*dg_w))**2)
                dqb += w * dg_dq
            else:
                w = math.exp( -1.0 * (la.norm(pos - dg_v)/dmax)**2)
                dqb += w * dg_dq
                
        #Hackhack
        if la.norm(dqb) == 0:
            if self._verbose:
                print('Really weird thing just happend!!!! blended dq is a zero vector')
                print('dqs:', dqs)
                print('locations:', locations)
            return np.array([1,0,0,0,0,0,0,0],dtype=np.float32)

        return dqb / la.norm(dqb)
    
    # Mesh vertices and normal extraction from current tsdf in canonical space
    def marching_cubes(self, tsdf = None, step_size = 0):

        if step_size < 1:
            step_size = self._marching_cubes_step_size
        
        if tsdf is not None:
            return measure.marching_cubes_lewiner(tsdf,
                                                  step_size = step_size,
                                                  allow_degenerate=False)

        self._vertices, self._faces, self._normals, values = measure.marching_cubes_lewiner(self._tsdf,
                                                                                            step_size = step_size,
                                                                                            allow_degenerate=False)
        if self._verbose:
            print("Marching Cubes result: number of extracted vertices is %d" % (len(self._vertices)))

    # Write the current warp field to file 
    def write_warp_field(self, path, filename):
        file = open( os.path.join(path, filename + '__' + str(self._itercounter) + '.p'),'wb')
        pickle.dump(self._nodes, file)


    # Write the canonical mesh to file 
    def write_canonical_mesh(self, path, filename):
        fpath = open(os.path.join(path,filename),'w')
        verts, faces, normals, values = measure.marching_cubes_lewiner(self._tsdf, allow_degenerate=False)
        for v in verts:
            fpath.write('v %f %f %f\n'%(v[0],v[1],v[2]))
        for n in normals:
            fpath.write('vn %f %f %f\n'%(n[0],n[1],n[2]))
        for f in faces:
            fpath.write('f %d %d %d\n'%(f[0],f[1],f[2]))
        fpath.close()

    # Process a warp field file and write the live frame mesh
    def write_live_frame_mesh(self,path,filename, warpfield_path):
        pass
    
    def average_edge_dist_in_face(self, f):
        v1 = self._vertices[f[0]]
        v2 = self._vertices[f[1]]
        v3 = self._vertices[f[2]]
        return (cal_dist(v1,v2) + cal_dist(v1,v3) + cal_dist(v2,v3))/3


