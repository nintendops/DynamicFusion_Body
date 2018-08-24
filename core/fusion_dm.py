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
import pyopencl as cl
from scipy.spatial import KDTree
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from skimage import measure
from .util import *
from .sdf import *
from . import *

DATA_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


# DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')


class FusionDM:
    def __init__(self, trunc_distance, K, tsdf_res=256, subsample_rate=5.0, knn=4, marching_cubes_step_size=3,
                 verbose=False, write_warpfield=True):

        self._itercounter = 0
        self._curr_tsdf = None
        self._tdist = abs(trunc_distance)
        self._tsdf_res = tsdf_res
        self._tsdf = np.zeros((tsdf_res, tsdf_res, tsdf_res)) + self._tdist
        self._tsdfw = np.zeros(self._tsdf.shape)
        self._lw = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        # intrinsic matrix K must be 3x3, non-singular
        self._K = K
        self._Kinv = la.inv(K)

        # coordinate mapping from XYZ indices to world coordinates
        self._IND = np.eye(4)
        self._INDinv = la.inv(self._IND)

        self._knn = knn
        self._marching_cubes_step_size = marching_cubes_step_size
        self._subsample_rate = subsample_rate
        self._nodes = []
        self._neighbor_look_up = []
        self._correspondences = []
        self._kdtree = None
        self._verbose = verbose
        self._write_warpfield = write_warpfield

        '''
        self.marching_cubes()
        average_distances = []
        for f in self._faces:
            average_distances.append(self.average_edge_dist_in_face(f))
        self._radius = subsample_rate * np.average(np.array(average_distances))
                
        if verbose:
            print("Constructing initial graph...")
        self.construct_graph()
        '''

    def compute_live_tsdf(self, depths, lws, UseAutoAlignment=False, useICP=False, outputMesh=False):
        if len(depths) != len(lws):
            raise ValueError('length of camera matrix array Ks must equal that of depth maps')

        tsdf_size = self._tsdf_res
        tsdf = np.zeros((tsdf_size, tsdf_size, tsdf_size)) + self._tdist
        tsdfw = np.zeros(tsdf.shape)

        avgs = []
        stds = []

        avg = np.array([-0.03, -0.43, -5.6], dtype='float32')
        std = 1.3

        # align data?
        if UseAutoAlignment:
            if self._verbose:
                print('estimating pt center and scale from input data')
            for idx in range(len(depths)):
                dm = depths[idx]
                dx, dy = dm.shape
                A = lws[idx]
                pts = []
                for x in range(dx):
                    for y in range(dy):
                        if dm[x][y] != 0:
                            A_inv = inverse_rigid_matrix(A)
                            uv = -1 * dm[x][y] * np.array([y, x, 1], dtype='float')
                            pos3 = np.matmul(self._Kinv, uv)
                            pos3_can = np.matmul(A_inv, np.append(pos3, 1))
                            pts.append(pos3_can)

                pts = np.array(pts)
                avgs.append(np.average(pts, axis=0))
                stds.append(np.std(pts, axis=0))

                avgs = np.array(avgs)
                stds = np.array(stds)
                avg = np.average(avgs, axis=0)
                std = np.average(stds)

        scale = 8 * std / self._tsdf_res
        self._IND[0, 0] = scale
        self._IND[1, 1] = scale
        self._IND[2, 2] = scale
        self._IND[0:3, 3] = avg - scale * self._tsdf_res / 2
        self._INDinv = la.inv(self._IND)

        num_of_map = len(depths)

        if self._verbose:
            print('estimate center pt of input depth maps:', avg)
            print('estimate std of input depth maps:', std)

        if useICP:
            for idx in range(num_of_map):
                print('fusing depth map ', idx)
                tsdf = np.zeros((tsdf_size, tsdf_size, tsdf_size)) + self._tdist
                tsdfw = np.zeros(tsdf.shape)
                (tsdf, tsdfw) = self.fuseDepths(depths[idx], lws[idx], tsdf, tsdfw, scale=10 * std / self._tsdf_res,
                                                center=avg)
                if idx == 0:
                    self._tsdf = tsdf
                    self._tsdfw = tsdfw
                    self.marching_cubes()
                else:
                    # perform rigid icp alignment
                    self._lw = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
                    self.solve(tsdf)
                    self.updateTSDF(tsdf)
        else:
            for idx in range(num_of_map):
                self._depthidx = idx
                print('fusing depth map ', idx)
                (tsdf, tsdfw) = self.fuseDepths(depths[idx], lws[idx], tsdf, tsdfw, scale=12 * std / self._tsdf_res,
                                                center=avg)
            self._tsdf = tsdf
            self._tsdfw = tsdfw

        if outputMesh:
            np.save('tsdf_temp', self._tsdf)
            self.write_canonical_mesh(DATA_PATH, 'test.obj')

        return (tsdf, tsdfw)

    def fuseDepths(self, dm, lw, tsdf, tsdf_w, scale=1.0, center=np.zeros(3), wmax=100.0):
        (dmx, dmy) = dm.shape
        print('shape of depth map:', dm.shape)
        sdf_center = np.zeros(3) + self._tsdf_res / 2

        pt_count = 0
        it = np.nditer(tsdf, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            pos = np.array(it.multi_index, dtype=np.float32)

            # align coordinate
            pos = scale * (pos - sdf_center) + center
            # pos = scale * (pos - sdf_center)
            lpos = np.matmul(lw, np.append(pos, 1))
            (u, v) = project_to_pixel(self._K, lpos)
            if u is not None and v is not None and u >= 0 and u < dmy - 1 and v >= 0 and v < dmx - 1:
                z = -1 * dm[int(round(v))][int(round(u))]
                if z > 0:
                    uc = z * np.array([u, v, 1])
                    # signed distance along principal axis of the camera
                    cpos = np.matmul(self._Kinv, uc)
                    tsdf_l = cpos[2] - lpos[2]
                    # tsdf_l = np.sign(cpos[2] - lpos[2]) * la.norm(cpos - lpos)
                    if tsdf_l > -1 * self._tdist:
                        pt_count += 1
                        # TODO: weight here may encode sensor uncertainty 
                        wi = 1
                        wi_t = tsdf_w[it.multi_index]
                        # Update (v(x),w(x))
                        it[0] = (scale * it[0] * wi_t + min(self._tdist, tsdf_l) * wi) / (scale * (wi + wi_t))
                        tsdf_w[it.multi_index] = min(wi + wi_t, wmax)

            it.iternext()

        if self._verbose:
            print("processedprojection pts: ", pt_count)
            print(tsdf.min())
        return (tsdf, tsdf_w)

    def setupCorrespondences(self, curr_tsdf, prune_result=True, tolerance=1.0):
        self._correspondences = []
        self._corridx = []
        lverts, lfaces, lnormals, lvalues = self.marching_cubes(curr_tsdf, step_size=1)

        print('lverts shape:', lverts.shape)

        l_kdtree = KDTree(lverts)
        i = 0

        for idx in range(len(self._vertices)):
            wn = dqb_warp_normal(self._lw, self._normals[idx])
            vp = dqb_warp(self._lw, self._vertices[idx])
            dists, nidxs = l_kdtree.query(vp, k=self._knn)
            best_pt = lverts[nidxs[0]]
            best_cost = 1

            for nidx in nidxs:
                p = lverts[nidx]
                cost = abs(np.dot(wn, vp - p))
                if cost < best_cost:
                    best_cost = cost
                    best_pt = p
            if best_cost <= tolerance:
                self._corridx.append(idx)
                self._correspondences.append(best_pt)

        '''
        if prune_result:
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
        '''

    # Solve for a rigid transformation with correspondences to current tsdf
    def solve(self, curr_tsdf):
        iteration = 3
        self._itercounter += 1
        solver_verbose_level = 0
        if self._verbose:
            solver_verbose_level = 2

        if self._verbose:
            print('estimating global transformation lw...')

        for iter in range(iteration):
            self.setupCorrespondences(curr_tsdf)
            values = self._lw
            opt_result = least_squares(self.computef_lw,
                                       values,
                                       max_nfev=100,
                                       verbose=solver_verbose_level)
            print('global transformation found:', DQTSE3(opt_result.x))
            self._lw = opt_result.x

    # x: current global transformation lw
    def computef_lw(self, x):
        f = []
        # Data Term only
        i = 0
        for idx in self._corridx:
            wn = dqb_warp_normal(x, self._normals[idx])
            vp = dqb_warp(x, self._vertices[idx])
            corrp = self._correspondences[i]
            p2s = np.dot(wn, vp - corrp)
            # f.append(np.sign(p2s) * math.sqrt(tukey_biweight_loss(abs(p2s),tdw)))
            f.append(p2s)
            i += 1
        return np.array(f)

    # Perform surface fusion for each voxel center with a tsdf query function for the live frame.
    def updateTSDF(self, curr_tsdf, wmax=100.0):

        it = np.nditer(self._tsdf, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            tsdf_s = np.copy(it[0])
            pos = np.array(it.multi_index, dtype=np.float32)
            warped_pos = dqb_warp(self._lw, pos)
            tsdf_l = interpolate_tsdf(warped_pos, curr_tsdf)
            if tsdf_l is not None and tsdf_l > -1 * self._tdist:
                wi = 1
                wi_t = self._tsdfw[it.multi_index]
                it[0] = (it[0] * wi_t + min(self._tdist, tsdf_l) * wi) / (wi + wi_t)
                self._tsdfw[it.multi_index] = min(wi + wi_t, wmax)
            it.iternext()

        if self._verbose:
            print("Completed fusion of two tsdf")

    # Mesh vertices and normal extraction from current tsdf in canonical space
    def marching_cubes(self, tsdf=None, step_size=1):

        if step_size < 1:
            step_size = self._marching_cubes_step_size

        if tsdf is not None:
            return measure.marching_cubes_lewiner(tsdf,
                                                  step_size=step_size)

        self._vertices, self._faces, self._normals, values = measure.marching_cubes_lewiner(self._tsdf,
                                                                                            step_size=step_size)
        if self._verbose:
            print("Marching Cubes result: number of extracted vertices is %d" % (len(self._vertices)))

    # Write the current warp field to file 
    def write_warp_field(self, path, filename):
        file = open(os.path.join(path, filename + '__' + str(self._itercounter) + '.p'), 'wb')
        pickle.dump(self._nodes, file)

    # Write the canonical mesh to file
    def write_canonical_mesh(self, path, filename):
        fpath = open(os.path.join(path, filename), 'w')
        verts, faces, normals, values = measure.marching_cubes_lewiner(self._tsdf, level=0, step_size=1,
                                                                       allow_degenerate=False)

        rot = self._IND[:3,:3]
        trans = self._IND[:3, 3]
        for v in verts:
            v = np.matmul(rot, v) + trans
            fpath.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for n in normals:
            n = np.matmul(rot, n)
            fpath.write('vn %f %f %f\n' % (n[0], n[1], n[2]))
        for f in faces:
            fpath.write('f %d//%d %d//%d %d//%d\n' % (f[0] + 1, f[0] + 1, f[1] + 1, f[1] + 1, f[2] + 1, f[2] + 1))
        fpath.close()

    # Process a warp field file and write the live frame mesh
    def write_live_frame_mesh(self, path, filename, warpfield_path):
        pass

    def average_edge_dist_in_face(self, f):
        v1 = self._vertices[f[0]]
        v2 = self._vertices[f[1]]
        v3 = self._vertices[f[2]]
        return (cal_dist(v1, v2) + cal_dist(v1, v3) + cal_dist(v2, v3)) / 3

    # ----------------------------------------------------- functions below not useful for now ---------------------------------------------#

    # Optional: we can compute a sparsity structure to speed up the optimizer
    def computeSparsity(self, n, m):
        sparsity = lil_matrix((n, m), dtype=np.float32)
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
                sparsity[data_term_length + 3 * idx, 8 * (idx) + i] = 1
                sparsity[data_term_length + 3 * idx + 1, 8 * (idx) + i] = 1
                sparsity[data_term_length + 3 * idx + 2, 8 * (idx) + i] = 1

            for nidx in self._neighbor_look_up[self._nodes[idx][0]]:
                for i in range(8):
                    sparsity[data_term_length + 3 * idx, 8 * (nidx) + i] = 1
                    sparsity[data_term_length + 3 * idx + 1, 8 * (nidx) + i] = 1
                    sparsity[data_term_length + 3 * idx + 2, 8 * (nidx) + i] = 1

        return sparsity

    # Compute residual function. Input is a flattened vector {dg_SE3}
    def computef(self, x, tdw, trw, rw):
        f = []
        dqs = np.split(x, len(x) / 8)

        dte = 0
        rte = 0
        # Data Term        
        for idx in range(len(self._vertices)):
            locations = self._neighbor_look_up[idx]
            knn_dqs = [dqs[i] for i in locations]
            vert_warped, n_warped = self.warp(self._vertices[idx], knn_dqs, locations, self._normals[idx],
                                              m_lw=self._lw)
            p2s = np.dot(n_warped, vert_warped - self._correspondences[idx])
            f.append(p2s)
            # f.append(np.sign(p2s) * math.sqrt(tukey_biweight_loss(abs(p2s),tdw)))
            dte += f[-1] ** 2
        # Regularization Term: Instead of regularization tree, just use the simpler knn nodes for now
        for idx in range(len(self._nodes)):
            dgi_se3 = dqs[idx]
            for nidx in self._neighbor_look_up[self._nodes[idx][0]]:
                dgj_v = self._nodes[nidx][1]
                dgj_se3 = dqs[nidx]
                diff = dqb_warp(dgi_se3, dgj_v) - dqb_warp(dgj_se3, dgj_v)
                for i in range(3):
                    f.append(rw * max(self._nodes[idx][3], self._nodes[nidx][3]) * diff[i])
                    # f.append(np.sign(diff[i]) * math.sqrt(rw * max(self._nodes[idx][3], self._nodes[nidx][3]) * huber_loss(diff[i], trw)))
                    rte += f[-1] ** 2

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

    def warp(self, pos, dqs=None, locations=None, normal=None, dmax=None, m_lw=None):
        if dqs is None or locations is None:
            dists, kdidx = self._kdtree.query(pos, k=self._knn + 1)
            locations = kdidx[:-1]
            dqs = [self._nodes[i][2] for i in locations]

        se3 = self.dq_blend(pos, dqs, locations, dmax)

        pos_warped = dqb_warp(se3, pos)
        if m_lw is not None:
            pos_warped = dqb_warp(m_lw, pos_warped)

        if normal is not None:
            normal_warped = dqb_warp_normal(se3, normal)
            if m_lw is not None:
                normal_warped = dqb_warp_normal(m_lw, normal_warped)
            return (pos_warped, normal_warped)
        else:
            return pos_warped

    '''
    Not sure how to determine dgw, so following idea from [Sumner 07] to calculate weights in DQB. 
    The idea is to use the knn + 1 node as a denominator. 
    '''

    # Interpolate a se3 matrix from k-nearest nodes to pos.
    def dq_blend(self, pos, dqs=None, locations=None, dmax=None):
        if dqs is None or locations is None:
            dists, locations = self._kdtree.query(pos, k=self._knn)
            dqs = [self._nodes[i][2] for i in locations]

        dqb = np.zeros(8)
        for idx in range(len(dqs)):
            dg_idx, dg_v, dg_se3, dg_w = self._nodes[locations[idx]]
            dg_dq = dqs[idx]
            if dmax is None:
                w = math.exp(-1.0 * (la.norm(pos - dg_v) / (2 * dg_w)) ** 2)
                dqb += w * dg_dq
            else:
                w = math.exp(-1.0 * (la.norm(pos - dg_v) / dmax) ** 2)
                dqb += w * dg_dq

        # Hackhack
        if la.norm(dqb) == 0:
            if self._verbose:
                print('Really weird thing just happend!!!! blended dq is a zero vector')
                print('dqs:', dqs)
                print('locations:', locations)
            return np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        return dqb / la.norm(dqb)

    # Construct deformation graph from canonical vertices
    def construct_graph(self):
        # uniform sampling
        nodes_v, nodes_idx = uniform_sample(self._vertices, self._radius)
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
                                np.array([1, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00], dtype=np.float32),
                                2 * self._radius))

        # construct kd tree
        self._kdtree = KDTree(nodes_v)
        self._neighbor_look_up = []
        for vert in self._vertices:
            dists, idx = self._kdtree.query(vert, k=self._knn)
            self._neighbor_look_up.append(idx)

    # Update the deformation graph after new surafce vertices are found
    def update_graph(self):
        self.marching_cubes()
        # update topology of existing nodes
        vert_kdtree = KDTree(self._vertices)
        for i in range(len(self._nodes)):
            pos = self._nodes[i][1]
            se3 = self._nodes[i][2]
            dist, vidx = vert_kdtree.query(pos)
            self._nodes[i] = (vidx, pos, se3, 2 * self._radius)

        # find unsupported surface points
        unsupported_vert = []
        for vert in self._vertices:
            dists, kdidx = self._kdtree.query(vert, k=self._knn)
            if min([la.norm(self._nodes[idx][1] - vert) / self._nodes[idx][3] for idx in kdidx]) >= 1:
                unsupported_vert.append(vert)

        nodes_new_v, nodes_new_idx = uniform_sample(unsupported_vert, self._radius)
        for i in range(len(nodes_new_v)):
            self._nodes.append((nodes_new_idx[i],
                                nodes_new_v[i],
                                self.dq_blend(nodes_new_v[i]),
                                2 * self._radius))


        if self._verbose:
            print("Inserted %d new deformation nodes. Current number of deformation nodes: %d" % (
                len(nodes_new_v), len(self._nodes)))

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


class FusionDM_GPU(FusionDM):
    def __init__(self, trunc_distance, K, tsdf_res=256, subsample_rate=5.0, knn=4, marching_cubes_step_size=3,
                 verbose=False, write_warpfield=True):
        FusionDM.__init__(self, trunc_distance=trunc_distance,
                          K=K, tsdf_res=tsdf_res, subsample_rate=subsample_rate,
                          knn=knn, marching_cubes_step_size=marching_cubes_step_size,
                          verbose=verbose,
                          write_warpfield=write_warpfield)
        self._cl_ctx = cl.create_some_context()
        self._cl_queue = cl.CommandQueue(self._cl_ctx)
        if verbose:
            self.verbose_gpu()

    def verbose_gpu(self):
        print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
        # Print each platform on this computer
        for platform in cl.get_platforms():
            print('=' * 60)
            print('Platform - Name:  ' + platform.name)
            print('Platform - Vendor:  ' + platform.vendor)
            print('Platform - Version:  ' + platform.version)
            print('Platform - Profile:  ' + platform.profile)
            # Print each device per-platform
            for device in platform.get_devices():
                print('    ' + '-' * 56)
                print('    Device - Name:  ' + device.name)
                print('    Device - Type:  ' + cl.device_type.to_string(device.type))
                print('    Device - Max Clock Speed:  {0} Mhz'.format(device.max_clock_frequency))
                print('    Device - Compute Units:  {0}'.format(device.max_compute_units))
                print('    Device - Local Memory:  {0:.0f} KB'.format(device.local_mem_size / 1024.0))
                print('    Device - Constant Memory:  {0:.0f} KB'.format(device.max_constant_buffer_size / 1024.0))
                print('    Device - Global Memory: {0:.0f} GB'.format(device.global_mem_size / 1073741824.0))
                print(
                    '    Device - Max Buffer/Image Size: {0:.0f} MB'.format(device.max_mem_alloc_size / 1048576.0))
                print('    Device - Max Work Group Size: {0:.0f}'.format(device.max_work_group_size))
        print('\n')

    def fuseDepths(self, dm, lw, tsdf, tsdf_w, scale=1.0, center=np.zeros(3), wmax=100.0):
        (dmy, dmx) = dm.shape
        print('shape of depth map:', dm.shape)
        sdf_center = np.zeros(3) + self._tsdf_res / 2
        kernel = r"""
        inline float interpolation(__global const float *depth, float px, float py)
        {
            int x = floor(px);
            int y = floor(py);
            float wx = px - x;
            float wy = py - y;
            
            int left_up_id = y * DM_X + x;
            int left_bot_id = (y+1) * DM_X + x;
            int right_up_id = left_up_id + 1;
            int right_bot_id = left_bot_id + 1;
            
            float up_depth = depth[left_up_id] * (1 - wx) + depth[right_up_id] * wx;
            float bot_depth = depth[left_bot_id] * (1 - wx) + depth[right_bot_id] * wx;
            float ret = up_depth * (1 - wy) + bot_depth * wy;
            
            return ret;
        }
        
        inline float naive(__global const float *depth, float px, float py)
        {
            int id = round(py) * DM_X + round(px);
            return depth[id];        
        }
        
        __kernel void fuse_depth(__global float *tsdf, __global float *tsdf_w, __global const float *depth, __global const float *proj, __global const float *K_inv)
        {
            int x = get_global_id(0);
            int y = get_global_id(1);
            int z = get_global_id(2);
        
            // index in the tsdf array
            int idx = z * RES_X * RES_Y + y * RES_X + x;
        
            // project to the image space
            float u = proj[0] * x + proj[1] * y + proj[2] * z + proj[3];
            float v = proj[4] * x + proj[5] * y + proj[6] * z + proj[7];
            float w = proj[8] * x + proj[9] * y + proj[10] * z + proj[11];
        
            // depth location
            float px = u / w;
            float py = v / w;
            if (px < 0 || py < 0 || px >= DM_X - 1 || py >= DM_Y - 1)
                return;
            float pz = -interpolation(depth, px, py);
            
            float dz;
            if (pz <= TDIST)
                dz = -TDIST;
            else {
                px *= pz;
                py *= pz;
                dz = K_inv[6] * (px - u) + K_inv[7] * (py - v) + K_inv[8] * (pz - w);
                dz = -dz;
            }
            float old_tsdf = tsdf[idx];
            /*
            if (old_tsdf > 0)
                tsdf[idx] = min(old_tsdf, dz);
            else if (dz <= 0)
                tsdf[idx] = max(old_tsdf, dz);
            */
            if (dz < TDIST)
            {
                float w = 1;
                float new_w = min(w + tsdf_w[idx], WMAX);
                tsdf[idx] = ((new_w - w) * old_tsdf + w * max(-TDIST, dz)) / new_w;
                tsdf_w[idx] = new_w;
            }
        }"""

        program = cl.Program(self._cl_ctx, """
        #define DM_X %d
        #define DM_Y %d 
        #define RES_X %d
        #define RES_Y %d
        #define RES_Z %d
        #define TDIST %ff
        #define WMAX %ff
        """ % (dmx, dmy,
               tsdf.shape[0], tsdf.shape[1], tsdf.shape[2],
               self._tdist, wmax
               ) + kernel).build()

        mf = cl.mem_flags
        tsdf = tsdf.astype(np.float32)
        tsdf_w = tsdf_w.astype(np.float32)

        tsdf_buffer = cl.Buffer(self._cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=tsdf)
        tsdf_w_buffer = cl.Buffer(self._cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=tsdf_w)
        proj = np.matmul(self._K, np.matmul(lw, self._IND))
        proj_buffer = cl.Buffer(self._cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=proj.astype(np.float32))
        Kinv_buffer = cl.Buffer(self._cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self._Kinv.astype(np.float32))
        depth_buffer = cl.Buffer(self._cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dm.astype(np.float32))

        program.fuse_depth(self._cl_queue, tsdf.shape, None, tsdf_buffer, tsdf_w_buffer, depth_buffer, proj_buffer, Kinv_buffer)
        cl.enqueue_read_buffer(self._cl_queue, tsdf_buffer, tsdf)
        cl.enqueue_read_buffer(self._cl_queue, tsdf_w_buffer, tsdf_w)
        self._cl_queue.finish()
        pt_count = 0
        if False:
            it = np.nditer(tsdf, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                pos = np.array(it.multi_index, dtype=np.float32)

                # align coordinate
                pos = np.matmul(self._IND, np.append(pos, 1))
                # pos = scale * (pos - sdf_center)
                lpos = np.matmul(lw, pos)
                (u, v) = project_to_pixel(self._K, lpos)
                if u is not None and v is not None and u >= 0 and u < dmy - 1 and v >= 0 and v < dmx - 1:
                    z = -1 * dm[int(round(v))][int(round(u))]
                    if z > 0:
                        uc = z * np.array([u, v, 1])
                        # signed distance along principal axis of the camera
                        cpos = np.matmul(self._Kinv, uc)
                        tsdf_l = cpos[2] - lpos[2]
                        # tsdf_l = np.sign(cpos[2] - lpos[2]) * la.norm(cpos - lpos)
                        if tsdf_l > -1 * self._tdist:
                            pt_count += 1
                            # TODO: weight here may encode sensor uncertainty
                            wi = 1
                            wi_t = tsdf_w[it.multi_index]
                            # Update (v(x),w(x))
                            it[0] = (scale * it[0] * wi_t + min(self._tdist, tsdf_l) * wi) / (scale * (wi + wi_t))
                            tsdf_w[it.multi_index] = min(wi + wi_t, wmax)

                it.iternext()

        if self._verbose:
            print("processedprojection pts: ", pt_count)
            print(tsdf.min(), tsdf.max())
        return (tsdf, tsdf_w)
