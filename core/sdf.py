import numpy as np
import os
from .gl import glm as glm
from .gl.glrender import GLRenderer
from .meshutil import regularize_mesh, load_mesh
from .colorutil import distinct_colors, image_color2idx
from .net import DHBC
import tensorflow as tf

'''
  ***  Signed distance field file is binary. Format:
    - resolutionX,resolutionY,resolutionZ (three signed 4-byte integers (all equal), 12 bytes total)
    - bminx,bminy,bminz (coordinates of the lower-left-front corner of the bounding box: (three double precision 8-byte real numbers , 24 bytes total)
    - bmaxx,bmaxy,bmaxz (coordinates of the upper-right-back corner of the bounding box: (three double precision 8-byte real numbers , 24 bytes total)
    - distance data (in single precision; data alignment: 
        [0,0,0],...,[resolutionX,0,0],
        [0,1,0],...,[resolutionX,resolutionY,0],
        [0,0,1],...,[resolutionX,resolutionY,resolutionZ]; 
    total num bytes: sizeof(float)*(resolutionX+1)*(resolutionY+1)*(resolutionZ+1))
    - closest point for each grid vertex(3 coordinates in single precision)
'''


def load_sdf(file_path, read_closest_points=False, verbose=False):
    '''

    :param file_path: file path
    :param read_closest_points: whether to read closest points for each grid vertex
    :param verbose: verbose flag
    :return:
        b_min: coordinates of the lower-left-front corner of the bounding box
        b_max: coordinates of the upper-right-back corner of the bounding box
        volume: distance data in shape (resolutionX+1)*(resolutionY+1)*(resolutionZ+1)
        closest_points: closest points in shape (resolutionX+1)*(resolutionY+1)*(resolutionZ+1)
    '''
    with open(file_path, 'rb') as fp:

        res_x = int(np.fromfile(fp, dtype=np.int32,
                                count=1))  # note: the dimension of volume is (1+res_x) x (1+res_y) x (1+res_z)
        res_x = - res_x
        res_y = -int(np.fromfile(fp, dtype=np.int32, count=1))
        res_z = int(np.fromfile(fp, dtype=np.int32, count=1))
        if verbose: print("resolution: %d %d %d" % (res_x, res_y, res_z))

        b_min = np.zeros(3, dtype=np.float64)
        b_min[0] = np.fromfile(fp, dtype=np.float64, count=1)
        b_min[1] = np.fromfile(fp, dtype=np.float64, count=1)
        b_min[2] = np.fromfile(fp, dtype=np.float64, count=1)
        if verbose: print("b_min: %f %f %f" % (b_min[0], b_min[1], b_min[2]))

        b_max = np.zeros(3, dtype=np.float64)
        b_max[0] = np.fromfile(fp, dtype=np.float64, count=1)
        b_max[1] = np.fromfile(fp, dtype=np.float64, count=1)
        b_max[2] = np.fromfile(fp, dtype=np.float64, count=1)
        if verbose: print("b_max: %f %f %f" % (b_max[0], b_max[1], b_max[2]))

        grid_num = (1 + res_x) * (1 + res_y) * (1 + res_z)
        volume = np.fromfile(fp, dtype=np.float32, count=grid_num)
        volume = volume.reshape(((1 + res_z), (1 + res_y), (1 + res_x)))
        volume = np.swapaxes(volume, 0, 2)
        if verbose: print("loaded volume from %s" % file_path)

        closest_points = None
        if read_closest_points:
            closest_points = np.fromfile(fp, dtype=np.float32, count=grid_num * 3)
            closest_points = closest_points.reshape(((1 + res_z), (1 + res_y), (1 + res_x), 3))
            closest_points = np.swapaxes(closest_points, 0, 2)

    return b_min, b_max, volume, closest_points


##############################
# The following comes all CNN code

def cnnInitialize():
    # Globally initialize a CNN sesseion
    print('Initialize network...')
    tf.Graph().as_default()
    dhbc = DHBC()
    input = tf.placeholder(tf.float32, [1, None, None, 1])
    feature = dhbc.forward(input)

    path = os.path.dirname(os.path.abspath(__file__))
    print(path)
    
    # Checkpoint
    checkpoint = path + '/models/model'
    print('Load checkpoit from {}...'.format(checkpoint))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(dhbc.feat_vars)
    saver.restore(sess, checkpoint)
    return feature, sess

def compute_correspondence(feature, sess, vertices, faces, znear=1.0, zfar=3.5, max_swi=70, width=512, height=512, flipyz=False):
    '''
    Compute a correspondence vector for mesh (vertices, faces)
    :param vertices: mesh vertices
    :param faces: mesh faces
    :param znear:
    :param zfar:
    :param max_swi:
    :param width:
    :param height:
    :param flipyz:
    :return: For each vertex a 16-digit correspondence vector. [N, 16]
    '''
    b = zfar * znear / (znear - zfar)
    a = -b / znear
    
    renderer = GLRenderer(b'generate_mesh_feature', (width, height), (0, 0), toTexture=True)
    proj = glm.perspective(glm.radians(70), 1.0, znear, zfar)

    vertices = regularize_mesh(vertices, flipyz)

    faces = faces.reshape([faces.shape[0] * 3])
    vertex_buffer = vertices[faces]
    vertex_color = distinct_colors(vertices.shape[0])
    vertex_color_buffer = (vertex_color[faces] / 255.0).astype(np.float32)

    cnt = np.zeros([vertices.shape[0]], dtype=np.int32)
    feat = np.zeros([vertices.shape[0], 16], dtype=np.float32)

    swi = 35
    dis = 200
    for rot in range(0, 360, 15):
        mod = glm.identity()
        mod = glm.rotate(mod, glm.radians(swi - max_swi / 2), glm.vec3(0, 1, 0))
        mod = glm.translate(mod, glm.vec3(0, 0, -dis / 100.0))
        mod = glm.rotate(mod, glm.radians(rot), glm.vec3(0, 1, 0))
        mvp = proj.dot(mod)

        rgb, z = renderer.draw(vertex_buffer, vertex_color_buffer, mvp.T)

        depth = ((zfar - b / (z - a)) / (zfar - znear) * 255).astype(np.uint8)
        features = sess.run(feature, feed_dict={input: depth.reshape([1, 512, 512, 1])}).reshape([512, 512, 16])

        vertex = image_color2idx(rgb)

        mask = vertex > 0
        vertex = vertex[mask]
        features = features[mask]
        for i in range(vertex.shape[0]):
            cnt[vertex[i] - 1] += 1
            feat[vertex[i] - 1] += features[i]

    for i in range(vertices.shape[0]):
        if cnt[i] > 0:
            feat[i] /= cnt[i]
    return feat
