import math
from scipy.spatial.transform import Rotation
import numpy as np
import cv2
import shutil
import tempfile
import logging

PI = math.pi

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def Transform(coords, angles):  # pitch, roll, yaw
    pitch, roll, yaw = angles

    R01 = Rotation.from_euler('ZYX', [yaw, pitch, roll])  # yaw pitch roll
    R01 = np.array(R01.as_matrix())

    R12 = np.array(Rotation.from_euler('ZX', [PI/2, PI/2]).as_matrix())  # from image coords to 'camera' (actor) coords
    R = R01 @ R12
    t = np.array(coords)[None].T  # [None] enables transposition

    T = np.concatenate((R, t), axis=1)  # make into a homogeneous transform
    T = np.concatenate((T, np.array([0, 0, 0, 1])[None]))
    return T


def createPointCloud(coords, angles, env, fn):
    totalSize = 0
    temp = tempfile.TemporaryFile()

    logging.info('Beginning point cloud extraction')
    for i in range(len(coords)):
        logging.info("Processing view number %d", (i+1))
        logging.info("\tSetting camera parameters")
        env.setOrientation(angles[i])  # move airsim camera to coords and rotate it
        env.setPosition(coords[i])

        logging.info("\tRetrieving color and disparity data")
        disp = env.getDisparity()  # get image and disparity map from airsim camera
        colors = env.getRGB()

        logging.info("\tReprojecting points from view")
        T = Transform(coords[i], angles[i])  # prepare transform from camera to world coordinates
        P = T @ env.Q
        verts = cv2.reprojectImageTo3D(disp, P)

        verts = verts.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        verts = np.hstack([verts, colors])
        totalSize += len(verts)
        np.savetxt(temp, verts, fmt='%f %f %f %d %d %d ')  # save points to a temporary file
        logging.info("\t Finished processing view")

    logging.info('Creating a .ply file')
    with open(fn, 'wb') as out:  # write ply header and copy data from temp
        out.write((ply_header % dict(vert_num=totalSize)).encode('utf-8'))
        temp.seek(0)
        shutil.copyfileobj(temp, out)
    temp.close()

    logging.info('Point cloud creation done')
