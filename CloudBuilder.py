import math
from scipy.spatial.transform import Rotation
import numpy as np
import cv2
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


def Transform(coords, angles):  # angles as pitch, roll, yaw
    pitch, roll, yaw = angles

    R01 = Rotation.from_euler('ZYX', [yaw, pitch, roll])  # yaw pitch roll
    R01 = np.array(R01.as_matrix())

    R12 = np.array(Rotation.from_euler('ZX', [PI/2, PI/2]).as_matrix())  # from image coords to 'camera' (actor) coords
    R = R01 @ R12
    t = np.array(coords)[None].T  # [None] enables transposition

    T = np.concatenate((R, t), axis=1)  # make into a homogeneous transform
    T = np.concatenate((T, np.array([0, 0, 0, 1])[None]))
    return T


def collectData(coords, angles, env, path, save_images=False):
    num_images = len(coords)
    total_size = num_images*env.h*env.w

    with open(path + 'point_cloud.ply', 'wb') as out:
        out.write((ply_header % dict(vert_num=total_size)).encode('utf-8'))
        logging.info('Beginning point cloud extraction')

        for i in range(num_images):
            logging.info("Processing view number %d", (i+1))
            logging.info("\tSetting camera parameters")
            env.setOrientation(angles[i])  # move airsim camera to coords and rotate it
            env.setPosition(coords[i])

            logging.info("\tRetrieving color and disparity data")
            disp = env.getDisparity()  # get image and disparity map from airsim camera
            colors = env.getRGB()

            if save_images:
                logging.info("\tSaving images")
                name = "images/img%d.jpg" % i
                cv2.imwrite(path+name, cv2.cvtColor(colors, cv2.COLOR_RGB2BGR))  # convert and save

            logging.info("\tReprojecting points")
            T = Transform(coords[i], angles[i])  # prepare transform from camera to world coordinates
            P = T @ env.Q
            verts = cv2.reprojectImageTo3D(disp, P)

            verts = verts.reshape(-1, 3)
            colors = colors.reshape(-1, 3)
            verts = np.hstack([verts, colors])
            np.savetxt(out, verts, fmt='%f %f %f %d %d %d ')  # save points
            logging.info("\tFinished processing view")

        logging.info('Saving point cloud')
