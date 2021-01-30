from tools.AirSimEnv import AirSimCV, AirSimUAV
from tools.Distorter import Distorter
from tools.data_extraction import *
import os
import numpy as np
import math
from math import sin, cos, tan
from scipy.spatial.transform import Rotation

PI = math.pi

cwd = os.getcwd()
data_path = cwd + '\\small_survey_data\\'
image_path = data_path + 'images\\'
disp_path = data_path + 'disps\\'
dist_path = data_path + 'distorted\\'


def make_waypoints(rect, altitude, cam_params, overlapalt=0, frontlap=0.8, sidelap=0.7, flip=True):
    x0, y0, difx, dify = rect
    fov, camw, camh = cam_params
    coords = []
    angles = []

    sizew = 2*math.tan(fov/2)*(altitude-overlapalt)  # width of captured rectangle at h=0
    sizeh = sizew*camh/camw  # height of captured rectangle at h=0

    offsety = sizew*(1-sidelap)
    offsetx = sizeh*(1-frontlap)

    numy = math.ceil(dify/offsety)  # num of columns
    numx = math.ceil(difx/offsetx)  # num of rows for

    if numx < 2:  # at least 2x2 area for reconstruction
        numx = 2
    if numy < 2:
        numy = 2

    for y in range(numy):
        for x in range(numx):
            if flip and y % 2 == 1:
                pos = [x0 + (numx-1-x)*offsetx, y0+y*offsety, altitude]
            else:
                pos = [x0+x*offsetx, y0+y*offsety, altitude]
            ori = [-90, 0, 0]  # face down
            coords.append(pos)
            angles.append(ori)

    return coords, angles, numx


def rotate_waypoints(coords, angles, angle=20):
    mean = coords_mean(coords)
    R = Rotation.from_euler('z', angle*PI/180)
    R = R.as_matrix()
    new_coords = np.array(coords) - mean + np.array([0, 0, -5])
    new_coords = np.dot(R, new_coords.transpose()).transpose()  # rotate coords about center by 20 degs
    new_coords += mean
    new_angles = np.array(angles) + [0, 0, -angle]
    return new_coords, new_angles


def main():

    # env = AirSimCV()
    env = AirSimUAV()
    env.disableLODs()

    cam_params = (env.FOV, env.w, env.h)
    ref_coords = (reflat, reflon, refalt) = (3, 3, 0)  # ref lat, lon and alt for topocentric coords
    # rect = (0, -60, 340, 120)  # big survey area
    rect = (0, -60, 120, 120)  # small survey area

    coords, angles, numx = make_waypoints(rect, altitude=40, cam_params=cam_params)  # fill area with waypoints survey
    env.survey(coords, angles[0], data_path, v=1, distort=True)

    # coords2xyz_txt(data_path, coords, form='jpg')
    # coords2gps_txt(data_path, coords, ref_coords, form='jpg')
    # env.save_rgbs_gps(coords, angles, image_path, ref_coords, form='jpg')
    # env.get_cloud(coords, angles, data_path)

    # env.save_disps(coords, angles, disp_path)
    # env.save_rgbs(coords, angles, image_path)
    # build_cloud_from_saved(coords, angles, data_path, env.w*env.h, env.Q)


if __name__ == "__main__":
    main()
