from tools.airsim_env import AirSimCV, AirSimUAV
from tools.data_extraction import *
import os
import numpy as np
import math
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

PI = math.pi

cwd = os.getcwd()
data_path = cwd + '\\controlled_survey\\'
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
    ref_coords = (reflat, reflon, refalt) = (3, 3, 0)  # ref lat, lon and alt for topocentric coords
    rect = (0, -60, 340, 120)  # big survey area
    # rect = (0, -60, 120, 120)  # small survey area

    # env = AirSimCV()
    env = AirSimUAV(gpsref=ref_coords)
    env.disable_lods()
    cam_params = (env.FOV, env.w, env.h)
    waypoints, angles, numx = make_waypoints(rect, altitude=35, cam_params=cam_params)  # fill area with waypoints survey
    env.survey(waypoints, angles[0], v=1, distort=True)

    # env.survey_controlled(waypoints, angles[0], data_path, v=1)

    # coords2xyz_txt(data_path, waypoints, form='jpg')
    # coords2gps_txt(data_path, waypoints, ref_coords, form='jpg')
    # env.save_rgbs_gps(waypoints, angles, image_path, ref_coords, form='jpg')
    # env.get_cloud(waypoints, angles, data_path)



if __name__ == "__main__":
    main()
