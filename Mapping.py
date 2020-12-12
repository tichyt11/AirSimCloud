from AirSimEnv import AirSimEnv
from DataTools import *
import os
import numpy as np
PI = math.pi


def make_trajectory(rect, h, cam_params, frontlap=0.8, sidelap=0.7):
    x0, y0, difx, dify = rect
    FOV, camw, camh = cam_params
    coords = []
    angles = []

    sizew = 2*math.tan(FOV/2)*h  # width of captured rectangle at h=0
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
            pos = [x0+x*offsetx, -y0-y*offsety, h]  # y axis is flipped -> minus sign
            ori = [-90, 0, 0]
            coords.append(pos)
            angles.append(ori)
    return coords, angles, numx


def main():
    data_path = os.getcwd() + '\\data\\'
    image_path = data_path + 'images\\'
    disp_path = data_path + 'disps\\'
    normalized = True

    # env = AirSimEnv()

    # cam_params = (env.FOV, env.w, env.h)
    cam_params = (PI/2, 640, 360)
    ref_coords = (reflat, reflon, refalt) = (0, 3, 0)  # ref lat, lon and alt for topocentric coords
    rect = (1, 1, 60, 60)

    coords, angles, numx = make_trajectory(rect, h=30, cam_params=cam_params)
    angles = np.array(angles)*PI/180  # angles as pitch, roll, yaw in radians

    # write_xyz_txt(data_path, coords, normalized, numx-1)
    # write_gps_txt(data_path, coords, ref_coords)

    # save_images(image_path, coords, angles, env)
    # write_exifs(image_path, coords, ref_coords)

    # get_cloud(coords, angles, env, data_path, normalized=normalized)

    # save_disps(coords, angles, env, disp_path)
    # save_rgbs(coords, angles, env, image_path)
    # size = env.w*env.h
    # build_cloud_from_saved(coords, angles, data_path, size, env.Q, normalized=normalized)

if __name__ == "__main__":
    main()
