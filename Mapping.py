from tools.AirSimEnv import AirSimCV
from tools.DataTools import *
import os
import numpy as np
import math
from math import sin, cos, tan
from scipy.spatial.transform import Rotation

PI = math.pi
cwd = os.getcwd()
data_path = cwd + '\\forest_maze\\'
image_path = data_path + 'images\\'
disp_path = data_path + 'disps\\'


def make_waypoints(rect, altitude, cam_params, frontlap=0.8, sidelap=0.7):
    x0, y0, difx, dify = rect
    fov, camw, camh = cam_params
    coords = []
    angles = []

    sizew = 2*math.tan(fov/2)*altitude  # width of captured rectangle at h=0
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
            pos = [x0+x*offsetx, y0+y*offsety, altitude]
            ori = [-90, 0, 0]
            coords.append(pos)
            angles.append(ori)

    return coords, angles, numx

# def make_waypoints_angle(rect, altitude, cam_params, theta=0, frontlap=0.8, sidelap=0.7):  # TODO every second row reverse
#     assert 0 < theta < 90
#     x0, y0, difx, dify = rect
#     fov, camw, camh = cam_params
#     coords = []
#     angles = []
#
#     sizew = 2*math.tan(fov/2)*altitude  # width of captured rectangle at h=0
#     sizeh = sizew*camh/camw  # height of captured rectangle at h=0
#
#     offsety = sizew*(1-sidelap)
#     offsetx = sizeh*(1-frontlap)
#
#     offset = offsetx * np.array([cos(theta*PI/180), sin(theta*PI/180)])  # (x, y) rotated offset between rows
#     k = tan(theta*PI/180)
#
#     def line_at_y(y, i):
#         x = k * y + i*offset[0]  # TODO rethink this whole function when rested
#         return x
#
#     def line_at_x(x, i):
#         y = (x - i*offset[0])/k
#         return y
#
#     for i in range(math.floor(difx/offset[0])): # above and including main line
#         y0 = 0
#         x0 = line_at_y(0, i)
#         if k*dify + i*offset[0] <= difx:  # line intersecting right side of the rect
#             y1 = dify
#             x1 = k*dify + i*offset[0]
#         else:
#             x1 = difx
#             y1 = line_at_x(difx, i)
#         vector = np.array([x1 - x0, y1 - y0])
#         lngth = np.linalg.norm(vector)
#         num_points = math.floor(lngth/offsety)
#         if num_points % 2 == 0:  # even number of points
#             startpoint = np.array([x0 ,y0]) + vector/2  # TODO bullshit
#
#
#     numy = math.ceil(dify/offsety)  # num of columns
#     numx = math.ceil(difx/offsetx)  # num of rows for
#
#     if numx < 2:  # at least 2x2 area for reconstruction
#         numx = 2
#     if numy < 2:
#         numy = 2
#
#     for y in range(numy):
#         for x in range(numx):
#             pos = [x0+x*offsetx, y0+y*offsety, altitude]
#             ori = [-90, 0, 0]
#             coords.append(pos)
#             angles.append(ori)
#
#     return coords, angles, numx

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

    env = AirSimCV()
    env.disableLODs()

    cam_params = (env.FOV, env.w, env.h)
    ref_coords = (reflat, reflon, refalt) = (3, 3, 0)  # ref lat, lon and alt for topocentric coords
    # rect = (0, 0, 140, 100)
    rect = (0, -60, 200, 120)


    coords, angles, numx = make_waypoints(rect, altitude=35, cam_params=cam_params)

    # coords2xyz_txt(data_path, coords, form='jpg')
    # coords2gps_txt(data_path, coords, ref_coords, form='jpg')
    # save_rgbs_gps(coords, angles, env, image_path, ref_coords, form='jpg')
    get_cloud(coords, angles, env, data_path)

    # save_disps(coords, angles, env, disp_path)
    # save_rgbs(coords, angles, env, image_path)
    # build_cloud_from_saved(coords, angles, data_path, env.w*env.h, env.Q)


if __name__ == "__main__":
    main()
