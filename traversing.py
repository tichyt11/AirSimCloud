from tools.airsim_env import AirSimUAV
from mapping import make_waypoints
import math
import os
import numpy as np
from tools.dem_handling import DemHandler
import matplotlib.pyplot as plt
from tools.geo import lla_from_topocentric
import time

PI = math.pi

def main():


    grid_origin = np.array([0, 0])  # bottom left corner of dsm world coordinates
    h, w = 440, 1000
    res = 0.3
    nodata = 1000
    minalt, maxalt = -5,6

    handler = DemHandler(grid_origin, h, w, res, nodata)
    heightmap, bounds = handler.create_heightmap_auto(gt_cloud, gt_dsm)
    print(bounds)
    print(heightmap.shape)
    shp = heightmap.shape
    print(round(math.ceil(2*(bounds[1] - bounds[0]) / res)/2), round(math.ceil(2*(bounds[3] - bounds[2]) / res)/2))
    print('grid_origin is ', bounds[0] + res/2, bounds[2] + res/2)
    print('start image coords are ', shp[0] - math.floor(abs(bounds[2])/res), math.floor(abs(bounds[0])/res))

    plt.imshow(np.ma.masked_array(heightmap, mask=heightmap == 1000))
    plt.show()

    # occupancy = handler.absolute_alt_occupancy(heightmap, minalt, maxalt)
    # world_path = handler.pick_path(heightmap, occupancy, 1)
    #
    # start = (16,16)
    # goal = (30,120)
    # world_path = handler.find_world_path(heightmap, occupancy, start, goal)

if __name__ == '__main__':
    main()
