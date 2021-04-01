import math
import os
import numpy as np
from tools.dem_handling import DemHandler
import matplotlib.pyplot as plt

PI = math.pi

data_path = os.getcwd() + '\\..\\maze_cv_data\\'
gt_dsm = data_path + 'GTsampled22.tif'


def main():

    grid_origin = np.array([0, 0])  # bottom left corner of dsm world coordinates
    h, w = 440, 1000
    res = 0.3
    nodata = 1000
    minalt, maxalt = -5,6

    handler = DemHandler(grid_origin, h, w, res, nodata)
    heightmap, bounds = handler.load_heightmap(gt_dsm)

    plt.imshow(np.ma.masked_array(heightmap, mask=heightmap == 1000))
    plt.show()

    occupancy = handler.absolute_alt_occupancy(heightmap, minalt, maxalt)
    world_path = handler.pick_path(heightmap, occupancy, 1)

    start = (16,16)
    goal = (30,120)
    world_path = handler.find_world_path(heightmap, occupancy, start, goal)


if __name__ == '__main__':
    main()
