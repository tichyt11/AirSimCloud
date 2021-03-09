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

    lat, lon, alt = GPSref = (0, 3, 0)

    # env = AirSimUAV(GPSref)
    # maze_path = os.getcwd() + '\\forest_maze\\'
    #
    # cam_params = (PI/2, 640, 360)
    # ref_coords = (reflat, reflon, refalt) = (0, 3, 0)  # ref lat, lon and alt for topocentric coords
    # rect = (0, 0, 120, 80)
    # coords, angles, _ = make_waypoints(rect, altitude=30, cam_params=cam_params)
    #
    # env.survey(coords, angles, survey_path)

    # cloud = maze_path + 'ODM_ultra.las'
    # dsm = maze_path + 'GTsampled.tif'
    # grid_origin = np.array([-15, -60])  # bottom left corner of dsm world coordinates
    # h, w = 500, 700
    # rel_alt, min_alt, max_alt = 1, -5, 6
    #
    # dsm = handler.createDSM(cloud)
    # actual_path = handler.PathPicker(dsm, rel_alt, min_alt, max_alt, start_at_origin=True)
    #
    # env = AirSimUAV()
    # env.moveTo([0, 0, 5])
    # env.setTraceLine()
    # env.moveOnPath(actual_path)
    # env.land()


if __name__ == '__main__':
    main()
