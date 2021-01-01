from tools.AirSimEnv import AirSimUAV
from Mapping import make_waypoints
import math
import os
import numpy as np
from tools.DsmHandler import DsmHandler

PI = math.pi


def main():

    survey_path = os.getcwd() + '\\survey\\'

    cam_params = (PI/2, 640, 360)
    ref_coords = (reflat, reflon, refalt) = (0, 3, 0)  # ref lat, lon and alt for topocentric coords
    rect = (0, 0, 120, 80)
    coords, angles, _ = make_waypoints(rect, altitude=30, cam_params=cam_params)

    # env.survey(coords, angles, survey_path)
    # env.moveTo([40, 40, 20])

    cloud = survey_path + 'cloud.ply'
    grid_origin = np.array([-10, -20])  # bottom left corner of dsm world coordinates
    h, w = 440, 500
    rel_alt, min_alt, max_alt = 1, -5, 10

    handler = DsmHandler(grid_origin, h, w, 0.3, 1000)
    dsm = handler.createDSM(cloud)
    actual_path = handler.PathPicker(dsm, rel_alt, min_alt, max_alt)

    env = AirSimUAV()
    env.takeOff()
    env.setTraceLine()
    env.moveOnPath(actual_path)
    env.land()


if __name__ == '__main__':
    main()
