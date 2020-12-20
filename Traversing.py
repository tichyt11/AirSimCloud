from AirSimEnv import AirSimUAV
from Mapping import make_waypoints, rotate_waypoints
import math
import os
from geo import lla_from_topocentric

PI = math.pi


def main():

    survey_path = os.getcwd() + '\\survey\\'

    cam_params = (PI/2, 640, 360)
    ref_coords = (reflat, reflon, refalt) = (0, 3, 0)  # ref lat, lon and alt for topocentric coords
    rect = (0, 0, 120, 80)
    coords, angles, _ = make_waypoints(rect, altitude=30, cam_params=cam_params)

    env = AirSimUAV()
    env.drawPoints(coords)
    env.takeOff()
    # env.survey(coords, angles, survey_path)
    env.moveTo([40, 40, 20])
    gps = env.getGeoPos(ref_coords)
    x, y, z = env.getXYZPos()
    print(lla_from_topocentric(x, y, z, reflat, reflon, refalt))
    print(gps)

if __name__=='__main__':
    main()