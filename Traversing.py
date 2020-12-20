from AirSimEnv import AirSimUAV, AirSimCV
from Mapping import make_waypoints, rotate_waypoints
import math
import os

PI = math.pi

def main():

    survey_path = os.getcwd() + '\\survey\\'
    image_path = survey_path + 'images\\'

    # +latitude == +x; +longitude == +y in unreal and airsim
    # +latitude == +x; +longitude == -y for me

    cam_params = (PI/2, 640, 360)
    ref_coords = (reflat, reflon, refalt) = (0, 3, 0)  # ref lat, lon and alt for topocentric coords
    rect = (0, 0, 120, 80)
    coords, angles, _ = make_waypoints(rect, altitude=30, cam_params=cam_params)


    env = AirSimUAV()
    env.drawPoints(coords)
    env.takeOff()
    env.setCameraAngle([0,0,0])
    env.survey(coords, angles, survey_path)


if __name__=='__main__':
    main()