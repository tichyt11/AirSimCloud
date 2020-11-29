from AirSimEnv import AirSimEnv
from CloudBuilder import *
import os

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

PI = math.pi


def show(img):  # displays opencv image
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getOptimalTrajectory(x0, y0, difx, dify, h, env, frontlap=0.8, sidelap=0.7):
    coords = []
    angles = []

    sizew = 2*math.tan(env.FOV/2)*h  # width of captured rectangle at h=0
    sizeh = sizew*env.h/env.w  # height of captured rectangle at h=0

    offsety = sizew*(1-sidelap)
    offsetx = sizeh*(1-frontlap)

    numy = math.ceil(dify/offsety)  # num of columns
    numx = math.ceil(difx/offsetx)  # num of rows for

    if numx < 2:
        numx = 2
    if numy < 2:
        numy = 2

    for y in range(numy):
        for x in range(numx):
            pos = [x0+x*offsetx, y0+y*offsety, -h]  # convert height to airsim coords (down)
            ori = [-90, 0, 0]
            coords.append(pos)
            angles.append(ori)
    return coords, angles, numx


def main():
    data_path = os.getcwd() + '\\' + 'data\\'

    env = AirSimEnv()

    coords, angles, numx = getOptimalTrajectory(-30, -30, 60, 60, h=30, env=env)
    angles = np.array(angles)*PI/180  # angles as pitch, roll, yaw in radians

    # writeGPS(data_path, coords, numx-1)
    saveImages(data_path + 'images\\', coords, angles, env)
    # buildCloud(coords, angles, env, data_path)


if __name__ == "__main__":
    main()
