from AirSimEnv import AirSimEnv
from CloudBuilder import *

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

PI = math.pi


def show(img):  # displays opencv image
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    env = AirSimEnv()

    n = 9
    coords = []
    scale = 15
    for i in range(3):
        for j in range(3):
            pos = [i*scale-3*scale, j*scale-3*scale, -40]
            coords.append(pos)
            coords.append(pos)
            coords.append(pos)
    angles = np.array([[-90, 0, 0], [-110, 0, 0], [-70, 0, 0]]*n)*PI/180  # angles as pitch, roll, yaw in radians

    collectData(coords, angles, env, path='C:/Users/tomtc/PycharmProjects/AirSimSLAM/data/', save_images=True)


if __name__ == "__main__":
    main()
