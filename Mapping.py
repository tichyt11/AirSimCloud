from AirSimEnv import AirSimEnv
from CloudBuilder import *

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

PI = math.pi


def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    env = AirSimEnv()

    n = 16
    coords = []
    scale = 10
    for i in range(4):
        for j in range(4):
            pos = [i*scale-3*scale, j*scale-3*scale, -40]
            coords.append(pos)
    angles = np.array([[-90, 0, 0]]*n)*PI/180  # pitch, roll, yaw

    fn = 'HelloPoints.ply'
    createPointCloud(coords, angles, env, fn)


if __name__ == "__main__":
    main()
