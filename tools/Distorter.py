import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.0001)


class Distorter:
    def __init__(self, intrinsic_matrix, dist_coeffs):
        self.dist_coeffs = np.array(dist_coeffs)
        self.intrinsic = intrinsic_matrix

    def compute_distortion_map(self, shape):
        h, w = shape
        coords = np.where(np.ones((h, w)) == 1)  # coords on the distorted image
        coords = np.flip(np.array(coords).transpose().reshape((1, h * w, 2)), axis=2).astype(np.float32)
        # where are their undistorted pre-images
        dist_map = cv2.undistortPointsIter(coords, self.intrinsic, self.dist_coeffs, P=self.intrinsic, R=np.eye(3), criteria=TERM_CRITERIA)
        dist_map = dist_map.reshape((h, w, 2))
        return dist_map

    def distort_image(self, img, dist_map=None):
        if dist_map is None:
            dist_map = self.compute_distortion_map(img.shape[:2])
        return cv2.remap(img, dist_map, None, cv2.INTER_CUBIC)

    def distort_images_dir(self, img_dir, dist_dir, dist_map=None):
        if not os.path.exists(img_dir):
            print('Image directory %s does not exist' % img_dir)
            return None
        if not os.path.exists(dist_dir):
            os.mkdir(dist_dir)

        for fname in os.listdir(img_dir):
            if fname.split('.')[-1] not in ['jpg', 'png', 'tif']:
                continue
            dist_path = os.path.join(dist_dir, 'dist' + fname)
            img = cv2.imread(os.path.join(img_dir, fname))
            dist_image = self.distort_image(img, dist_map)
            cv2.imwrite(dist_path, dist_image)
            print('Info: Distorted %s' % fname)
        # save camera info into text file for convenience
        textfile = os.path.join(dist_dir, 'camera.txt')
        with open(textfile, 'w') as f:
            for row in self.intrinsic:
                np.savetxt(f, row, fmt='%.3f')
            np.savetxt(f, self.dist_coeffs)


def main():
    f = 960
    w, h = 1920, 1080
    intrinsic = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
    k1, k2, p1, p2 = 0.002, 0.001, 0.003, 0.001,
    dist_coeffs = np.array([k1, k2, p1, p2])

    imgpath = os.path.join(os.getcwd(), '../maze_cv_data/images/rgb0.jpg')
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    MyDistorter = Distorter(intrinsic, dist_coeffs)
    distorted_image = MyDistorter.distort_image(img)

    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.imshow(distorted_image)
    plt.show()


if __name__ == '__main__':
    main()