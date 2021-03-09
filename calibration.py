from tools.distortion import Distorter
import os
import numpy as np
import math
import cv2

PI = math.pi

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)


def calibrate(calibration_path):
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []

    for file in os.listdir(calibration_path):
        if file.endswith(".png"):
            fname = os.path.join(calibration_path, file)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret:
                print('Info: Checkerboard detected in %s' % file)
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

    print('Info: Calibration done')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist


cwd = os.getcwd()
data_path = cwd + '\\calibration_data\\'
image_path = data_path + 'images\\'  # checkerboard images directory
dist_path = data_path + 'distorted\\'


def main():
    f = 960
    w, h = 1920, 1080
    intrinsic = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    k1, k2, p1, p2, k3 = 0.01, 0.05, 0.002, -0.001, -0.06
    dist_coeffs = np.array([k1, k2, p1, p2, k3])

    myDist = Distorter(intrinsic, dist_coeffs)
    dist_map = myDist.compute_distortion_map((h, w))
    myDist.distort_images_dir(image_path, dist_path, dist_map)

    intrinsic_est, dist_est = calibrate(dist_path)
    print('Initial intrinsic matrix:\n', intrinsic, '\n Estimated:\n', intrinsic_est)
    print('Initial distortion coefficients:\n', dist_coeffs, '\n Estimated:\n', dist_est)


if __name__ == '__main__':
    main()
