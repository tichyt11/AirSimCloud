import os
import numpy as np
import math
import cv2
import getopt
import sys

PI = math.pi

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)

def check_file(fname):
    return os.path.exists(fname)

def calibrate(calibration_path):
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []

    for file in os.listdir(calibration_path):
        if file.endswith(".png") or file.endswith(".jpg"):
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

def main(argv):

    calibration_dir = None
    out_path = None

    try:
        opts, args = getopt.getopt(argv,'hi:o:')
    except getopt.GetoptError:
        print('Arguments parsing error')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('calibrate.py -i <calibrationdir> -o <outputfile>')
            sys.exit()
        elif opt in ('-i'):
            calibration_dir = arg
        elif opt in ('-o'):
            out_path = arg
    
    if calibration_dir is None:
        calibration_dir = os.path.join(os.getcwd(), 'calibration')
    if out_path is None:
        out_path = os.path.join(os.getcwd(), 'output/calibration.txt')

    if not check_file(calibration_dir):
        print('Directory %s does not exist.' % calibration_dir)
        return None

    intrinsic_est, dist_est = calibrate(calibration_dir)
    focal = intrinsic_est[0][2]
    k1, k2, p1, p2, k3 = dist_est[0]

    with open(out_path, 'w') as f:
        f.write('%f\n%f\n%f\n%f\n%f\n%f\n' % (focal, k1, k2, k3, p1, p2))

if __name__ == '__main__':
    main(sys.argv[1:])