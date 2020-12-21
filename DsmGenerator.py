import pdal
import tifffile
import os
import numpy as np
import GUI
from matplotlib import cm
from scipy import signal
import math
from AirSimEnv import AirSimUAV
from matplotlib import pyplot as plt

cmd = """
 [
     "%s",
     {
         "resolution": %f,
         "gdaldriver":"GTiff",
         "filename":"%s",
         "output_type":"max",
         "nodata":%d,
         "origin_x": %d,
         "origin_y": %d,
         "height"   : %d,
         "width"  : %d
     }
 ]
 """

RES = 0.3
grid_origin = np.array([-10, -20])
h, w = 440, 500
data_path = os.getcwd() + '\\unnormalized_data\\'
NODATA = 1000


def circle_kernel(r):
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = x*x + y*y <= r*r
    return mask.astype(np.uint8)


def image2world(row, col):
    x = RES*col + grid_origin[0]
    y = RES*(h-row) + grid_origin[1]
    return x, y


def image2worldXYZ(row, col, array, bias):
    z = array[row, col] + bias
    x = RES * col + grid_origin[0]
    y = RES * (h - row) + grid_origin[1]
    return x, y, z


def world2image(x, y, res, h, origin):
    col = (x - origin[0])/res
    row = h - (y - origin[1])/res
    col = math.floor(col)
    row = math.floor(row)
    return row, col


def createDSM(fin, fout):
    print('Creating %s' % fout)
    command = cmd % (fin.replace('\\', '\\\\'), RES, fout.replace('\\', '\\\\'), NODATA,
                     grid_origin[0], grid_origin[1], h, w)
    pipeline = pdal.Pipeline(command)
    pipeline.execute()
    dsm = tifffile.imread(fout)
    return dsm


def loadDSM(fname):
    return tifffile.imread(fname)


def generateMetrics(dsm, groundtruth, threshold):
    mask = groundtruth == NODATA
    groundtruth = np.ma.masked_array(groundtruth, mask=mask)
    dsm = np.ma.masked_array(dsm, mask=mask)  # take out only values for which gt is available

    valid = (dsm == NODATA).sum()  # number of elements which have valid data

    dsm = np.ma.masked_array(dsm, mask=(dsm == NODATA))
    num_less = (dsm < groundtruth - threshold).sum()  # number of elements less than GT by at least threshold
    num_more = (dsm > groundtruth + threshold).sum()  # number of elements more than GT by at least threshold

    metrics = {'valid': valid, 'num_less': num_less, 'num_more': num_more}
    return metrics


def generateOccupancyGrid(dsm, min_val, max_val):
    occupancy = (dsm < min_val) | (dsm > max_val)
    plt.show()

    kernel = circle_kernel(4)
    grown_obstacles = signal.convolve2d(occupancy, kernel, mode='same', boundary='fill') > 0
    return grown_obstacles


GTCloud = data_path + 'Clouds\\GT.ply'
GTDsm = data_path + 'DSMs\\GT_DSM.tif'

odm_cloud = data_path + 'Clouds\\ODMCloud.las'
odm_dsm = data_path + 'DSMs\\ODM_DSM.tif'

fin1 = data_path + 'ColCloud.las'
fin3 = data_path + 'Omvgs_Cloud.las'
fout1 = data_path + 'Col_DSM.tif'
fout3 = data_path + 'Omvgs_DSM.tif'


def main():

    GTarray = createDSM(odm_cloud, odm_dsm)
    # array1 = createDSM(fin1, fout1)
    # array2 = createDSM(fin2, fout2)
    # array3 = createDSM(fin3, fout3)

    # valid_gt = GTarray != NODATA
    # num_valid = valid_gt.sum()  # number of GT elements with valid data

    # wo = world2image(0, 0, RES, h, grid_origin)  # world origin coords on image
    array = GTarray
    vis = np.ma.masked_array(array, mask=GTarray==NODATA)
    vis = (vis - vis.min())/vis.max()  # normalize
    vis = np.uint8(cm.terrain(vis)*255)  # render as rgb for visualization

    occupancy = generateOccupancyGrid(array, -11, 3)

    picker = GUI.Picker(vis, occupancy, array, image2world)
    path = picker.path

    bias = 1.5
    actual_path = [image2worldXYZ(x, y, array, bias) for x, y in path]
    env = AirSimUAV()
    env.takeOff()
    env.setTraceLine()
    env.moveOnPath(actual_path)
    env.land()


if __name__ == '__main__':
    main()
