import pdal
import tifffile
import os
import numpy as np
from tools import GUI
from matplotlib import cm
from scipy import signal
import math
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


def circle_kernel(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    mask = x * x + y * y <= r * r
    return mask.astype(np.uint8)


def square_kernel(r):
    mask = np.ones((2 * r, 2 * r))
    return mask.astype(np.uint8)


class DsmHandler:
    def __init__(self, grid_origin, h, w, res=0.3, no_data=1000):
        self.grid_origin = grid_origin
        self.h, self.w = h, w
        self.res = res
        self.no_data = no_data

    def image2world(self, row, col):
        x = self.res*col + self.grid_origin[0]
        y = self.res*(self.h-row) + self.grid_origin[1]
        return x, y

    def image2worldXYZ(self, row, col, array, alt):
        z = array[row, col] + alt
        x = self.res * col + self.grid_origin[0]
        y = self.res * (self.h - row) + self.grid_origin[1]
        return x, y, z

    def world2image(self, x, y, origin):  # for given world point returns its coords in the dsm
        col = (x - origin[0])/self.res
        row = self.h - (y - origin[1])/self.res
        col = math.floor(col)
        row = math.floor(row)
        return row, col

    def createDSM(self, fin, fout=None):
        if not fout:
            fout = fin.strip('ply').strip('las') + 'tif'
        print('Creating %s' % fout)
        command = cmd % (fin.replace('\\', '\\\\'), self.res, fout.replace('\\', '\\\\'), self.no_data,
                         self.grid_origin[0], self.grid_origin[1], self.h, self.w)
        pipeline = pdal.Pipeline(command)
        pipeline.execute()
        return self.loadDSM(fout)

    def loadDSM(self, fname):
        return tifffile.imread(fname)

    def generateMetrics(self, dsm, groundtruth, threshold):
        mask = groundtruth == self.no_data
        groundtruth = np.ma.masked_array(groundtruth, mask=mask)
        dsm = np.ma.masked_array(dsm, mask=mask)  # take out only values for which gt is available

        valid = (dsm == self.no_data).sum()  # number of elements which have valid data

        dsm = np.ma.masked_array(dsm, mask=(dsm == self.no_data))
        num_less = (dsm < groundtruth - threshold).sum()  # number of elements less than GT by at least threshold
        num_more = (dsm > groundtruth + threshold).sum()  # number of elements more than GT by at least threshold

        metrics = {'valid': valid, 'num_less': num_less, 'num_more': num_more}
        return metrics

    def generateOccupancyGrid(self, dsm, min_val, max_val, grow_size=4):
        occupancy = (dsm < min_val) | (dsm > max_val)
        plt.show()

        kernel = circle_kernel(grow_size)
        grown_obstacles = signal.convolve2d(occupancy, kernel, mode='same', boundary='fill') > 0
        return grown_obstacles

    def GeneratePath(self, dsm, rel_alt, min_alt, max_alt):
        vis = np.ma.masked_array(dsm, mask=dsm == self.no_data)
        vis = (vis - vis.min())/vis.max()  # normalize
        vis = np.uint8(cm.terrain(vis)*255)  # render as rgb for visualization

        occupancy = self.generateOccupancyGrid(dsm, min_alt, max_alt)

        picker = GUI.Picker(vis, occupancy, dsm, self.image2world)
        world_path = [self.image2worldXYZ(x, y, dsm, rel_alt) for x, y in picker.path]
        return world_path


data_path = os.getcwd() + '\\unnormalized_data\\'

GTCloud = data_path + 'Clouds\\GT.ply'
GTDsm = data_path + 'DSMs\\GT_DSM.tif'

odm_cloud = data_path + 'Clouds\\ODMCloud.las'
odm_dsm = data_path + 'DSMs\\ODM_DSM.tif'

fin1 = data_path + 'ColCloud.las'
fin3 = data_path + 'Omvgs_Cloud.las'
fout1 = data_path + 'Col_DSM.tif'
fout3 = data_path + 'Omvgs_DSM.tif'


def main():
    grid_origin = np.array([-10, -20])  # bottom left corner of dsm world coordinates
    h, w = 440, 500

    handler = DsmHandler(grid_origin, h, w, 0.3, 1000)
    dsm = handler.createDSM(odm_cloud)

    actual_path = handler.GeneratePath(dsm, 1, -11, 5)


if __name__ == '__main__':
    main()
