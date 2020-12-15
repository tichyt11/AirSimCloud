import pdal
import tifffile
import os
import math
import numpy as np
import GUI
from matplotlib import cm
from scipy import signal


def image2world(row, col, res, h, origin):
    x = res*col + origin[0]
    y = res*(h-row) + origin[1]
    return x, y


def world2image(x, y, res, h, origin):
    col = (x - origin[0])/res
    row = h - (y - origin[1])/res
    col = math.floor(col)
    row = math.floor(row)
    return row, col


def circle_kernel(r):
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = x*x + y*y <= r*r
    return mask.astype(np.uint8)


data_path = os.getcwd() + '\\data\\'
fin = data_path + 'ColCloud.las'
fout = data_path + 'DSM.tif'
tilesize = 0.3
grid_origin = np.array([-46, -64])
h, w = 400, 300

cmd = """
 [
     "%s",
     {
         "resolution": %f,
         "gdaldriver":"GTiff",
         "filename":"%s",
         "output_type":"max",
         "nodata":50,
         "origin_x": %d,
         "origin_y": %d,
         "height"   : %d,
         "width"  : %d
     }
 ]
 """ % (fin.replace('\\', '\\\\'), tilesize, fout.replace('\\', '\\\\'), grid_origin[0], grid_origin[1], h, w)

pipeline = pdal.Pipeline(cmd)
pipeline.execute()
array = tifffile.imread(fout)

wo = world2image(0, 0, tilesize, h, grid_origin)  # world origin coords on image

occupancy = (array >= 7) | (array < -5)
vis = (array - array.min())/array.max()
vis = np.uint8(cm.terrain(vis)*255)

kernel = circle_kernel(4)
grown_obstacles = signal.convolve2d(occupancy, kernel, mode='same', boundary='fill') > 0

picker = GUI.Picker(vis, grown_obstacles, array)
