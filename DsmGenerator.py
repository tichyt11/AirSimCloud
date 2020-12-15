import pdal
import tifffile
import os
import math
import numpy as np
import GUI


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


data_path = os.getcwd() + '\\data\\'
fin = data_path + 'ColCloud.las'
fout = data_path + 'DSM.tif'
tilesize = 0.1
grid_origin = np.array([-46, -64])
h, w = 1240, 920

cmd = """
 [
     "%s",
     {
         "resolution": %f,
         "gdaldriver":"GTiff",
         "filename":"%s",
         "output_type":"max",
         "nodata":100,
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

array = (array - array.min())
occupancy = (array >= 10) | (array < 1)
vis = 255*array/array.max()
vis = vis.astype(np.uint8)

picker = GUI.Picker(vis, occupancy)
