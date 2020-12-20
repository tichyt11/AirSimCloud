import pdal
import tifffile
import os
import numpy as np
import GUI
from matplotlib import cm
from scipy import signal
import matplotlib.pyplot as plt

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
 """


# def image2world(row, col, res, h, origin):
#     x = res*col + origin[0]
#     y = res*(h-row) + origin[1]
#     return x, y


# def world2image(x, y, res, h, origin):
#     col = (x - origin[0])/res
#     row = h - (y - origin[1])/res
#     col = math.floor(col)
#     row = math.floor(row)
#     return row, col


def circle_kernel(r):
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = x*x + y*y <= r*r
    return mask.astype(np.uint8)

res = 0.1
grid_origin = np.array([-46, -64])
h, w = 1240, 920
data_path = os.getcwd() + '\\data\\'

# GTin = data_path + 'point_cloud.ply'
GTout = data_path + 'GT_DSM.tif'
# GTcmd = cmd % (GTin.replace('\\', '\\\\'), res, GTout.replace('\\', '\\\\'), grid_origin[0], grid_origin[1], h, w)
# GTpipeline = pdal.Pipeline(GTcmd)
# GTpipeline.execute()
#
# fin1 = data_path + 'ColCloud.las'
# fin2 = data_path + 'ODMCloud.las'
# fin3 = data_path + 'Omvgs_Cloud.las'
fout1 = data_path + 'Col_DSM.tif'
fout2 = data_path + 'ODM_DSM.tif'
fout3 = data_path + 'Omvgs_DSM.tif'

# cmd1 = cmd % (fin1.replace('\\', '\\\\'), res, fout1.replace('\\', '\\\\'), grid_origin[0], grid_origin[1], h, w)
# cmd2 = cmd % (fin2.replace('\\', '\\\\'), res, fout2.replace('\\', '\\\\'), grid_origin[0], grid_origin[1], h, w)
# cmd3 = cmd % (fin3.replace('\\', '\\\\'), res, fout3.replace('\\', '\\\\'), grid_origin[0], grid_origin[1], h, w)

# pipeline1 = pdal.Pipeline(cmd1)
# pipeline2 = pdal.Pipeline(cmd2)
# pipeline3 = pdal.Pipeline(cmd3)
# pipeline1.execute()
# pipeline2.execute()
# pipeline3.execute()

GTarray = tifffile.imread(GTout)
array1 = tifffile.imread(fout1)
array2 = tifffile.imread(fout2)
array3 = tifffile.imread(fout3)

mask = GTarray == 50
num_valid = mask.size - mask.sum()  # number of GT elements with valid data
GTarray = np.ma.masked_array(GTarray, mask=mask)
array1 = np.ma.masked_array(array1, mask=mask)
array2 = np.ma.masked_array(array2, mask=mask)  # erase entries that are not in GT
array3 = np.ma.masked_array(array3, mask=mask)

comp1 = (array1 != 50).sum()
comp2 = (array2 != 50).sum()
comp3 = (array3 != 50).sum()
print('completeness - colmap, odm, omvgs :', comp1, comp2, comp3)

threshold = 0.5
num_s1 = (array1 < GTarray - threshold).sum()
num_s2 = (array2 < GTarray - threshold).sum()
num_s3 = (array3 < GTarray - threshold).sum()
print('smaller than GT by more than %.1f meters - colmap , odm, omvgs:' % threshold, num_s1, num_s2, num_s3)

threshold = 0.5
num_b1 = ((array1 > GTarray + threshold) & (array1 != 50)).sum()
num_b2 = ((array2 > GTarray + threshold) & (array2 != 50)).sum()
num_b3 = ((array3 > GTarray + threshold) & (array3 != 50)).sum()
print('bigger than GT by more than %.1f meters - colmap , odm:' % threshold, num_b1, num_b2, num_b3)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(array1)
ax2.imshow(GTarray)
ax3.imshow(array1 > GTarray + threshold)
plt.show()



def image2world(row, col):
    x = res*col + grid_origin[0]
    y = res*(h-row) + grid_origin[1]
    return x, y

# wo = world2image(0, 0, res, h, grid_origin)  # world origin coords on image

# occupancy = (array >= 7) | (array < -5)
# vis = (array - array.min())/array.max()
# vis = np.uint8(cm.terrain(vis)*255)

# kernel = circle_kernel(4)
# grown_obstacles = signal.convolve2d(occupancy, kernel, mode='same', boundary='fill') > 0
# picker = GUI.Picker(vis, grown_obstacles, array, image2world)
