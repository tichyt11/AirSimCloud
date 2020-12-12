import pdal
import tifffile
import os
import json
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

data_path = os.getcwd() + '\\data\\'
fin = data_path + 'COLMAP_dockerFeature_WinRest.las'
fout = data_path + 'testout.tif'

cmd = """
 [
     "%s",
     {
         "resolution": 0.1,
         "gdaldriver":"GTiff",
         "filename":"%s",
         "output_type":"max",
         "nodata":100
     }
 ]
 """ % (fin.replace('\\', '\\\\'), fout.replace('\\', '\\\\'))
pipeline = pdal.Pipeline(cmd)
pipeline.execute()
metadata = json.loads(pipeline.metadata)['metadata']
key = list(metadata.keys())[0]
data = metadata[key]

minx = data['minx']
maxx = data['maxx']
miny = data['miny']
maxy = data['maxy']

array = tifffile.imread(fout)

# w = math.ceil((maxx - minx)*10)
# h = int(len(array)/w)

array = array - array.min()
# array = (array <= 10) & (array > 2)  # create occupancy grid

mpl.use('TkAgg')
plt.imshow(array)
plt.show()






