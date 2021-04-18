import pdal
import tifffile
import os
import numpy as np
from matplotlib import cm, pyplot as plt
from scipy import signal
import math
import tools.cython_files.thetastar as search
import tools.search_gui as gui
import json

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

cmd_auto = """
 [
     "%s",
     {
         "resolution": %f,
         "gdaldriver":"GTiff",
         "filename":"%s",
         "output_type":"max",
         "nodata":1000
     }
 ]
 """


def circle_kernel(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    mask = x * x + y * y <= r*r + math.sqrt(2)*r + 0.5  # r := r + sqrt(0.5) increase the radius by tile half diagonal
    return mask.astype(np.uint8)


def square_kernel(r):
    r = math.ceil(r)
    mask = np.ones((2 * r, 2 * r))
    return mask.astype(np.uint8)


class DemHandler:
    def __init__(self, res=0.3, grid_origin=None, h=None, w=None, no_data=1000):
        self.grid_origin = grid_origin
        self.h, self.w = h, w
        self.res = res
        self.no_data = no_data

        self.grow_size = math.ceil(math.sqrt(0.5)/res)
        self.heightmap = None

    def image2world(self, row, col):
        x = self.res*col + self.grid_origin[0]
        y = self.res*(self.h-row) + self.grid_origin[1]
        return x, y

    def image2world_z(self, row, col, heightmap, alt):
        z = heightmap[row, col] + alt
        x = self.res * col + self.grid_origin[0]
        y = self.res * (self.h - row) + self.grid_origin[1]
        return x, y, z

    def world2image(self, x, y):  # for given world point returns its coords in the dsm
        col = (x - self.grid_origin[0])/self.res
        row = self.h - (y - self.grid_origin[1])/self.res
        col = math.floor(col)
        row = math.floor(row)
        return row, col

    def create_heightmap(self, fin, fout=None):
        if self.h is not None and self.w is not None and self.grid_origin is not None:
            if not fout:
                fout = fin.strip('.ply').strip('.las') + '.tif'
            print('Creating %s' % fout)
            command = cmd % (fin.replace('\\', '\\\\'), self.res, fout.replace('\\', '\\\\'), self.no_data,
                             self.grid_origin[0], self.grid_origin[1], self.h, self.w)
            pipeline = pdal.Pipeline(command)
            pipeline.execute()
            self.heightmap = self.load_heightmap(fout)
            return self.heightmap
        else:
            print('Missing parameters for heightmap creation. Maybe try create_heightmap_auto.')

    def create_heightmap_auto(self, fin, fout=None):
        if not fout:
            fout = fin.strip('.ply').strip('.las') + '_auto.tif'
        print('Creating %s' % fout)
        command = cmd_auto % (fin.replace('\\', '\\\\'), self.res, fout.replace('\\', '\\\\'))
        pipeline = pdal.Pipeline(command)
        pipeline.execute()

        metadata = json.loads(pipeline.metadata)['metadata']['readers.las']
        minx = metadata['minx']
        miny = metadata['miny']
        maxx = metadata['maxx']
        maxy = metadata['maxy']
        bounds = [minx, maxx, miny, maxy]  # for future convenience all 4, only minx and miny are needed

        self.update_grid_origin(bounds)  # set self.grid_origin
        self.heightmap = self.load_heightmap(fout)  # save heightmap
        self.h, self.w = self.heightmap.shape
        return self.heightmap

    def update_grid_origin(self, bounds):
        grid_origin = (bounds[0] + self.res / 2, bounds[2] + self.res / 2)
        self.grid_origin = grid_origin

    def load_heightmap(self, fname):
        return tifffile.imread(fname)

    def grow_heightmap(self, heightmap, grow_size=None):
        if grow_size is None:
            grow_size = self.grow_size
        kern = circle_kernel(grow_size)
        coords = np.where(kern)
        coords = np.array([coords[0], coords[1]])

        grown = np.zeros(heightmap.shape)
        padded = np.pad(heightmap, grow_size, constant_values=heightmap.min())  # pad by minimal value

        for i in range(padded.shape[0] - 2 * grow_size):
            for j in range(padded.shape[1] - 2 * grow_size):
                center = np.array([[i], [j]])
                cs = coords + center
                maxz = max(padded[tuple(cs)])
                grown[i, j] = maxz
        return grown

    def absolute_alt_occupancy(self, heightmap, minalt, maxalt):
        occupancy = (heightmap < minalt) | (heightmap > maxalt) | (heightmap == self.no_data)
        return occupancy

    def gradient_field(self, heightmap):
        sobel_u = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/8
        gx = signal.convolve2d(heightmap, sobel_u, mode='valid', boundary='fill')
        gy = signal.convolve2d(heightmap, sobel_u.transpose(), mode='valid', boundary='fill')
        gradient_magnitude = np.sqrt(np.square(gx) + np.square(gy))
        gradient_magnitude = np.pad(gradient_magnitude, 1)
        return gradient_magnitude

    def gradient_occupancy(self, heightmap, maxgrad):
        gradient_field = self.gradient_field(heightmap)
        occupancy = gradient_field > maxgrad
        return occupancy

    def combined_occupancy(self, heightmap, minalt, maxalt, maxgrad, grow_size=None):
        if grow_size is None:
            grow_size = self.grow_size

        abs_occ = self.absolute_alt_occupancy(heightmap, minalt, maxalt)
        grad_occ = self.gradient_occupancy(heightmap, maxgrad)
        occupancy = abs_occ | grad_occ
        kern = circle_kernel(grow_size)
        grown_occupancy = signal.convolve2d(occupancy, kern, mode='same') >= 1
        return grown_occupancy

    def pick_path(self, heightmap, occupancy, alt=1, start_at_origin=False, world_path=True):
        vis = np.ma.masked_array(heightmap, mask=heightmap == self.no_data)
        vis = (vis - vis.min())/vis.max()  # normalize
        vis = np.uint8(cm.terrain(vis)*255)  # render as rgb for visualization

        if start_at_origin:
            start = self.world2image(0, 0)
        else:
            start = None

        picker = gui.Picker(vis, occupancy, heightmap, alt, self.image2world, manual_start=start)
        if world_path:
            return [self.image2world_z(x, y, heightmap, alt) for x, y in picker.path]
        else:
            return picker.path

    def find_world_path(self, heightmap, occupancy, start, goal, alt=1, world_path=True):  # finds world waypoint between start and goal
        Planner = search.PathFinder(occupancy, heightmap)
        start = self.world2image(start[0], start[1])
        goal = self.world2image(goal[0], goal[1])
        grid_path = Planner.thetastar(start, goal, alt)
        if grid_path is not None:
            if world_path:
                return [self.image2world_z(x, y, heightmap, alt) for x, y in grid_path]
            else:
                return grid_path
        return None

data_path = os.getcwd() + '\\..\\maze_cv_data\\'
gt_cloud = data_path + 'GTsampled.las'
gt_dsm = data_path + 'GTsampled22.tif'


def main():
    grid_origin = np.array([0, 0])  # bottom left corner of dsm world coordinates
    h, w = 440, 1000
    res = 0.3
    nodata = 1000
    minalt, maxalt = -3,10

    handler = DemHandler(res, grid_origin, h, w, nodata)
    # heightmap = handler.create_heightmap_auto(gt_cloud, gt_dsm)
    heightmap = handler.load_heightmap(gt_dsm)

    occupancy = handler.combined_occupancy(heightmap, minalt, maxalt, 0.6)
    grown = handler.grow_heightmap(heightmap)

    start = (16, 16)
    goal = (30, 120)
    world_path = handler.find_world_path(heightmap, occupancy, start, goal)


if __name__ == '__main__':
    main()
