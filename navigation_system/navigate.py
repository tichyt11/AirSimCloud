import subprocess
import os
import getopt
import math
import sys
import json
import numpy as np
from tools.geo import ecef_from_topocentric_transform
from tools.dem_handling import DemHandler
import pdal
import tifffile

def check_file(fname):
    return os.path.exists(fname)

def load_tif(fname):
    assert check_file(fname)
    print('Loading %s' % fname)
    array = tifffile.imread(fname)
    return array

def load_metadata(fname):
    assert check_file(fname)
    org = [None, None]
    with open(fname, 'r') as f:
        params = [float(p.strip('\n')) for p in f.readlines()]
    res, org[0], org[1], h, w = params
    return res, org, h, w

def save_path(fname, path):
    print('Saving path to %s' % fname)
    with open(fname, 'w') as f:
        for point in path:
            f.write('%f, %f, %f\n' % point)

def check_obstacles_world(coords, occupancy, Handler):
    row, col = Handler.world2image(coords[0], coords[1])
    if not (row >= 0 and row < Handler.h): 
        print('Y coordinate %f is outside bounds.' % coords[1])
        return 1
    if not (col >= 0 and col < Handler.w):
        print('X coordinate %f is outside bounds.' % coords[0])
        return 1
    if occupancy[row, col]:
        print('There is an obstacle at coordinates [%f, %f].' % (coords[0], coords[1]))
        return 1
    return 0

def main(argv):

    start_coords = None
    goal_coords = None

    try:
        opts, args = getopt.getopt(argv,'ho:s:g:')
    except getopt.GetoptError:
        print('Arguments parsing error')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('navigate.py -o pathfile.txt')
            sys.exit()
        elif opt in ('-o'):
            generated_path_path = arg
        elif opt in ('-s'):
            arg = arg.split(',')
            start_coords = [float(arg[0]), float(arg[1])] 
        elif opt in ('-g'):
            arg = arg.split(',')
            goal_coords = [float(arg[0]), float(arg[1])] 

    #-----------------------------------------------------------------------

    cwd = os.getcwd()

    work_dir = os.path.join(cwd, 'workdir')
    output_dir = os.path.join(cwd, 'output')

    transformed_cloud_path = os.path.join(output_dir, 'transformed_cloud.las')
    heightmap_path = os.path.join(output_dir, 'heightmap.tif')
    grown_heightmap_path = os.path.join(output_dir, 'grown_heightmap.tif')
    occupancy_path = os.path.join(output_dir, 'occupancy_grid.tif')
    metadata_path = os.path.join(output_dir, 'metadata.txt')

    generated_path_path = os.path.join(output_dir, 'generated_path_coordinates.txt')

    #-----------------------------------------------------------------------
    

    print('Loading heightmap')
    grown_heightmap = load_tif(grown_heightmap_path)
    print('Loading occupancy grid')
    occupancy_grid = load_tif(occupancy_path)
    print('Loading metadata')
    res, org, h, w = load_metadata(metadata_path)
    Handler = DemHandler(res, org, h, w)
    print('Grid map spans from [%f, %f] to [%f, %f].' % (org[0], org[1], Handler.image2world(0,w)[0], Handler.image2world(0,w)[1]))
    
    path = None
    while start_coords is None:  # choose start coords
        start_coords = (float(input('Input starting x coordinate: ')), float(input('Input starting y coordinate: ')))
        if check_obstacles_world(start_coords, occupancy_grid, Handler):
            start_coords = None
            print('Input a different start point, please.')
        
    while goal_coords is None:  # choose goal coords
        goal_coords = (float(input('Input goal x coordinate: ')), float(input('Input goal y coordinate: ')))
        if check_obstacles_world(goal_coords, occupancy_grid, Handler):
            goal_coords = None
            print('Input a different goal point, please.')

    print('Looking for path between [%f, %f] and [%f, %f]' % (start_coords[0], start_coords[1], goal_coords[0], goal_coords[1]))
    path = Handler.find_world_path(grown_heightmap, occupancy_grid, start_coords, goal_coords)
    if path is None:
        print('There is no path between these points, try again')
    else:
        print('Path has been found!')
        save_path(generated_path_path, path)
            
if __name__=='__main__':
    main(sys.argv[1:])
