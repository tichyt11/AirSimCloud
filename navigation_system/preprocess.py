import subprocess
import os
import math
import sys
import json
import numpy as np
from tools.geo import ecef_from_topocentric_transform
from tools.dem_handling import DemHandler
import pdal
import tifffile
import getopt

pdal_transform_cmd = '''[
    "%s",
    {
        "type":"filters.transformation",
        "matrix":"%s"
    },
    {
        "type":"writers.las",
        "filename":"%s"
    }
]'''

def mkdir_ine(dirname):
    """Create the folder if it does not exist"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def check_file(fname):
    return os.path.exists(fname)

def adjust_camera_params(database_file, intrinsics):
    with open(database_file) as f:
        database = json.load(f)
    intr = database['intrinsics'][0]['value']['ptr_wrapper']['data']
    intr['value0']['focal_length'] = intrinsics[0]
    intr['disto_t2'] = intrinsics[1:]
    database['intrinsics'][0]['value']['ptr_wrapper']['data'] = intr
    with open(database_file, 'w') as f:
        json.dump(database, f, indent=4)
    print('Camera model inserted into openMVG database')


def parse_transform(t_file, ref_coords):
    reflat, reflon, refalt = ref_coords  # ref lat, lon and alt of topocentric origin
    Tec = ecef_from_topocentric_transform(reflat, reflon, refalt)  # transform matrix from ECEF to topocentric coordinates
    Tinv_m = np.linalg.inv(Tec)

    rot = np.zeros((3, 3))
    t = np.zeros((3, 1))

    with open(t_file, 'r') as f:
        lines = f.readlines()
    s = float(lines[5].lstrip(' scale: '))
    rot[0, :] = list(map(float, lines[7].split()))
    rot[1, :] = list(map(float, lines[8].split()))
    rot[2, :] = list(map(float, lines[9].split()))
    for i in range(len(t)):
        t[i] = float(lines[10].lstrip(' translation: ').split('  ')[i])

    scale_m = np.array([[s, 0, 0, 0],
                        [0, s, 0, 0],
                        [0, 0, s, 0],
                        [0, 0, 0, 1]])  # scaling matrix

    conc = np.array([[0, 0, 0, 1]])
    T = np.concatenate((rot, t), 1)
    T_m = np.concatenate((T, conc))  # homogeneous transform matrix

    transform_m = Tinv_m @ T_m @ scale_m
    print('Transformation matrix built')
    return transform_m


def apply_transform(in_file, out_file, transform_m):
    transform_string = ''
    for i in transform_m.flatten():
        transform_string += '%s ' % str(i) 
    transform_string = transform_string.rstrip(' ')
    cmd = pdal_transform_cmd % (in_file, transform_string, out_file)
    pipeline = pdal.Pipeline(cmd)
    pipeline.execute()    
    print('Transform applied')

def save_tif(fname, array):
    print('Saving %s' % (fname))
    tifffile.imsave(fname, array)

def save_metadata(fname, Handler):
    res = Handler.res
    org = Handler.grid_origin
    h = Handler.h
    w = Handler.w
    params = res, org[0], org[1], h, w
    with open(fname, 'w') as f:
        for p in params:
            f.write('%s\n' % str(p))

def load_calibration(calibration_file):
    intrinsics = [] # focal, k1, k2, k3, p1, p2 
    with open(calibration_file, 'r') as f:
        for line in f.readlines():
            intrinsics.append(float(line))
    return intrinsics


def main(argv):

    images_dir = None
    output_dir = None
    work_dir = None
    calibration_file = None

    try:
        opts, args = getopt.getopt(argv,'ho:s:g:c:')
    except getopt.GetoptError:
        print('Arguments parsing error')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('preprocess.py -i <imagedir> -o <outdir> -w <workdir>')
            sys.exit()
        elif opt in ('-o'):
            output_dir = arg
        elif opt in ('-i'):
            images_dir = arg
        elif opt in ('-w'):
            work_dir = arg
        elif opt in ('-c'):
            calibration_file = arg

    cwd = os.getcwd()

    if images_dir is None:
        images_dir = os.path.join(cwd, 'images')
        if not check_file(images_dir):
            print('Image dir %s does not exist' % images_dir)
    if output_dir is None:
        output_dir = os.path.join(cwd, 'output')
    if work_dir is None:
        work_dir = os.path.join(cwd, 'workdir')

    #-----------------------------------------------------------------------

    camera_database_path = os.path.join(cwd, 'sensor_width_camera_database.txt')

    database_file = os.path.join(work_dir, 'sfm_data.json')
    transform_file = os.path.join(work_dir, 'transform.txt')
    sfm_log_file = os.path.join(output_dir, 'SfMlog.txt')

    matches_path = os.path.join(work_dir, 'matches')
    SfM_path = os.path.join(work_dir, 'SfM')
    IncrementalSfM_path = os.path.join(SfM_path, 'sfm_data.bin')
    GeoSfM_path = os.path.join(SfM_path, 'sfm_data_adjusted.bin')
    NormSfM_path = os.path.join(SfM_path, 'sfm_data_local.bin')

    MVS_path = os.path.join(work_dir, 'MVS')
    MVS_images = os.path.join(MVS_path, 'images')
    MVS_scene_path = os.path.join(MVS_path, 'scene.mvs')
    cloud_path = os.path.join(output_dir, 'cloud.ply')

    transformed_cloud_path = os.path.join(output_dir, 'transformed_cloud.las')
    heightmap_path = os.path.join(output_dir, 'heightmap.tif')
    grown_heightmap_path = os.path.join(output_dir, 'grown_heightmap.tif')
    occupancy_path = os.path.join(output_dir, 'occupancy_grid.tif')
    metadata_path = os.path.join(output_dir, 'metadata.txt')
    generated_path_path = os.path.join(output_dir, 'generated_path_coordinates.txt')


    #-----------------------------------------------------------------------
    # SETTINGS Change these to camera intrinsics, desired resolution etc.

    generate_cloud = False

    camera_model = 4  # 1 = PINHOLE, 4 = k1, k2, k3, t1, t2
    intrinsics = 960, 0.01, 0.05, -0.06, 0.002, -0.001 # f, k1, k2, k3, t1, t2

    if calibration_file is not None and check_file(calibration_file):
        intrinsics = load_calibration(calibration_file)

    cloud_resolution = 1 # 0 = fullsize, 1 = half size, ... , 3 = 1/8 size

    dem_auto_size = True
    dem_resolution = 0.3  # in meters
    ref_coords = (reflat, reflon, refalt) = (3, 3, 0)  # ref lat, lon and alt (GPS) coordinates of point cloud origin (0,0,0)
    dem_grid_origin = (-20,-40)  # world coordinates of lower left corner of DEM
    dem_area = (100, 100)  # height and width of area in meters (height in x+ direction, width in y+ direction)

    dem_grow_size = 2 + math.ceil(math.sqrt(0.5)/dem_resolution)  # grow the max Z value of heightmap by this radius
    min_alt, max_alt = -3, 10  # minimum and maximum terrain height allowed (with respect to ref_coords alt)
    max_grad = 2*dem_resolution  # maximum allowed spatial rate of change of terrain


    #-----------------------------------------------------------------------
    # Commands for openMVG and openMVS
    Initdatabase_cmd = 'openMVG_main_SfMInit_ImageListing -i %s -d %s -o %s -c %d -f %f -m 1 -P' % (images_dir, camera_database_path, work_dir, camera_model, intrinsics[0])
    SfMpipeline_cmd = [
        'openMVG_main_ComputeFeatures -i %s/sfm_data.json -o %s' % (work_dir, matches_path),
        'openMVG_main_ComputeMatches -i %s/sfm_data.json -o %s' % (work_dir, matches_path),
        'openMVG_main_IncrementalSfM -i %s/sfm_data.json -m %s -o %s -f NONE ' % (work_dir, matches_path, SfM_path),
        'openMVG_main_ChangeLocalOrigin -i %s -o %s -l "0;0;0" ' % (IncrementalSfM_path, SfM_path),
    ]
    GeoRegistration_cmd = 'openMVG_main_geodesy_registration_to_gps_position -i %s -o %s' % (NormSfM_path, GeoSfM_path)
    MVSpipeline_cmd = [
        'openMVG_main_openMVG2openMVS -i %s -o %s -d %s' % (NormSfM_path, MVS_scene_path, MVS_images),
        'DensifyPointCloud %s -w %s -o %s --resolution-level %d' % (MVS_scene_path, MVS_path, cloud_path, cloud_resolution)
    ]

    #-----------------------------------------------------------------------
    # Pipeline starts here
    if generate_cloud:
        subprocess.run(Initdatabase_cmd, shell=True)  # init database
        adjust_camera_params(database_file, intrinsics) # put camera parameters from this file into openMVG database

        mkdir_ine(output_dir)
        print('Starting SfM pipeline (OpenMVG)')
        with open(sfm_log_file, 'w') as log:
            for i,cmd in enumerate(SfMpipeline_cmd):
                subprocess.run(cmd, shell=True, stdout=log)
                print('SfM stage %d finished' % i)
        print('SfM done')

        with open(transform_file, 'w') as f:  # write transform into text file
            subprocess.run(GeoRegistration_cmd, shell=True, stdout=f)

        print('Starting MVS pipeline (OpenMVS)')
        mkdir_ine(MVS_images)
        for cmd in MVSpipeline_cmd:
            subprocess.run(cmd, shell=True)
        print('MVS done')

        transform_matrix = parse_transform(transform_file, ref_coords)
        apply_transform(cloud_path, transformed_cloud_path, transform_matrix)


    if dem_auto_size:
        Handler = DemHandler(dem_resolution)
    else:
        h, w = math.ceil(dem_area[0]/res), math.ceil(dem_area[1]/res)
        Handler = DemHandler(dem_resolution, dem_grid_origin, h, w)

    print('Generating heightmap')
    heightmap = Handler.create_heightmap_auto(transformed_cloud_path, heightmap_path)
    print('Growing heightmap')
    grown_heightmap = Handler.grow_heightmap(heightmap, dem_grow_size)
    print('Generating occupancy grid')
    occupancy_grid = Handler.combined_occupancy(heightmap, min_alt, max_alt, max_grad, dem_grow_size)
    save_tif(grown_heightmap_path, grown_heightmap)
    save_tif(occupancy_path, occupancy_grid)
    print('Saving metadata')
    save_metadata(metadata_path, Handler)
        
if __name__=='__main__':
    main(sys.argv[1:])
