import math
from scipy.spatial.transform import Rotation
import cv2
from tools.geo import *
import tifffile
import piexif
from PIL import Image
import os
from tools.distortion import distort_image

PI = math.pi

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        print('Warning: Creating %s directory' % dir_name)
        os.mkdir(dir_name)


def coords2gps_txt(path, coords, ref_coords, index=None, form='jpg', precision=0.001):  # for ODM
    reflat, reflon, refalt = ref_coords

    indeces = [0, index, len(coords) - 1] if index else range(len(coords))

    with open(path + 'geo.txt', 'w') as f:
        f.write("EPSG:4326\n")
        for i in indeces:
            x, y, z = coords[i]
            lat, lon, alt = lla_from_topocentric(x, y, z, reflat, reflon, refalt)
            f.write("rgb%d.%s %.15f %.15f %.15f %f %f %f %f %f\n" % (i, form, lon, lat, alt, 0, 0, 0, precision, precision))


def coords2xyz_txt(path, coords, index=None, form='jpg'):  # for COLMAP
    indeces = [0, index, len(coords) - 1] if index else range(len(coords))

    with open(path + 'xyz.txt', 'w') as f:
        for i in indeces:
            x, y, z = coords[i]
            f.write("rgb%d.%s %.15f %.15f %.15f\n" % (i, form, x, y, z))


def coords_mean(coords):
    mean = sum(np.array(coords)) / len(coords)
    mean[2] = 0
    return mean


def dec2rat(num):  # write decimal float as 3 rational numbers (6* int16)
    f1 = 1 << 10
    f2 = 1 << 10

    x1 = num*f1
    degs = math.floor(x1)

    x2 = (x1 - degs)*f2*60
    mins = math.floor(x2)

    x3 = (x2 - mins)*60
    secs = math.floor(x3)
    return (degs, f1), (mins, f1*f2), (secs, f1*f2)


def rat2dec(rat):
    (degs, f1) = rat[0]
    (mins, f2) = rat[1]
    (secs, f3) = rat[2]
    return degs/f1 + mins/f2/60 + secs/f3/3600


def create_exif_bytes(lat, lon, alt):  # create exif for Pillow image
    zeroth_ifd = {
                piexif.ImageIFD.XResolution: (1920, 1),
                piexif.ImageIFD.YResolution: (1080, 1)
            }
    gps_ifd = {
        piexif.GPSIFD.GPSVersionID: (2, 2, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: 'N' if lat > 0 else 'S',
        piexif.GPSIFD.GPSLongitudeRef: 'E' if lon > 0 else 'W',
        piexif.GPSIFD.GPSAltitudeRef: 0 if alt > 0 else 1,
        piexif.GPSIFD.GPSLatitude: (dec2rat(abs(lat))),
        piexif.GPSIFD.GPSLongitude: (dec2rat(abs(lon))),
        piexif.GPSIFD.GPSAltitude: (math.floor(abs(alt*1024)), 1024),  # multiply by 1024 for mm resolution
        piexif.GPSIFD.GPSDOP: (1, 1024)
    }

    exif_dict = {"0th": zeroth_ifd, "GPS": gps_ifd}
    exif_bytes = piexif.dump(exif_dict)
    return exif_bytes


def save_image_gps(image_array, gps, fname):
    lat, lon, alt = gps
    exif_bytes = create_exif_bytes(lat, lon, alt)
    img = Image.fromarray(image_array)
    img.save(fname, exif=exif_bytes, quality=95)


def camera2world_transform(coords, angles):  # angles as pitch, roll, yaw
    pitch, roll, yaw = angles

    flipZY = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # flip z -> height+ <=> z+ and flip y for righthanded coords
    R01 = np.array(Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix())  # yaw pitch roll from camera to world
    R12 = np.array(Rotation.from_euler('ZX', [90, 90], degrees=True).as_matrix())  # from image coords to (actor) coords for camera facing downwards

    R = flipZY @ R01 @ R12
    t = np.array(coords)[None].T  # [None] enables transposition

    T = np.concatenate((R, t), axis=1)  # make into a homogeneous transform
    T = np.concatenate((T, np.array([0, 0, 0, 1])[None]))
    return T


def build_cloud_from_saved(coords_list, angles_list, path, size, reproj_matrix, form='jpg'):  # for separate depth and color
    num_images = len(coords_list)
    total_size = num_images*size
    with open(path + 'GT.ply', 'wb') as out:
        out.write((ply_header % dict(vert_num=total_size)).encode('utf-8'))

        for i in range(num_images):

            disp_file = path + 'disps\\disp%d.tif' % i
            disp = tifffile.imread(disp_file)
            rgb_file = path + 'images\\rgb%d.%s' % (i, form)
            colors = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)

            T = camera2world_transform(coords_list[i], angles_list[i])  # prepare transform from camera to world coordinates
            P = T @ reproj_matrix
            verts = cv2.reprojectImageTo3D(disp, P)

            verts = verts.reshape(-1, 3)
            colors = colors.reshape(-1, 3)
            verts = np.hstack([verts, colors])
            np.savetxt(out, verts, fmt='%f %f %f %d %d %d ')  # save points
            print('View number %d done' % i)
        print('Saving point cloud')


def airsim_format_rec(rec_dir, gps_ref=(3, 3, 0), distort_map=None, precision=0.001):
    rec_txt = os.path.join(rec_dir, 'airsim_rec.txt')
    xyz_txt = os.path.join(rec_dir, 'xyz.txt')
    image_dir = os.path.join(rec_dir, 'images')
    new_image_dir = os.path.join(rec_dir, 'exif_images')
    create_dir(new_image_dir)

    if distort_map is not None:
        dist_dir = os.path.join(rec_dir, 'distorted')
        create_dir(dist_dir)

    reflat, reflon, refalt = gps_ref
    with open(rec_txt, 'r') as f:
        f.readline()  # throw away header line
        lines = f.readlines()

    xyz = open(xyz_txt, 'w')
    for i, line in enumerate(lines):
        data = line.split('\t')
        old_image_name = data[-1].strip('\n')
        old_image_path = os.path.join(image_dir, old_image_name)
        img = cv2.cvtColor(cv2.imread(old_image_path), cv2.COLOR_BGR2RGB)  # load recorded image

        new_image_name = 'rgb%d.jpg' % i
        new_image_path = os.path.join(new_image_dir, new_image_name)

        x, y, z = list(map(float, data[1:4]))  # x,y,z coordinates of the camera
        lat, lon, alt = lla_from_topocentric(x, -y, -z, reflat, reflon, refalt)  # transform ENU to WGS

        if distort_map is not None:
            distorted_image_path = os.path.join(dist_dir, new_image_name)
            dist_img = distort_image(img, distort_map)
            save_image_gps(dist_img, (lat, lon, alt), distorted_image_path)  # save with exif gps

        save_image_gps(img, (lat, lon, alt), new_image_path)  # save the image with exif gps
        xyz.write("%s %.15f %.15f %.15f\n" % (new_image_name, x, -y, -z))  # write down camera coordinates

    xyz.close()
