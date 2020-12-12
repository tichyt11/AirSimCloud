import math
from scipy.spatial.transform import Rotation
import cv2
from Projections import *
import tifffile

import piexif
from PIL import Image

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


def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_gps_txt(path, coords, ref_coords, index=None):  # for ODM
    precision = 1
    reflat, reflon, refalt = ref_coords

    indeces = [0, index, len(coords) - 1] if index else range(len(coords))

    with open(path + 'geo.txt', 'w') as f:
        f.write("EPSG:4326\n")
        for i in indeces:
            x, y, z = coords[i]
            lat, lon, alt = lla_from_topocentric(x, y, z, reflat, reflon, refalt)
            f.write("img%d.png %.15f %.15f %.15f %f %f %f %f %f\n" % (i, lon, lat, alt, 0, 0, 0, precision, precision))


def write_xyz_txt(path, coords, normalize=True, index=None):  # for COLMAP
    indeces = [0, index, len(coords) - 1] if index else range(len(coords))

    mean = coords_mean(coords)

    with open(path + 'xyz.txt', 'w') as f:
        for i in indeces:
            x, y, z = coords[i]
            if normalize:
                x -= mean[0]
                y -= mean[1]
            f.write("img%d.png %.15f %.15f %.15f\n" % (i, x, y, z))


def save_images(path, coords, angles, env):
    for i in range(len(coords)):
        env.setOrientation(angles[i])  # move airsim camera to coords and rotate it
        env.setPosition(coords[i])
        colors = env.getRGB()
        name = "img%d.png" % i
        cv2.imwrite(path + name, cv2.cvtColor(colors, cv2.COLOR_RGB2BGR))  # convert and save


def coords_mean(coords):
    mean = sum(np.array(coords)) / len(coords)
    mean[2] = 0
    return mean


def dec2rat(num):  # TODO: this is temporary solution, it can be more elegant, is there a better way?
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


def write_exifs(path, coords, ref_coords):  # for oMVG and ODM
    reflat, reflon, refalt = ref_coords

    for i in range(len(coords)):
        fname = path + "img%d.png" % i
        img = Image.open(fname)
        x, y, z = coords[i]
        lat, lon, alt = lla_from_topocentric(x, y, z, reflat, reflon, refalt)

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
        img.save('%s' % fname, exif=exif_bytes, quality='keep')


def camera2world_transform(coords, angles):  # angles as pitch, roll, yaw
    pitch, roll, yaw = angles

    flipZY = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # flip z -> height+ <=> z+ and flip y for righthanded coords
    R01 = Rotation.from_euler('ZYX', [yaw, pitch, roll])  # yaw pitch roll
    R01 = np.array(R01.as_matrix())  # from camera coords to world coords
    R12 = np.array(Rotation.from_euler('ZX', [PI/2, PI/2]).as_matrix())  # from image coords to 'camera' (actor) coords

    R = flipZY @ R01 @ R12
    t = np.array(coords)[None].T  # [None] enables transposition

    T = np.concatenate((R, t), axis=1)  # make into a homogeneous transform
    T = np.concatenate((T, np.array([0, 0, 0, 1])[None]))
    return T


def save_disps(coords, angles, env, path):
    for i in range(len(coords)):
        env.setOrientation(angles[i])
        env.setPosition(coords[i])
        disp = env.getDisparity()

        fname = path + 'disp%d.tif' % i
        tifffile.imsave(fname, disp)


def save_rgbs(coords, angles, env, path):
    for i in range(len(coords)):
        env.setOrientation(angles[i])
        env.setPosition(coords[i])
        rgb = env.getRGB()

        fname = path + 'rgb%d.tif' % i
        tifffile.imsave(fname, rgb)


def build_cloud_from_saved(coords, angles, path, size, reproj_matrix, normalized):
    num_images = len(coords)
    total_size = num_images*size
    with open(path + 'point_cloud.ply', 'wb') as out:
        out.write((ply_header % dict(vert_num=total_size)).encode('utf-8'))

        for i in range(num_images):

            disp_file = path + 'disp%d.tif' % i
            disp = tifffile.imread(disp_file)
            rgb_file = path + 'rgb%d.tif' % i
            colors = tifffile.imread(rgb_file)

            if normalized:
                norm_coords = np.array(coords[i]) - coords_mean(coords)
                T = camera2world_transform(norm_coords, angles[i])
            else:
                T = camera2world_transform(coords[i], angles[i])  # prepare transform from camera to world coordinates

            P = T @ reproj_matrix
            verts = cv2.reprojectImageTo3D(disp, P)

            verts = verts.reshape(-1, 3)
            colors = colors.reshape(-1, 3)
            verts = np.hstack([verts, colors])
            np.savetxt(out, verts, fmt='%f %f %f %d %d %d ')  # save points
            print('View number %d done' % i)
        print('Saving point cloud')


def get_cloud(coords, angles, env, path, normalized=False):
    num_images = len(coords)
    print('Collectign data from %d views' % num_images)
    total_size = num_images*env.h*env.w

    with open(path + 'point_cloud.ply', 'wb') as out:
        out.write((ply_header % dict(vert_num=total_size)).encode('utf-8'))

        for i in range(num_images):
            env.setOrientation(angles[i])  # move airsim camera to coords and rotate it
            env.setPosition(coords[i])

            disp = env.getDisparity()  # get image and disparity map from airsim camera
            colors = env.getRGB()

            if normalized:
                norm_coords = np.array(coords[i]) - coords_mean(coords)
                T = camera2world_transform(norm_coords, angles[i])
            else:
                T = camera2world_transform(coords[i], angles[i])  # prepare transform from camera to world coordinates

            P = T @ env.Q
            verts = cv2.reprojectImageTo3D(disp, P)

            verts = verts.reshape(-1, 3)
            colors = colors.reshape(-1, 3)
            verts = np.hstack([verts, colors])
            np.savetxt(out, verts, fmt='%f %f %f %d %d %d ')  # save points
            print('View number %d done' % i)

        print('Saving point cloud')
