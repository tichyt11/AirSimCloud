import airsim
import numpy as np
import cv2
import math
import time
import os
from tools.data_extraction import save_image_gps, camera2world_transform, ply_header
from tools.Distorter import Distorter
from tools import geo


PI = math.pi


class AirSimBase:
    def __init__(self, dist=None):
        self.camerainfo = self.client.simGetCameraInfo(str(0))

        self.FOV = self.camerainfo.fov * math.pi / 180  # convert to radians
        self.h, self.w = self.getRGB().shape[:2]
        fl = 0.5 * self.w / math.tan(self.FOV / 2)  # formula for focal length
        b = 0.25  # baseline from airsim documentation

        self.Q = np.array([
            [1, 0, 0, -self.w / 2],
            [0, 1, 0, -self.h / 2],
            [0, 0, 0, fl],
            [0, 0, 1 / b, 0]
        ])

        self.K = np.array([[fl, 0, self.w / 2],
                           [0, fl, self.h / 2],
                           [0, 0, 1]])

        if not dist:
            self.dist_coeffs = np.array([0.01, 0.05, 0.03, -0.01])
        else:
            self.dist_coeffs = np.array(dist)

        self.Dister = Distorter(self.K, self.dist_coeffs)


    def getRGB(self):
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        rgb = np.frombuffer(responses[0].image_data_uint8, np.uint8)
        rgb = rgb.reshape(responses[0].height, responses[0].width, 3)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb

    def getDisparity(self):
        responses = self.client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized, True, False)])
        disp = np.array(responses[0].image_data_float) * responses[0].width
        disp = disp.reshape(responses[0].height, responses[0].width).astype(np.float32)
        return disp

    def disableLODs(self):
        commands = ['r.forceLOD 0', 'r.forceLODShadow 0', 'foliage.forceLOD 0']
        for cmd in commands:
            self.client.simRunConsoleCommand(cmd)

    def drawPoints(self, coords, rgba=(1, 0, 0, 1), duration=60):
        points = [airsim.Vector3r(x, -y, -z) for x, y, z in coords]
        self.client.simPlotPoints(points, rgba, duration=duration)

    def ping(self):
        start = time.time()
        self.client.ping()
        return time.time() - start


class AirSimCV(AirSimBase):
    def __init__(self):
        self.client = airsim.VehicleClient()
        self.client.confirmConnection()

        super().__init__()

    def getPosition(self):
        info = self.client.simGetCameraInfo(str(0)).pose.position
        position = (info.x_val, -info.y_val, -info.z_val)  # flips z and y
        return position

    def getOrientation(self):
        quaternion = self.client.simGetCameraInfo(str(0)).pose.orientation
        return quaternion

    def setPosition(self, coords):
        quat = self.getOrientation()
        self.setPose(coords, quat, quat=True)

    def setOrientation(self, angles):  # pitch roll yaw
        coords = self.getPosition()
        self.setPose(coords, angles)

    def setPose(self, coords, orientation, quat=False):
        x, y, z = coords
        vector = airsim.Vector3r(x, -y, -z)  # flips z and y

        if not quat:
            pitch, roll, yaw = np.array(orientation)*PI/180  # to radians
            orientation = airsim.to_quaternion(pitch, roll, yaw)
        camera_pose = airsim.Pose(vector, orientation)
        self.client.simSetCameraPose("0", camera_pose)

    def get_cloud(self, coords, angles, path):
        num_images = len(coords)
        print('Collecting data from %d views' % num_images)

        total_size = num_images * self.h * self.w
        self.setPose(coords[0], angles[0])  # init position

        with open(path + 'GT.ply', 'wb') as out:
            out.write((ply_header % dict(vert_num=total_size)).encode('utf-8'))

            for i in range(num_images):
                time.sleep(0.3)
                disp = self.getDisparity()  # get image and disparity map from airsim camera
                colors = self.getRGB()
                if i + 1 < num_images:
                    self.setPose(coords[i + 1],
                                angles[i + 1])  # move right after taking data, for depth to load properly

                T = camera2world_transform(coords[i], angles[i])  # prepare transform from camera to world coordinates
                P = T @ self.Q
                verts = cv2.reprojectImageTo3D(disp, P)

                verts = verts.reshape(-1, 3)
                colors = colors.reshape(-1, 3)
                verts = np.hstack([verts, colors])
                np.savetxt(out, verts, fmt='%f %f %f %d %d %d ')  # save points
                print('View number %d done' % i)

            print('Saving point cloud')

    def save_disps(self, coords, angles, disp_dir):  # save disparities as tif images
        for i in range(len(coords)):
            self.setPose(coords[i], angles[i])  # move airsim camera to coords and rotate it
            time.sleep(0.3)  # wait for rendering
            disp = self.getDisparity()

            fname = os.path.join(disp_dir, 'disp%d.tif' % i)
            cv2.imwrite(fname, disp)

    def save_rgbs(self, coords, angles, image_dir, fmt='jpg'):  # save rgb images
        for i in range(len(coords)):
            self.setPose(coords[i], angles[i])  # move airsim camera to coords and rotate it
            time.sleep(0.3)  # wait for rendering
            rgb = self.getRGB()
            fname = os.path.join(image_dir, 'rgb%d.%s' % (i, fmt))
            cv2.imwrite(fname, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))  # convert and save
            print('Saved image %d' % i)

    def save_rgbs_gps(self, coords, angles, image_dir, ref_origin, fmt='jpg'):
        reflat, reflon, refalt = ref_origin

        for i in range(len(coords)):
            self.setPose(coords[i], angles[i])  # move airsim camera to coords and rotate it
            time.sleep(0.3)  # wait for rendering
            rgb = self.getRGB()
            x, y, z = coords[i]
            gps = geo.lla_from_topocentric(x, y, z, reflat, reflon, refalt)
            fname = os.path.join(image_dir, 'rgb%d.%s' % (i, fmt))
            save_image_gps(rgb, gps, fname)
            print('Saved image %d' % i)


class AirSimUAV(AirSimBase):
    def __init__(self, gpsref=(3, 3, 0)):

        self.GPSref = gpsref

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        super().__init__()

    def getColInfo(self):
        info = self.client.simGetCollisionInfo()
        collided = info.has_collided
        # timestamp = info.time_stamp
        return collided

    def getXYZpos(self):
        pos = self.client.simGetGroundTruthKinematics().position
        x, y, z = pos.x_val, -pos.y_val, -pos.z_val  # flip y and z from Airsim coordinate system
        return x, y, z

    def getcameraXYZpos(self):
        pos = self.client.simGetCameraInfo().position  # camera has different cooridnates than the drone
        x, y, z = pos.x_val, -pos.y_val, -pos.z_val  # flip y and z from Airsim coordinate system
        return x, y, z

    def getGPSpos(self):
        reflat, reflon, refalt = self.GPSref
        sim_gps = self.client.getGpsData().gnss.geo_point
        sim_lat, sim_lon, alt = sim_gps.latitude, sim_gps.longitude, sim_gps.altitude

        lon = sim_lat - reflat + reflon  # AirSim lat becomes my lon
        lat = -(sim_lon - reflon) + reflat  # AirSim -lon becomes my lat
        return lat, lon, alt

    def getGPShome(self):
        reflat, reflon, refalt = self.GPSref
        sim_gps = self.client.getHomeGeoPoint()
        sim_lat, sim_lon, alt = sim_gps.latitude, sim_gps.longitude, sim_gps.altitude

        lon = sim_lat - reflat + reflon  # AirSim lat becomes my lon
        lat = -(sim_lon - reflon) + reflat  # AirSim -lon becomes my lat
        return lat, lon, alt

    def takeOff(self):
        self.client.takeoffAsync().join()
        self.hover()

    def hover(self):
        self.client.hoverAsync().join()

    def setCameraAngle(self, orientation):
        pitch, roll, yaw = np.array(orientation)*PI/180
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(pitch, roll, yaw))
        self.client.simSetCameraPose("0", camera_pose)

    def moveTo(self, coords, v=5, wait=True):
        x, y, z = coords
        y, z = -y, -z  # flip y and z into Airsim coordinate system
        if wait:
            self.client.moveToPositionAsync(x, y, z, v).join()
        else:
            self.client.moveToPositionAsync(x, y, z, v)

    def moveByVelocity(self, vx, vy, vz, t):
        vy = -vy
        vz = -vz
        return self.client.moveByVelocityAsync(vx, vy, vz, t).join()

    def moveFromTo(self, fromcoords, tocoords, v=10):
        x0, y0, z0 = fromcoords
        x, y, z = tocoords
        delta_x = x - x0
        delta_y = y - y0
        delta_z = z - z0
        t = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)/v
        vx = delta_x / t
        vy = delta_y / t
        vz = delta_z / t
        self.moveByVelocity(vx, vy, vz, t)
        self.moveByVelocity(0, 0, 0, v)  # settle
        self.hover()

    def moveOnPath(self, path, v=5, wait=True):
        trip_time = 1000000  # temp value
        lookahead = -1
        airsim_path = [airsim.Vector3r(x, -y, -z) for x, y, z in path]  # flip y and z into Airsim coordinate system
        if wait:
            self.client.moveOnPathAsync(airsim_path, v, trip_time, airsim.DrivetrainType.ForwardOnly,
                                        airsim.YawMode(False, 0), lookahead, 1).join()
        else:
            self.client.moveOnPathAsync(airsim_path, v, trip_time, airsim.DrivetrainType.ForwardOnly,
                                        airsim.YawMode(False, 0), lookahead, 1)

    def survey(self, waypoints, gimbal_angle, data_dir, v=5, distort=False):
        if not os.path.exists(data_dir):
            print('Warning: Creating %s' % data_dir)
            os.makedirs(data_dir)

        image_dir = os.path.join(data_dir, 'images')
        if not os.path.exists(image_dir):
            print('Warning: Creating %s' % image_dir)
            os.makedirs(image_dir)

        dist_dir = os.path.join(data_dir, 'distorted')
        if distort and not os.path.exists(dist_dir):
            print('Warning: Creating %s' % dist_dir)
            os.makedirs(dist_dir)

        img_xyz = os.path.join(image_dir, 'xyz.txt')
        dist_xyz = os.path.join(dist_dir, 'xyz.txt')
        reflat, reflon, refalt = self.GPSref
        img_xyz = open(img_xyz, 'w')
        if distort:
            dist_xyz = open(dist_xyz, 'w')

        x, y = self.getXYZpos()[:2]
        z = waypoints[0][2]
        self.setCameraAngle(gimbal_angle)
        self.moveTo((x, y, z), v*2)  # fly to survey altitude

        for i, point in enumerate(waypoints):

            self.moveFromTo(self.getXYZpos(), point, v)
            # self.moveTo(point, v)
            self.setCameraAngle(gimbal_angle)  # set camera gimbal angle
            rgb = self.getRGB()

            gps = self.getGPSpos()
            lat, lon, alt = gps
            # x, y, z = self.getXYZpos()  # coordinates could be inconsistent with gps due to time delay
            x, y, z = geo.topocentric_from_lla(lat, lon, alt, reflat, reflon, refalt)

            img_xyz.write('rgb%d.jpg %.15f %.15f %.15f\n' % (i, x, y, z))  # write xyz file for colmap
            fname = os.path.join(image_dir, 'rgb%d.jpg' % i)
            save_image_gps(rgb, (lat, lon, alt), fname)  # save gps into EXIF for ODM and oMVG
            if distort:  # distortion enabled
                dist_img = self.Dister.distort_image(rgb)
                dist_xyz.write('rgb%d.jpg %.15f %.15f %.15f\n' % (i, x, y, z))
                dist_name = os.path.join(dist_dir, 'dist_rgb%d.jpg' % i)
                save_image_gps(dist_img, (lat, lon, alt), dist_name)  # save gps into EXIF for ODM and oMVG
            print('Info: image %d out of %d saved' % (i, len(waypoints)))

        img_xyz.close()
        if distort:
            dist_xyz.close()
        print('Survey complete')

    def land(self):
        self.client.landAsync().join()
        self.client.armDisarm(False)

    def setTraceLine(self, rgba=(1, 0, 0, 1), thickness=1):
        self.client.simSetTraceLine(rgba, thickness)
