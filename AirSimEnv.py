import airsim
import numpy as np
import cv2
import math
import time
from DataTools import save_image_gps

PI = math.pi


class AirSimBase:
    def __init__(self):
        self.client = None
        self.camerainfo = None
        self.FOV = None
        self.h, self.w = None, None
        self.Q = None

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

    def setupIntrinsics(self):
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


class AirSimCV(AirSimBase):
    def __init__(self):
        self.client = airsim.VehicleClient()
        self.client.confirmConnection()

        self.setupIntrinsics()

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


class AirSimUAV(AirSimBase):
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.setupIntrinsics()

    def getColInfo(self):
        info = self.client.simGetCollisionInfo()
        collided = info.has_collided
        timestamp = info.time_stamp
        return collided, timestamp

    def getXYZPos(self):
        pos = self.client.simGetGroundTruthKinematics().position
        x, y, z = pos.x_val, -pos.y_val, -pos.z_val  # flip y and z from Airsim coordinate system
        return x, y, z

    def getGeoPos(self, ref_gps):
        reflat, reflon, refalt = ref_gps
        sim_gps = self.client.getGpsData().gnss.geo_point
        sim_lat, sim_lon, alt = sim_gps.latitude, sim_gps.longitude, sim_gps.altitude

        lon = sim_lat - reflat + reflon
        lat = -(sim_lon - reflon) + reflat
        return lat, lon, alt

    def getGeoOrigin(self, ref_gps):
        reflat, reflon, refalt = ref_gps
        sim_gps = self.client.getHomeGeoPoint()
        sim_lat, sim_lon, alt = sim_gps.latitude, sim_gps.longitude, sim_gps.altitude

        lon = sim_lat - reflat + reflon
        lat = -(sim_lon - reflon) + reflat
        return lat, lon, alt

        # latitude == x longitude == y in airsim
        # +latitude == +x; +longitude == -y for me

        # topocentric - x east Y north
        # latitude is y, longitude is x

        # mylong = airsimLat - reflat + reflong
        # mylat  = -(airsimLong - reflong) + reflat

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

    def survey(self, path, angles, data_path, v=5):
        xyz_file = open(data_path + 'xyz.txt', 'w')

        x, y = self.getXYZPos()[:2]
        z = path[0][2]  # fly to survey altitude
        self.moveTo((x, y, z), v*2)

        for i, point in enumerate(path):
            self.moveTo(point, v)
            self.setCameraAngle(angles[i])
            time.sleep(0.5)  # wait a bit
            rgb = self.getRGB()
            gps = self.getGeoPos()
            x, y, z = self.getXYZPos()
            xyz_file.write('rgb%d.jpg %.15f %.15f %.15f\n' % (i, x, y, z))
            lat, lon, alt = gps.latitude, gps.longitude, gps.altitude
            fname = data_path + 'images\\rgb%d.jpg' % i
            save_image_gps(rgb, (lat, lon, alt), fname)

        xyz_file.close()
        print('Survey complete')

    def setTraceLine(self, rgba=(1, 0, 0, 1), thickness=1):
        self.client.simSetTraceLine(rgba, thickness)
