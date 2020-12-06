import airsim
import numpy as np
import cv2
import math


class AirSimEnv:
    def __init__(self):
        self.client = airsim.VehicleClient()
        self.client.confirmConnection()
        self.camerainfo = self.client.simGetCameraInfo(str(0))

        self.FOV = self.camerainfo.fov * math.pi/180  # convert to radians
        self.h, self.w = self.getRGB().shape[:2]
        fl = 0.5*self.w/math.tan(self.FOV/2)  # formula for focal length
        b = 0.25  # baseline from airsim documentation

        self.Q = np.array([
                [1, 0, 0, -self.w/2],
                [0, 1, 0, -self.h/2],
                [0, 0, 0, fl],
                [0, 0, 1/b, 0]
            ])

    def getPosition(self):
        info = self.client.simGetCameraInfo(str(0)).pose.position
        position = (info.x_val, info.y_val, info.z_val)
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
        vector = airsim.Vector3r(x, -y, -z)  # flips z

        if not quat:
            pitch, roll, yaw = orientation
            orientation = airsim.to_quaternion(pitch, roll, yaw)
        camera_pose = airsim.Pose(vector, orientation)
        self.client.simSetCameraPose("0", camera_pose)

    def getRGB(self):
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        rgb = np.frombuffer(responses[0].image_data_uint8, np.uint8)
        rgb = rgb.reshape(responses[0].height, responses[0].width, 3)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb

    def getDisparity(self):
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized, True, False)])
        disp = np.array(responses[0].image_data_float) * responses[0].width
        disp = disp.reshape(responses[0].height, responses[0].width).astype(np.float32)
        return disp
