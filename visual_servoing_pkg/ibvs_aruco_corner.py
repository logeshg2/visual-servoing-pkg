#!/usr/bin/env python3

import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from visual_servoing_pkg.msg import ArucoCorner
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from ComDependencies.robot_controller import robot


class IBVS_aruco(Node):
    def __init__(self):
        super().__init__("ibvs_aruco_node")

        # aruco variables
        self.cur_top_left = None
        self.cur_top_right = None
        self.cur_bottom_right = None
        self.cur_bottom_left = None
        # target aruco points
        self.tar_top_left = None
        self.tar_top_right = None
        self.tar_bottom_right = None
        self.tar_bottom_left = None

        # image jacobian | velocity variables
        self.pixelVel_gain = 0.1
        self.pixelVel = None
        self.imgJacob = None
        self.ee_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.curCamVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.prevCamVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # camera properties
        self.fx = self.fy = 950.0
        self.cx = 320.0
        self.cy = 240.0
        self.Z = 1000       # distance from camera to target (assuming it is 1m away) - this is point depth

        # robot arm controllers
        self.bot = robot("192.168.1.9")
        self.triggered = False

        # PID control (TODO: tune this)
        self.KPX = 2*(0.0001)
        self.KIX = 1*(0.000001)
        self.KDX = 0*(0.00001)
        self.KPY = 2*(0.0001)
        self.KIY = 1*(0.000001)
        self.KDY = 0*(0.00001)
        self.KPZ = 2*(0.0001)
        self.KIZ = 1*(0.000001)
        self.KDZ = 0*(0.00001)

        # ros2 comm variables
        self.vel_gen_group = MutuallyExclusiveCallbackGroup()
        self.corner_sub = self.create_subscription(ArucoCorner, "/aruco_corners", self.corners_sub_cb, 10, self.vel_gen_group)
        self.inc_srv_trig = self.create_service(SetBool, '/trigger_servoing', self.trigger_servoing_cb)
        self.main_timer = self.create_timer(1/20, self.main_timer_cb, self.vel_gen_group)


    def trigger_servoing_cb(self, request, response):
        if (request.data):
            self.triggered = True
        else:
            self.triggered = False
        
        # go to tracking position
        self.bot.write_cartesian_position(coords=self.tracking_pose, blocking=False)
        time.sleep(2)
        while (self.bot.is_moving()):
            time.sleep(0.1)

        response.success = True
        response.message = "trigger successful"
        return response

    def computeImgPointJacobian(self, u, v, Z = 1000):
        """
        Function 'computeImgPointJacobian' is used to compute image jacobian or interaction matrix (J) of the given pixel point (u, v).
        """
        # compute image coordinates (x, y) from (u, v)
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        Z = Z      # point (x, y) depth (in world frame)

        # image jacobian template(or formula) - 2x6
        img_jacobian = np.array([[(-1/Z), 0, (x/Z), (x*y), -(1+(x*x)), y], 
                                 [0, (-1/Z), (y/Z), (1+(y*y)), (-x*y), -x]])
        
        return img_jacobian

    def corners_sub_cb(self, msg):
        if (msg.top_left is not None):
            self.cur_top_left = msg.top_left
            self.cur_top_right = msg.top_right
            self.cur_bottom_right = msg.bottom_right
            self.cur_bottom_left = msg.bottom_left
        else:
            self.cur_top_left = None
            self.cur_top_right = None
            self.cur_bottom_right = None
            self.cur_bottom_left = None

    def computePixelPointVel(self, des_point, cur_point):
        """
        function to compute pixel velocity between two points.
        function use `self.pixelVel_gain` to adjust the velocity scale.
        """

        pix_vel = self.pixelVel_gain * (np.subtract(np.array(des_point), np.array(cur_point)))
        return pix_vel

    def computeCamVel(self):
        # compute pixel velocity
        top_left_pix_vel = self.computePixelPointVel(self.tar_top_left, self.cur_top_left)
        top_right_pix_vel = self.computePixelPointVel(self.tar_top_right, self.cur_top_right)
        bottom_right_pix_vel = self.computePixelPointVel(self.tar_bottom_right, self.cur_bottom_right)
        bottom_left_pix_vel = self.computePixelPointVel(self.tar_bottom_left, self.cur_bottom_left)
        # flatten pixel velocities (here - 8x1 vector)
        self.pixelVel = np.array([top_left_pix_vel.flatten(), top_right_pix_vel.flatten(), bottom_right_pix_vel.flatten(), bottom_left_pix_vel.flatten()])
        self.pixelVel = self.pixelVel.flatten().T     # [[u1_dot], [v1_dot], [u2_dot], [v2_dot], [u3_dot], [v3_dot], [u4_dot], [v4_dot]] - 8x1

        # compute jacobian matrix (combain all for points interation matrix)
        top_left_jacob = self.computeImgPointJacobian(self.cur_top_left[0], self.cur_top_left[1])
        top_right_jacob = self.computeImgPointJacobian(self.cur_top_right[0], self.cur_top_right[1])
        bottom_right_jacob = self.computeImgPointJacobian(self.cur_bottom_right[0], self.cur_bottom_right[1])
        bottom_left_jacob = self.computeImgPointJacobian(self.cur_bottom_left[0], self.cur_bottom_left[1])
        # points jacobian (for all for points) - 8x6 matrix
        pointsJacob = np.vstack([top_left_jacob, top_right_jacob, bottom_right_jacob, bottom_left_jacob])
        # make it as 8x8 square matrix - by adding 2 zeros at end on each row
        np.append(pointsJacob[0], [0.0, 0.0])
        np.append(pointsJacob[1], [0.0, 0.0])
        np.append(pointsJacob[2], [0.0, 0.0])
        np.append(pointsJacob[3], [0.0, 0.0])
        np.append(pointsJacob[4], [0.0, 0.0])
        np.append(pointsJacob[5], [0.0, 0.0])
        np.append(pointsJacob[6], [0.0, 0.0])
        np.append(pointsJacob[7], [0.0, 0.0])

        # compute camVel 
        # camVel = inv(pointsJacobian) @ pixelVel
        camVel = np.linalg.pinv(pointsJacob) @ self.pixelVel        # (8x1) = (8x8) @ (8x1)
        # NOTE: `camVel.flatten()` -> [Vx, Vy, Vz, Wx, Wy, Wz, 0, 0]

        return camVel[:6].flatten()
    
    def computeEEVel(self):
        # check aruco corners detection
        if (self.cur_top_left is not None):
            # compute desired camera velocity
            self.curCamVel = self.computeCamVel()
            self.get_logger().info(f"Computed Cam velocity: {self.curCamVel}")

            # for now - lets servo only on x, y, and z (or 3D servoing)
            self.vel_error = np.subtract(self.curCamVel, self.prevCamVel)
            self.ee_vel[0] = (self.vel_error[0] * self.KPX) + (self.vel_error[0] * self.KIX) + (self.vel_error[0] * self.KDX)
            self.ee_vel[1] = (self.vel_error[1] * self.KPY) + (self.vel_error[1] * self.KIY) + (self.vel_error[1] * self.KDY)
            self.ee_vel[2] = (self.vel_error[2] * self.KPZ) + (self.vel_error[2] * self.KIZ) + (self.vel_error[2] * self.KDZ)
            self.prevCamVel = self.curCamVel.copy()
        else:
            self.ee_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def main_timer_cb(self):
        pass


def main():
    rclpy.init()

    try:
        node = IBVS_aruco()
        rclpy.spin(node)
    except Exception as e:
        print(f"Shutting down IBVS node:\nException: {e}")
        node.destroy_node()
        # rclpy.shutdown()

if __name__ == "__main__":
    main()