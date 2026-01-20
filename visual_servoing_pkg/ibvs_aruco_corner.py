#!/home/logesh/robotic_toolbox_ws/toolbox_env/bin/python3

# TODO
# 1. put target aruco positions in yaml and other constant variables too.

import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from visual_servoing_pkg.msg import ArucoCorner
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from ComDependencies.robot_controller import robot
from fanuc_vel_controller.fanuc_model import Fanuc, sm


class IBVS_aruco(Node):
    def __init__(self):
        super().__init__("ibvs_aruco_node")

        # aruco variables
        self.cur_top_left = None
        self.cur_top_right = None
        self.cur_bottom_right = None
        self.cur_bottom_left = None
        # target aruco points
        self.tar_top_left = np.array([370, 149])
        self.tar_top_right = np.array([376, 259])
        self.tar_bottom_right = np.array([265, 265])
        self.tar_bottom_left = np.array([259, 155])

        # image jacobian | velocity variables
        self.pixelVel_gain = 0.01
        self.pixelVel = None
        self.imgJacob = None
        self.ee_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.curCamVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.prevCamVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # camera properties
        # self.K = np.load('/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/cameraParams.npz')['arr_0']
        self.K = np.array([
            [900.0, 0.0, 320.0],
            [0.0, 900.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        self.Kinv = np.linalg.inv(self.K)
        self.Z = 2                      # distance from camera to target (assuming it is 1m away) - this is point depth

        # link 6 (or end-effector) to camera transform
        self.eTc = sm.SE3(0.070, 0.0, 0.120)
        self.eTc *= sm.SE3().Rz(np.deg2rad(90))
        self.ADeTc = np.zeros((6, 6))           # adjoint transformation
        self.ADeTc[0:3, 0:3] = self.eTc.R
        self.ADeTc[3:6, 3:6] = self.eTc.R
        self.ADeTc[0:3, 3:6] = np.multiply(np.array([self.eTc.t]).T, self.eTc.R)
        self.ADcTe = np.linalg.inv(self.ADeTc)  # opposite adjoint transformation

        # robot arm controllers
        self.fanuc_model = Fanuc()
        self.bot = robot("192.168.1.9")
        self.triggered = False
        self.tracking_pose = [60.0, 240.0, 120.0, 179.65, 0.69, 67.63]
        self.dt = 1.0   # parameter for velocity integration

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
        self.corner_sub = self.create_subscription(ArucoCorner, "/aruco_corners", self.corners_sub_cb, 10, callback_group=self.vel_gen_group)
        self.inc_srv_trig = self.create_service(SetBool, '/trigger_servoing', self.trigger_servoing_cb)
        self.main_timer = self.create_timer(1/100, self.main_timer_cb, self.vel_gen_group)


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

    def computeImgPointJacobian(self, u, v, Z = 2):
        """
        Function 'computeImgPointJacobian' is used to compute image jacobian or interaction matrix (J) of the given pixel point (u, v).
        """
        # compute image coordinates (x, y) from (u, v)
        # x = (u - self.cx) / self.fx
        # y = (v - self.cy) / self.fy
        point = np.array([[u,v,1]]).T
        xy = self.Kinv @ point
        x = xy[0, 0]
        y = xy[1, 0]
        Z = Z      # point (x, y) depth (in world frame)

        # image jacobian template(or formula) - 2x6
        img_jacobian = self.K[0:2, 0:2] @ np.array([[(-1/Z), 0, (x/Z), (x*y), -(1+(x*x)), y], 
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
            self.ee_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
        self.pixelVel = np.array([self.pixelVel.flatten()]).T     # [[u1_dot], [v1_dot], [u2_dot], [v2_dot], [u3_dot], [v3_dot], [u4_dot], [v4_dot]] - 8x1

        # compute jacobian matrix (combain all for points interation matrix)
        top_left_jacob = self.computeImgPointJacobian(self.cur_top_left[0], self.cur_top_left[1])
        top_right_jacob = self.computeImgPointJacobian(self.cur_top_right[0], self.cur_top_right[1])
        bottom_right_jacob = self.computeImgPointJacobian(self.cur_bottom_right[0], self.cur_bottom_right[1])
        bottom_left_jacob = self.computeImgPointJacobian(self.cur_bottom_left[0], self.cur_bottom_left[1])
        # points jacobian (for all for points) - 8x6 matrix
        pointsJacob = np.vstack([top_left_jacob, top_right_jacob, bottom_right_jacob, bottom_left_jacob])

        # compute camVel 
        # camVel = inv(pointsJacobian) @ pixelVel
        camVel = np.linalg.pinv(pointsJacob) @ self.pixelVel        # (6x1) = (6x8) @ (8x1)
        # NOTE: `camVel.flatten()` -> [Vx, Vy, Vz, Wx, Wy, Wz]

        return camVel.flatten()
    
    def computeEEVel(self):
        # check aruco corners detection
        if (self.cur_top_left is not None):
            # compute desired camera velocity
            self.curCamVel = self.computeCamVel()
            # self.get_logger().info(f"Computed Cam velocity: {self.curCamVel}")

            # camera velocity to end effector velocity
            temp_ee_vel = self.ADeTc @ np.array([self.curCamVel.flatten()]).T

            # for now - lets servo only on x, y, and z (or 3D servoing)
            self.ee_vel[0] = temp_ee_vel[0]
            self.ee_vel[1] = temp_ee_vel[1]
            self.ee_vel[2] = temp_ee_vel[2]

            # self.vel_error = np.subtract(self.curCamVel, self.prevCamVel)
            # self.vel_error = temp_ee_vel
            # self.ee_vel[0] = (self.vel_error[0] * self.KPX) + (self.vel_error[0] * self.KIX) + (self.vel_error[0] * self.KDX)
            # self.ee_vel[1] = (self.vel_error[1] * self.KPY) + (self.vel_error[1] * self.KIY) + (self.vel_error[1] * self.KDY)
            # self.ee_vel[2] = (self.vel_error[2] * self.KPZ) + (self.vel_error[2] * self.KIZ) + (self.vel_error[2] * self.KDZ)
            # self.ee_vel[0] *= -1    # invert x-axis
            # self.ee_vel[0], self.ee_vel[1] = self.ee_vel[1], self.ee_vel[0]

            self.prevCamVel = self.curCamVel.copy()
        else:
            self.ee_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def integrateVel(self, qpos, qvel):
        # update joint position by integrating velocity
        for i in range(6):
            qpos[i] += (self.dt * qvel[i])
        
        return qpos


    def main_timer_cb(self):
        if (self.triggered):
            # compute EE velocity
            self.computeEEVel()
            self.get_logger().info(f"EE Vel: {np.round(self.ee_vel, 4)}")

            # read the current cartesion position
            cur_joint_pose = self.bot.read_current_joint_position()
            # current joint position (deg to rad) + J23 coupling
            rad_arr = np.deg2rad(cur_joint_pose)
            # remove coupling - J[3]' = J[3] + J[2]
            rad_arr[2] = rad_arr[2] + rad_arr[1]

            # ee velocity to joint velocity
            current_jacobian = self.fanuc_model.jacobe(q=np.array(rad_arr))             # 6x6 matrix
            joint_vels = (np.linalg.pinv(current_jacobian) @ np.array([self.ee_vel]).T)  # 6x6 @ 6x1 => 6x1
            joint_vels = joint_vels.flatten()      # [Vj1, Vj2, Vj3, Vj4, Vj5, Vj6]
            # (i guess) - joint_vels are in rad/sec

            """
            # some filtering has to be done on the joint velocities before adding to the current joint positioni
            ### TODO: filter to joint_vels
            
            # worked after inverting the target velocity of joint 2 (may be it is inverted)
            joint_vels[1] *= -1

            # removing velocity on J4 - safety reasons
            joint_vels[3] = 0.0

            # add that to current joint position
            target_rad_arr = np.add(rad_arr, joint_vels)
            """

            target_rad_arr = self.integrateVel(qpos=rad_arr, qvel=joint_vels)
            
            # adding coupling - J[3]' = J[3] - J[2]
            target_rad_arr[2] = target_rad_arr[2] - target_rad_arr[1]
            target_joint_pose = np.rad2deg(target_rad_arr).tolist()

            # write register and sync-movement
            self.get_logger().info(f"Computed Joint Position: {np.round(target_joint_pose, 4)}")
            self.bot.write_joint_pose(target_joint_pose, blocking=False)

def main():
    rclpy.init()

    # try:
    node = IBVS_aruco()
    rclpy.spin(node)
    # except Exception as e:
        # print(f"Shutting down IBVS node:\nException: {e}")
        # node.destroy_node()
        # rclpy.shutdown()

if __name__ == "__main__":
    main()