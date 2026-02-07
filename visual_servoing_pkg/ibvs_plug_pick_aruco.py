#!/usr/bin/env python3

# TODO
# 1. put target aruco positions in yaml and other constant variables too.

import cv2
import time
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from geometry_msgs.msg import Pose, TransformStamped
from tf2_ros import transform_broadcaster
from visual_servoing_pkg.msg import ArucoCorner
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from ComDependencies.robot_controller import robot
# from fanuc_vel_controller.fanuc_model import sm
import pinocchio


class IBVS_plug_pick(Node):
    def __init__(self):
        super().__init__("ibvs_aruco_node")

        # aruco variables
        self.cur_top_left = None
        self.cur_top_right = None
        self.cur_bottom_right = None
        self.cur_bottom_left = None
        # target aruco points
        self.tar_top_left = np.array([270, 196])
        self.tar_top_right = np.array([363, 196])
        self.tar_bottom_right = np.array([363, 288])
        self.tar_bottom_left = np.array([270, 289])
        self.tar_Z = 0.163
        # aruco pose variable
        self.arucoPose = None
        # aruco object points
        self.markerLength = 0.025
        self.object_points = np.array([
            [-self.markerLength / 2, -self.markerLength / 2, 0],
            [self.markerLength / 2, -self.markerLength / 2, 0],
            [self.markerLength / 2, self.markerLength / 2, 0],
            [-self.markerLength / 2, self.markerLength / 2, 0]
        ])
        # aruco corners depth in camera frame
        self.cornerDepth = np.array([-1.0, -1.0, -1.0, -1.0])   # [topLeft, topRight, bottomRight, bottomLeft]
        # aruco desired points jacobian or interation matrix (8x6)
        self.desiredIntMat = np.array([])

        # image jacobian | velocity variables
        self.lambdaVar =  0.4                  # exponential decay factor (Lambda)
        self.pixelVel = None
        self.imgJacob = None
        self.ee_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.curCamVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.prevCamVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # camera intrinsic (realsense)
        self.K = pickle.load(open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/camera_matrix_rs.pkl", "rb"))
        self.Kinv = np.linalg.inv(self.K)
        
        # camera extrinsic for realsense (eye in hand config)
        mat = pickle.load(open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/eye_in_hand_rs.pkl", "rb"))
        rvec, _ = cv2.Rodrigues(mat[0:3, 0:3])
        rot, _ = cv2.Rodrigues(rvec)
        self.eTc = np.eye(4)
        self.eTc[0:3, 3] = mat[0:3, 3]
        self.eTc[0:3, 0:3] = rot
        self.cTe = np.linalg.pinv(self.eTc)

        # adjoint transformation (camera frame velocity to end-effector frame velocity transform)
        self.ADeTc = np.zeros((6, 6))
        self.ADeTc[0:3, 0:3] = self.eTc[0:3, 0:3]
        self.ADeTc[3:6, 3:6] = self.eTc[0:3, 0:3]
        eTc_t = self.eTc[0:3, 3]
        etc_x = np.array([               # skew symmetric matrix of translation (eTc.t)
            [0, (-1 * eTc_t[2]), eTc_t[1]],
            [eTc_t[2], 0, (-1 * eTc_t[0])],
            [(-1 * eTc_t[1]), eTc_t[0], 0]
        ])
        self.ADeTc[3:6, 0:3] = etc_x @ self.eTc[0:3, 0:3]
        # print("Adjoint Transformation (Ad_eTc):\n", self.ADeTc)

        # robot arm controllers
        self.bot = robot("192.168.1.9")
        self.triggered = False
        self.converged = False
        self.picked = False
        self.inServoing = False
        self.tracking_pose = [60.0, 240.0, 120.0, 179.65, 0.69, 67.63]
        self.dt = 1.0   # parameter for velocity integration

        # setup pinocchio
        self.robotModel = pinocchio.buildModelsFromUrdf("/home/logesh/fanuc_ws/src/fanuc_ros2_drivers/src/fanuc_description/urdf/lrmate200id4s.urdf")[0]
        self.robotData = pinocchio.createDatas(self.robotModel)[0]
        self.eeFrameId = self.robotModel.getFrameId("tool0")

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

        # velocity filter
        self.noPoints = 10
        self.maFilterArr = np.array([
            [0.0 for i in range(self.noPoints)],
            [0.0 for i in range(self.noPoints)],
            [0.0 for i in range(self.noPoints)],
            [0.0 for i in range(self.noPoints)],
            [0.0 for i in range(self.noPoints)],
            [0.0 for i in range(self.noPoints)],
        ])
        self.maIdx = 0

        # compute desired interaction matrix - this will be constant
        self.computeDesiredInteractionMat()

        # tf2 componenets
        self.tf_broadcaster = transform_broadcaster.TransformBroadcaster(self)

        # ros2 comm variables
        self.vel_gen_group = MutuallyExclusiveCallbackGroup()
        self.corner_sub = self.create_subscription(ArucoCorner, "/aruco_corners", self.corners_sub_cb, 10, callback_group=self.vel_gen_group)
        self.inc_srv_trig = self.create_service(SetBool, '/trigger_servoing', self.trigger_servoing_cb)
        self.pose_sub = self.create_subscription(Pose, "/aruco_pose", self.pose_sub_cb, 10, callback_group=self.vel_gen_group)
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

        # default gripper state
        self.bot.air_gripper_control('open')

        response.success = True
        response.message = "trigger successful"
        return response

    def computeDesiredInteractionMat(self):
        """
        Function to compute desired aruco corner points interaction matrix
        This matrix computed will be used for `Approximation of Interaction Matrix`.
        """

        # desire aruco points (corners) interaction matrix
        p1_jac = self.computeInteractionMatrix(self.tar_top_left[0], self.tar_top_left[1], self.tar_Z)
        p2_jac = self.computeInteractionMatrix(self.tar_top_right[0], self.tar_top_right[1], self.tar_Z)
        p3_jac = self.computeInteractionMatrix(self.tar_bottom_right[0], self.tar_bottom_right[1], self.tar_Z)
        p4_jac = self.computeInteractionMatrix(self.tar_bottom_left[0], self.tar_bottom_left[1], self.tar_Z)
        # points jacobian (for all for points) - 8x6 matrix
        # desired points interaction matrix
        self.desiredIntMat = np.vstack([p1_jac, p2_jac, p3_jac, p4_jac])

    def computeInteractionMatrix(self, u, v, Z):
        """
        Function 'computeInteractionMatrix' is used to compute image jacobian (J) or interaction matrix (L) of the given pixel point (u, v).
        
        Args:
            - u : in pixel
            - v : in pixel
            - Z : in meters (depth of point in camera frame)
        
        """
        # compute image coordinates (x, y) from (u, v)
        # x = (u - self.cx) / self.fx
        # y = (v - self.cy) / self.fy
        point = np.array([[u,v,1]]).T
        xy = self.Kinv @ point
        x = xy[0, 0]
        y = xy[1, 0]
        Z = Z

        # image jacobian template(or formula) - 2x6
        img_jacobian = np.array([[(-1/Z), 0, (x/Z), (x*y), -(1+(x*x)), y], 
                                [0, (-1/Z), (y/Z), (1+(y*y)), (-x*y), -x]])
        
        return img_jacobian

    def setAdaptiveGain(self, lam_0, lam_inf, lam_s0):
        """
        Function to compute adaptive gain based on the error vector and other paramters
        This approach was inspired from `ViSP team`.
        Args:
            - lam_0 : float, gain value at error = 0    (i.e., where error is small)
            - lam_inf : float, gain value at error = infinity (i.e., where error is very large)
            - lam_s0 : float, gain at slope in 0 (i.e., idk)
        
        - e_vec : np.ndarray(), dtype=float64, error vector
        """
        
        # parameters
        a = lam_0 - lam_inf
        b = lam_s0 / a
        c = lam_inf
        
        # compute infinite norm of error vector (i.e., getting abs max of error vector)
        x_norm = 0.0
        for err in self.pixelVel:
            abs_err = abs(err[0])
            if (abs_err > x_norm):
                x_norm = abs_err 

        # adaptive gain (lam_adapt)
        lam_adapt = (a * np.exp(-1 * b * x_norm)) + c

        # set adaptive gain to `self.lambdaVar`
        self.lambdaVar = lam_adapt

    def corners_sub_cb(self, msg):
        if (msg.top_left is not None):
            self.cur_top_left = msg.top_left
            self.cur_top_right = msg.top_right
            self.cur_bottom_right = msg.bottom_right
            self.cur_bottom_left = msg.bottom_left
        else:
            self.inServoing = False
            self.cur_top_left = None
            self.cur_top_right = None
            self.cur_bottom_right = None
            self.cur_bottom_left = None
            self.ee_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def pose_sub_cb(self, msg):
        if (msg.position is not None and msg.position.x != -1.0):
            # pose extraction
            self.arucoPose = np.eye(4)
            self.arucoPose[0:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
            # self.arucoPose.t *= 1000        # to mm
            rotm = Rotation.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]).as_matrix()
            # quat = sm.UnitQuaternion([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])  # NOTE: [w, x, y, z]
            self.arucoPose[0:3, 0:3] = rotm
        else:
            self.inServoing = False
            self.arucoPose = None
            self.ee_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def computeImgPointVel(self, des_point, cur_point):
        """
        function to compute image pixel (i.e., change in image place) velocity between two points.
        from 'Visual Servo Control Part I: Basic Approaches' literature -> error = current - desired 
        """

        desPoint = np.array([des_point[0], des_point[1], 1]).reshape(3, 1)
        curPoint = np.array([cur_point[0], cur_point[1], 1]).reshape(3, 1)
        des_xy = self.Kinv @ desPoint
        cur_xy = self.Kinv @ curPoint
        
        des_xy = des_xy.flatten()[:2]
        cur_xy = cur_xy.flatten()[:2]

        pix_vel = np.subtract(np.array(cur_xy), np.array(des_xy))           # difference of points in image plane
        return pix_vel

    def computeCornerDepth(self):
        """ Function to compute depth (Z) from camera frame to the corners (in aruco case) """
        
        # depth of the corners are determined using object points
        for idx, pt in enumerate(self.object_points):
            cTa = self.arucoPose                # camera to aruco pose
            aTp = np.eye(4)                     # aruco to corner point pose
            aTp[0:3, 3] = pt
            cTp = cTa @ aTp                     # camera to corner point pose
            # store depth (in camera frame)
            self.cornerDepth[idx] = cTp[2, 3]   # depth (Z value)   

    def computeCamVel(self):
        # compute depth corners
        self.computeCornerDepth()
        # populated - `self.cornerDepth` array - [topLeft, topRight, bottomRight, bottomLeft]

        # compute pixel velocity
        top_left_pix_vel = self.computeImgPointVel(self.tar_top_left, self.cur_top_left)
        top_right_pix_vel = self.computeImgPointVel(self.tar_top_right, self.cur_top_right)
        bottom_right_pix_vel = self.computeImgPointVel(self.tar_bottom_right, self.cur_bottom_right)
        bottom_left_pix_vel = self.computeImgPointVel(self.tar_bottom_left, self.cur_bottom_left)
        # flatten pixel velocities (here - 8x1 vector)
        self.pixelVel = np.array([top_left_pix_vel.flatten(), top_right_pix_vel.flatten(), bottom_right_pix_vel.flatten(), bottom_left_pix_vel.flatten()])
        self.pixelVel = np.array([self.pixelVel.flatten()]).T     # [[u1_dot], [v1_dot], [u2_dot], [v2_dot], [u3_dot], [v3_dot], [u4_dot], [v4_dot]] - 8x1

        # compute interation matrix (combain all for points interation matrix)
        top_left_jacob = self.computeInteractionMatrix(self.cur_top_left[0], self.cur_top_left[1], self.cornerDepth[0])
        top_right_jacob = self.computeInteractionMatrix(self.cur_top_right[0], self.cur_top_right[1], self.cornerDepth[1])
        bottom_right_jacob = self.computeInteractionMatrix(self.cur_bottom_right[0], self.cur_bottom_right[1], self.cornerDepth[2])
        bottom_left_jacob = self.computeInteractionMatrix(self.cur_bottom_left[0], self.cur_bottom_left[1], self.cornerDepth[3])
        # points jacobian (for all for points) - 8x6 matrix
        # current points jacobian
        pointsJacob = np.vstack([top_left_jacob, top_right_jacob, bottom_right_jacob, bottom_left_jacob])

        # [IMP]
        # Approximation of Interaction Matrix   (8x6)
        approxIntMat = (pointsJacob + self.desiredIntMat) / 2

        # Adaptive gain (lambda_adapt)
        self.setAdaptiveGain(0.5, 0.3, 30.0)           # default - [1.666, 0.666, 1.666] 
        # tuning adaptive gain parameter using constant lambda
        # self.lambdaVar = 0.3                              # uncomment and tune lambda 0, and inf

        # compute camVel 
        # camVel = -1 * self.lambdaVar * (inv(approxIntMat) @ pixelVel)
        camVel = -1 * self.lambdaVar * (np.linalg.pinv(approxIntMat) @ self.pixelVel)        # (6x1) = (6x8) @ (8x1)
        # NOTE: self.lambdaVar is negative for Eye in Hand, and positive for Eye to Hand
        # NOTE: `camVel.flatten()` -> [Vx, Vy, Vz, Wx, Wy, Wz]
        
        # print(np.round(camVel.flatten(), 4))
        
        return camVel.flatten()
    
    def computeEEVel(self):
        # check aruco corners detection
        if (self.cur_top_left is not None and self.cur_top_left[0] != -1):
            # compute desired camera velocity
            self.curCamVel = self.computeCamVel()
            # self.get_logger().info(f"Computed Cam velocity: {self.curCamVel}")

            # camera velocity to end effector velocity
            # using adjoint transformation (Ad_eTc)
            self.ee_vel = (self.ADeTc @ self.curCamVel).flatten()       # (6,)
            # self.ee_vel[3:6] = [0.0, 0.0, 0.0]                        # comment to use angular velocities
            self.inServoing = True
        else:
            self.inServoing = False
            self.ee_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def integrateVel(self, qpos, qvel):
        # update joint position by integrating velocity
        for i in range(6):
            qpos[i] += (self.dt * qvel[i])
        
        return qpos

    def maVelFilter(self, jointVels):
        """
        Function to perform filtering on computed velocity.
        using simple `Moving Average` filter approach here. 
        """
        # TODO: use matrix multiplication to do this - instead of brute for approach

        fVels = np.array([0.0 for i in range(6)])
        for idx in range(6):    # iterate over joints
            self.maFilterArr[idx][self.maIdx] = jointVels[idx]
            fVels[idx] = np.sum(self.maFilterArr[idx]) / self.noPoints
        
        self.maIdx += 1
        self.maIdx %= self.noPoints

        return fVels

    def TF_publisher(self, trans, rotm):
        # broadcast cTo tf
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'tool0'
        tf.child_frame_id = "target_ee"
        
        tf.transform.translation.x = trans[0]
        tf.transform.translation.y = trans[1]
        tf.transform.translation.z = trans[2]

        quat = Rotation.from_matrix(rotm).as_quat()

        tf.transform.rotation.x = quat[0]
        tf.transform.rotation.y = quat[1]
        tf.transform.rotation.z = quat[2]
        tf.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(tf)

    def main_timer_cb(self):
        # perform visual servoing
        if (self.triggered):
            # compute EE velocity
            self.computeEEVel()
            # self.get_logger().info(f"EE Vel: {self.ee_vel}")

            # velocity filter (TODO: Kalman filter instead on moving average)
            self.ee_vel = self.maVelFilter(self.ee_vel)
            print(np.round(self.ee_vel, 4))

            # check convergence ( only when servoing )
            if (self.inServoing and np.all(self.ee_vel < 0.0004)):
                self.get_logger().info(f"Visual Servoing Converged!")
                self.triggered = False
                self.converged = True
                self.inServoing = False
                self.get_logger().info(f"Performing picking operations")
                return

            # read the current cartesion position
            cur_joint_pose = self.bot.read_current_joint_position()
            # current joint position (deg to rad) + J23 coupling
            rad_arr = np.deg2rad(cur_joint_pose)
            # remove coupling - J[3]' = J[3] + J[2]
            rad_arr[2] = rad_arr[2] + rad_arr[1]

            # ee velocity to joint velocity
            current_jacobian = pinocchio.computeFrameJacobian(self.robotModel, self.robotData, np.array(rad_arr), self.eeFrameId)   # 6x6 
            joint_vels = (np.linalg.pinv(current_jacobian) @ np.array([self.ee_vel]).T)  # 6x6 @ 6x1 => 6x1
            joint_vels = joint_vels.flatten()      # [Vj1, Vj2, Vj3, Vj4, Vj5, Vj6]

            # compute target joint angle from target velocity
            target_rad_arr = self.integrateVel(qpos=rad_arr, qvel=joint_vels)
            
            # adding coupling - J[3]' = J[3] - J[2]
            target_rad_arr[2] = target_rad_arr[2] - target_rad_arr[1]
            target_joint_pose = np.rad2deg(target_rad_arr).tolist()

            # write register and sync-movement
            # self.get_logger().info(f"Computed Joint Position: {np.round(target_joint_pose, 4)}")
            self.bot.write_joint_pose(target_joint_pose, blocking=False)

        # perform picking (after convergence)
        if (self.converged and not self.picked):
            # compute current ee pose
            curPose = self.bot.read_current_cartesian_pose()
            bTe = np.eye(4) 
            bTe[0:3, 3] = np.array(curPose[0:3]) / 1000.0
            bTe[0:3, 0:3] = Rotation.from_euler("xyz", np.array(curPose[3:6]), degrees=True).as_matrix()
            
            # aruco to grip pose
            aTo = np.eye(4)
            aTo[0:3, 3] = np.array([0.004, 0.042, -0.025])
            aTo[0:3, 0:3] = Rotation.from_euler("xyz", (0, 0, -90), degrees=True).as_matrix()
            # camera to grip pose (self.arucoPose -> cTa)
            cTo = self.arucoPose @ aTo

            # ee to grip pose
            eTo = self.eTc @ cTo
            eTo[2, 3] -= 0.110
            # eTo[0:3, 0:3] = np.eye(3)

            # base to object pose
            bTo = bTe @ eTo

            # compute pose as list
            cartPose = np.empty(6)
            cartPose[0:3] = bTo[0:3, 3] * 1000
            cartPose[3:6] = Rotation.from_matrix(bTo[0:3, 0:3]).as_euler("xyz", degrees=True)
            cartPose[3:5] = np.array([-179.9, 0.0])     # assuming the plug is perpendicular

            # go to target cartesian pose
            self.bot.write_cartesian_position(cartPose, blocking=False)
            time.sleep(2)
            while (self.bot.is_moving()):
                time.sleep(0.1)
            
            # decrease z to pick (after aligning)
            curPose = self.bot.read_current_cartesian_pose()
            curPose[2] -= 17    # move 17 mm down
            self.bot.write_cartesian_position(curPose, blocking=False)
            time.sleep(2)
            while (self.bot.is_moving()):
                time.sleep(0.1)

            # close gripper
            time.sleep(1)
            self.bot.air_gripper_control("close")
            time.sleep(1)

            # move up in z
            curPose = self.bot.read_current_cartesian_pose()
            curPose[2] += 40
            self.bot.write_cartesian_position(curPose, blocking=False)
            time.sleep(2)
            while (self.bot.is_moving()):
                time.sleep(0.1)

            # move down -> check force -> if high than threshold -> go up again -> explore x-y coord -> if reached 15mm down -> then stop
            # this did not work
            """
            fz_thresh = 100
            hit = False
            for step in range(1, 15):
                fz = self.bot.read_force_sensor_values()[0]
                print(fz)
                if (fz >= fz_thresh):
                    hit = True
                    break
                curPose = self.bot.read_current_cartesian_pose()
                curPose[2] -= 1
                # move the arm
                self.bot.set_speed_percent(10)
                self.bot.write_cartesian_position(curPose, blocking=False)
                time.sleep(2)
                while (self.bot.is_moving()):
                    time.sleep(0.1)

            if (not hit):
                print("Arm reached the gripping position - no hit")
            else:
                print("Arm hit the target - going back")
                self.bot.set_speed_percent(10)
                self.bot.write_joint_pose(tempPose, blocking=False)
                time.sleep(3)
                while (self.bot.is_moving()):
                    time.sleep(0.1)
            """

            """
            # compute desired joint config
            curJointConfig = self.bot.read_current_joint_position()
            tarJointConfig = curJointConfig + self.diff2reach_pick
            # move to target pick location
            self.bot.write_joint_pose(tarJointConfig, blocking=False)
            time.sleep(1)
            while (self.bot.is_moving()):
                time.sleep(0.1)

            # move z down
            curJointConfig = self.bot.read_current_joint_position()
            tarJointConfig = curJointConfig + self.pickmove
            # move to target pick location
            self.bot.write_joint_pose(tarJointConfig, blocking=False)
            time.sleep(1)
            while (self.bot.is_moving()):
                time.sleep(0.1)

            # close gripper
            self.bot.air_gripper_control("close")

            # move z up
            curJointConfig = self.bot.read_current_joint_position()
            tarJointConfig = curJointConfig - self.pickmove
            # move to target pick location
            self.bot.write_joint_pose(tarJointConfig, blocking=False)
            time.sleep(1)
            while (self.bot.is_moving()):
                time.sleep(0.1)

            # goback to tracking position
            self.bot.write_cartesian_position(coords=self.tracking_pose, blocking=False)
            time.sleep(2)
            while (self.bot.is_moving()):
                time.sleep(0.1)
            """
            # perform picking
            self.picked = True


def main():
    rclpy.init()

    try:
        node = IBVS_plug_pick()
        rclpy.spin(node)
    except Exception as e:
        print(f"Shutting down IBVS node:\nException: {e}")
        node.destroy_node()
        # rclpy.shutdown()

if __name__ == "__main__":
    main()