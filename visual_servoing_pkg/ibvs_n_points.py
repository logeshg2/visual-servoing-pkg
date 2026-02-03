#!/usr/bin/env python3

"""Script to perform image based visual servoing on n-points method"""

import cv2
import time
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from visual_servoing_pkg.msg import MatchedPoints
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from ComDependencies.robot_controller import robot
import pinocchio


class IBVS_n_points(Node):
    def __init__(self):
        super().__init__("ibvs_n_points_node")

        # image jacobian | velocity variables
        self.lambdaVar =  0.4                  # exponential decay factor (Lambda)
        self.pixelVel = None
        self.imgJacob = None
        self.ee_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.curCamVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.prevCamVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # desired points jacobian or interation matrix
        self.desiredIntMat = np.array([])
        # current points jacobian or interation matrix
        self.currentIntMat = np.array([])

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
        # self.bot = robot("192.168.1.9")
        self.triggered = False
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

        # variables
        self.depthImg = None
        self.match_0 = None
        self.match_1 = None
        self.cv_bridge = CvBridge()
        # reference depth image
        self.refDepthImg = np.load("/home/logesh/fanuc_ws/src/visual-servoing-pkg/doc/images/ref_img_socket_depth.npz")['depthArr']
        self.refDepthImg = np.float64(self.refDepthImg) / 1000.0            # in meters

        # NOTE: Desired interation matric will not be constant - for n-points approach
        # compute desired interaction matrix - this will be constant
        # self.computeDesiredInteractionMat()

        # ros2 comm variables
        self.vel_gen_group = MutuallyExclusiveCallbackGroup()
        self.inc_srv_trig = self.create_service(SetBool, '/trigger_servoing', self.trigger_servoing_cb)
        self.depthImg_sub = self.create_subscription(Image, "/depth_image", self.depthImg_cb, 10)
        self.matchPoints_sub = self.create_subscription(MatchedPoints, "/matched_points", self.matched_points_cb, 10)
        self.main_timer = self.create_timer(1/100, self.main_timer_cb, self.vel_gen_group)


    def depthImg_cb(self, msg):
        """Function to extract depth image from ros2 image message"""
        
        if (msg is not None):
            self.depthImg = self.cv_bridge.imgmsg_to_cv2(msg)
            self.depthImg = np.float64(self.depthImg) / 1000.0          # in meters
        else:
            self.depthImg = None
            self.get_logger().warn(f"Depth image is None!")

    def matched_points_cb(self, msg):
        """Function to extract matched points from the ros2 custom message"""

        if (msg.rows is not None and (msg.rows != 0 and msg.rows != -1)):
            rows = msg.rows
            cols = msg.cols

            self.match_0 = np.array(msg.match_0).reshape((rows, cols))
            self.match_1 = np.array(msg.match_1).reshape((rows, cols))

        else:
            self.match_0 = None
            self.match_1 = None
            # self.get_logger().warn(f"Matched points published are not enough!")

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

    def computeDesiredInteractionMat(self):
        """
        Function to compute desired n - points interaction matrix
        This matrix computed will be used for `Approximation of Interaction Matrix`.
        """

        """
        # desire aruco points (corners) interaction matrix
        p1_jac = self.computeInteractionMatrix(self.tar_top_left[0], self.tar_top_left[1], self.tar_Z)
        p2_jac = self.computeInteractionMatrix(self.tar_top_right[0], self.tar_top_right[1], self.tar_Z)
        p3_jac = self.computeInteractionMatrix(self.tar_bottom_right[0], self.tar_bottom_right[1], self.tar_Z)
        p4_jac = self.computeInteractionMatrix(self.tar_bottom_left[0], self.tar_bottom_left[1], self.tar_Z)
        # points jacobian (for all for points) - 8x6 matrix
        # desired points interaction matrix
        self.desiredIntMat = np.vstack([p1_jac, p2_jac, p3_jac, p4_jac])
        """

        tempLst = []
        # compute interation matrix for all matched points in reference image
        # for ref point depth - using depth frame (reference depth frame)
        for point in self.match_0:
            tempLst.append(
                self.computeInteractionMatrix(point[0], point[1], self.refDepthImg[point[0], point[1]])
            )

        # desired points interaction matrix
        self.desiredIntMat = np.array([])
        self.desiredIntMat = np.vstack(tempLst)

    def computeCurrentInteractionMat(self):
        """
        Function to compute current n - points interaction matrix
        This matrix computed will be used for `Approximation of Interaction Matrix`.
        """

        """
        # compute interation matrix (combain all for points interation matrix)
        top_left_jacob = self.computeInteractionMatrix(self.cur_top_left[0], self.cur_top_left[1], self.cornerDepth[0])
        top_right_jacob = self.computeInteractionMatrix(self.cur_top_right[0], self.cur_top_right[1], self.cornerDepth[1])
        bottom_right_jacob = self.computeInteractionMatrix(self.cur_bottom_right[0], self.cur_bottom_right[1], self.cornerDepth[2])
        bottom_left_jacob = self.computeInteractionMatrix(self.cur_bottom_left[0], self.cur_bottom_left[1], self.cornerDepth[3])
        # points jacobian (for all for points) - 8x6 matrix
        # current points jacobian
        self.currentIntMat = np.vstack([top_left_jacob, top_right_jacob, bottom_right_jacob, bottom_left_jacob])
        """

        tempLst = []
        # compute interation matrix for all matched points in current image
        # for ref point depth - using current depth frame
        for point in self.match_1:
            tempLst.append(
                self.computeInteractionMatrix(point[0], point[1], self.depthImg[point[0], point[1]])
            )

        # desired points interaction matrix
        self.currentIntMat = np.array([])
        self.currentIntMat = np.vstack(tempLst)

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
        """
        Function to compute camera velocity based on image pixel velocity or pixel error.
        For N-Points approach, at each inference desired interaction matrix is computed + current points interation matrix is also formed.
        """

        # compute desired interation matrix of the matched feature points in the reference image.
        self.computeDesiredInteractionMat()

        # compute current points interation matrix of the matched feature points in the current image.
        self.computeCurrentInteractionMat()

        # [IMP]
        # Approximation of Interaction Matrix
        approxIntMat = (self.currentIntMat + self.desiredIntMat) / 2

        """
        # compute pixel velocity
        top_left_pix_vel = self.computeImgPointVel(self.tar_top_left, self.cur_top_left)
        top_right_pix_vel = self.computeImgPointVel(self.tar_top_right, self.cur_top_right)
        bottom_right_pix_vel = self.computeImgPointVel(self.tar_bottom_right, self.cur_bottom_right)
        bottom_left_pix_vel = self.computeImgPointVel(self.tar_bottom_left, self.cur_bottom_left)
        # flatten pixel velocities (here - 8x1 vector)
        self.pixelVel = np.array([top_left_pix_vel.flatten(), top_right_pix_vel.flatten(), bottom_right_pix_vel.flatten(), bottom_left_pix_vel.flatten()])
        self.pixelVel = np.array([self.pixelVel.flatten()]).T     # [[u1_dot], [v1_dot], [u2_dot], [v2_dot], [u3_dot], [v3_dot], [u4_dot], [v4_dot]] - 8x1
        """

        # compute pixel velocity
        tempLst = []
        for refPoint, curPoint in zip(self.match_0, self.match_1):
            tempLst.extend(
                self.computeImgPointVel(refPoint, curPoint).flatten().tolist()
            )
        self.pixelVel = np.array(tempLst).reshape(((len(self.match_0) * 2), 1))     # column vector

        # Adaptive gain (lambda_adapt)
        self.setAdaptiveGain(0.5, 0.3, 30.0)           # default - [1.666, 0.666, 1.666] 
        # tuning adaptive gain parameter using constant lambda
        # self.lambdaVar = 0.3                              # uncomment and tune lambda 0, and inf

        # compute camVel 
        # camVel = -1 * self.lambdaVar * (inv(approxIntMat) @ pixelVel)
        camVel = -1 * self.lambdaVar * (np.linalg.pinv(approxIntMat) @ self.pixelVel)
        # NOTE: self.lambdaVar is negative for Eye in Hand, and positive for Eye to Hand
        # NOTE: `camVel.flatten()` -> [Vx, Vy, Vz, Wx, Wy, Wz]
        
        # print(np.round(camVel.flatten(), 4))
        
        return camVel.flatten()
    
    def computeEEVel(self):
        # check matches between refImg and curImg - and also matched points should be atleast 4 (IMP)
        if (self.match_0 is not None and len(self.match_0) >= 4):
            # compute desired camera velocity
            self.curCamVel = self.computeCamVel()
            # self.get_logger().info(f"Computed Cam velocity: {self.curCamVel}")

            # camera velocity to end effector velocity
            # using adjoint transformation (Ad_eTc)
            self.ee_vel = (self.ADeTc @ self.curCamVel).flatten()       # (6,)
            # self.ee_vel[3:6] = [0.0, 0.0, 0.0]                        # comment to use angular velocities
            # print(np.round(camVel[3:].flatten(), 4))
            # print(np.round(self.ee_vel[3:], 4))
            # print()
            # self.ee_vel[3] *= -1
            # self.ee_vel[4] *= -1
            # self.ee_vel[5] *= -1

        else:
            n_matches = 0 if (self.match_0 is None) else len(self.match_0)
            self.get_logger().warn(f"Not enough feature matches to perform servoing: {n_matches}")
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


    def main_timer_cb(self):
        if (self.triggered):
            # compute EE velocity
            self.computeEEVel()
            # self.get_logger().info(f"EE Vel: {self.ee_vel}")

            # velocity filter (TODO: Kalman filter instead on moving average)
            self.ee_vel = self.maVelFilter(self.ee_vel)
            print(np.round(self.ee_vel, 4))

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
            # (i guess) - joint_vels are in rad/sec

            """
            # some filtering has to be done on the joint velocities before adding to the current joint positioni
            ### TODO: filter to joint_vels - low pass filter like Kalman filter or Alpha-Beta filter
            
            # worked after inverting the target velocity of joint 2 (may be it is inverted)
            joint_vels[1] *= -1

            # removing velocity on J4 - safety reasons
            joint_vels[3] = 0.0

            # add that to current joint position
            target_rad_arr = np.add(rad_arr, joint_vels)
            """
            # NOTE: commenting joint 2 (flip) - fixed using pinocchio pkg
            # joint_vels[1] *= -1     # this is due to wrong ee-jacobian (i guess)
            # joint_vels[3] = 0.0

            target_rad_arr = self.integrateVel(qpos=rad_arr, qvel=joint_vels)
            
            # adding coupling - J[3]' = J[3] - J[2]
            target_rad_arr[2] = target_rad_arr[2] - target_rad_arr[1]
            target_joint_pose = np.rad2deg(target_rad_arr).tolist()

            # TODO: PD controller for target joint pose
            """
            # PD control on target joint pose
            pose_diff = np.subtract(target_joint_pose - cur_joint_pose)
            """

            # write register and sync-movement
            # self.get_logger().info(f"Computed Joint Position: {np.round(target_joint_pose, 4)}")
            self.bot.write_joint_pose(target_joint_pose, blocking=False)

def main():
    rclpy.init()

    try:
        node = IBVS_n_points()
        rclpy.spin(node)
    except Exception as e:
        print(f"Shutting down IBVS node:\nException: {e}")
        node.destroy_node()
        # rclpy.shutdown()

if __name__ == "__main__":
    main()