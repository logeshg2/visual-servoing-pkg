#!/usr/bin/env python3

# TODO
# 1. put target aruco positions in yaml and other constant variables too.

import cv2
import csv
import time
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import Pose, TransformStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from ComDependencies.robot_controller import robot
# from fanuc_vel_controller.fanuc_model import sm
import pinocchio


class PBVS_aruco(Node):
    def __init__(self):
        super().__init__("pbvs_aruco_node")

        # degub tools (logging)
        """
        fp = open("/home/logesh/Desktop/ee_vel.csv", "w")
        fp1 = open("/home/logesh/Desktop/cam_vel.csv", "w")
        fp2 = open("/home/logesh/Desktop/aruco_pose.csv", "w")
        self.writer = csv.writer(fp)
        self.writer1 = csv.writer(fp1)
        self.writer2 = csv.writer(fp2)
        """
        
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
        # aruco pose variable
        self.arucoPose = None

        # image jacobian | velocity variables
        self.pixelVel_gain = 0.02
        self.lambdaVar =    0.3                # exponential decay factor (Lambda)
        self.pixelVel = None
        self.imgJacob = None
        self.ee_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.curCamVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.prevCamVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        """
        # camera intrinsic properties
        cameraParam_fp = open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/camera_matrix.pkl", "rb")
        self.K = pickle.load(cameraParam_fp)
        self.K[0, 2] = 320.0            # calibration is little off
        self.K[1, 2] = 240.0
        """
        # camera intrinsic (realsense)
        self.K = np.array([
            [607.0556030273438, 0.0, 328.3836364746094],
            [0.0, 606.6974487304688, 241.04295349121094],
            [0.0, 0.0, 1.0]
        ])
        self.Kinv = np.linalg.inv(self.K)
        # self.Z = 2                      # distance from camera to target (assuming it is 1m away) - this is point depth # TODO: need to tune this
        """
        # camera extrinsic properties
        camTrans_fp = open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/hand_eye_trans.pkl", "rb")
        camRotm_fp = open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/hand_eye_rotm.pkl", "rb")
        self.camTrans = pickle.load(camTrans_fp)
        self.camTrans /= 1000           # mm to m
        self.camRotm = pickle.load(camRotm_fp)
        """

        # camera extrinsic for realsense (eye in hand config)
        mat = pickle.load(open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/eye_in_hand_rs.pkl", "rb"))
        rvec, _ = cv2.Rodrigues(mat[0:3, 0:3])
        rot, _ = cv2.Rodrigues(rvec)
        self.eTc = np.eye(4)
        self.eTc[0:3, 3] = mat[0:3, 3]
        self.eTc[0:3, 0:3] = rot
        self.cTe = np.linalg.pinv(self.eTc)
        
        """
        # link 6 (or end-effector) to camera transform
        # self.eTc = sm.SE3(0.070, 0.0, 0.120)
        # self.eTc *= sm.SE3().Rz(np.deg2rad(90))
        self.eTc = sm.SE3()
        self.eTc.t = self.camTrans
        self.eTc.R = self.camRotm
        # camera to end-effector transform (cTe)
        self.cTe = self.eTc.inv()
        """
        
        # adjoint transformation (camera frame velocity to end-effector frame velocity transform)
        self.ADeTc = np.zeros((6, 6))
        self.ADeTc[0:3, 0:3] = self.eTc[0:3, 0:3]
        self.ADeTc[3:6, 3:6] = self.eTc[0:3, 0:3]
        eTc_t = self.eTc[0:3, 3]
        etc_x = np.array([               # skew symmetric matrix of translation (eTc_t)
            [0, (-1 * eTc_t[2]), eTc_t[1]],
            [eTc_t[2], 0, (-1 * eTc_t[0])],
            [(-1 * eTc_t[1]), eTc_t[0], 0]
        ])
        self.ADeTc[3:6, 0:3] = etc_x @ self.eTc[0:3, 0:3]
        # print("Adjoint Transformation (Ad_eTc):\n", self.ADeTc)

        # adjoint transformation cVe - transforms velocity from camera to end effector frame
        # this seems worng - refer paper for more details (ISSUE)
        """
        self.cVe = np.zeros((6,6))
        self.cVe[0:3, 0:3] = self.cTe.R
        self.cVe[3:6, 3:6] = self.cTe.R
        cte_x = np.array([               # skew symmetric matrix of translation (cTe.t)
            [0, (-1 * self.cTe.t[2]), self.cTe.t[1]],
            [self.cTe.t[2], 0, (-1 * self.cTe.t[0])],
            [(-1 * self.cTe.t[1]), self.cTe.t[0], 0]
        ])
        self.cVe[0:3, 3:6] = cte_x @ self.cTe.R
        """

        # robot arm controllers
        self.bot = robot("192.168.1.9")
        self.triggered = False
        self.tracking_pose = [60.0, 300.0, 120.0, 179.65, 0.69, 67.63]
        self.dt = 1.0   # parameter for velocity integration

        # setup pinocchio
        self.robotModel = pinocchio.buildModelsFromUrdf("/home/logesh/fanuc_ws/src/fanuc_ros2_drivers/src/fanuc_description/urdf/lrmate200id4s.urdf")[0]
        self.robotData = pinocchio.createDatas(self.robotModel)[0]
        self.eeFrameId = self.robotModel.getFrameId("tool0")

        # PID control (TODO: tune this)
        self.KPX = 5*(0.0001)
        self.KIX = 2*(0.000001)
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

        self.startTime = None

        # ros2 comm variables
        self.tf_broadcaster = TransformBroadcaster(self)
        self.vel_gen_group = MutuallyExclusiveCallbackGroup()
        self.pose_sub = self.create_subscription(Pose, "/aruco_pose", self.pose_sub_cb, 10, callback_group=self.vel_gen_group)
        self.inc_srv_trig = self.create_service(SetBool, '/trigger_servoing', self.trigger_servoing_cb)
        self.main_timer = self.create_timer(1/100, self.main_timer_cb, self.vel_gen_group)

        # csv logging stuff
        # log aruco pose
        # fp = open("/home/logesh/Desktop/arucoPose.csv", "w")
        # fp1 = open("/home/logesh/Desktop/pbvs_camvel.csv", "w")
        # fp2 = open("/home/logesh/Desktop/pbvs_joint_pos.csv", "w")
        # self.pose_writer = csv.writer(fp)
        # self.camvel_writer = csv.writer(fp1)
        # self.q_writer = csv.writer(fp2)
        fp = open("/home/logesh/Desktop/pbvs_ee_pose.csv","w")
        self.ee_pose_writer = csv.writer(fp)


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

        self.startTime = time.perf_counter()

        response.success = True
        response.message = "trigger successful"
        return response

    def computeImgPointJacobian(self, u, v, Z = 0.3):
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
            self.arucoPose = None
            self.ee_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def computePixelPointVel(self, des_point, cur_point):
        """
        function to compute pixel velocity between two points.
        function use `self.pixelVel_gain` to adjust the velocity scale.
        """

        pix_vel = self.pixelVel_gain * (np.subtract(np.array(des_point), np.array(cur_point)))
        return pix_vel

    def setAdaptiveGain(self, lam_0, lam_inf, lam_s0, error_array):
        """
        Function to compute adaptive gain based on the error vector and other paramters
        This approach was inspired from `ViSP team`.
        Args:
            - lam_0 : float, gain value at error = 0    (i.e., where error is small)
            - lam_inf : float, gain value at error = infinity (i.e., where error is very large)
            - lam_s0 : float, gain at slope in 0 (i.e., idk)
            - error_array : array[float], error array
        
        - e_vec : np.ndarray(), dtype=float64, error vector
        """
        
        # parameters
        a = lam_0 - lam_inf
        b = lam_s0 / a
        c = lam_inf
        
        # compute infinite norm of error vector (i.e., getting abs max of error vector)
        x_norm = 0.0
        for err in error_array:
            abs_err = abs(err[0])
            if (abs_err > x_norm):
                x_norm = abs_err 

        # adaptive gain (lam_adapt)
        lam_adapt = (a * np.exp(-1 * b * x_norm)) + c

        # set adaptive gain to `self.lambdaVar`
        self.lambdaVar = lam_adapt

    def computeCamVel(self):
        """
        Function to compute camera veloctiy Vc from current arucoPose to desired pose.
        This method is uses PBVS approach.
        
        returns: camVel
        """

        camVel = np.zeros((6,1))
        # camVel[0:3, :] -> translation camera velocity (Vc)
        # camVel[3:6, :] -> rotation camera velocity (Wc) 

        # log pose to csv
        # lst = []
        # lst.extend(self.arucoPose[0:3, 3].tolist())
        # lst.extend(Rotation.from_matrix(self.arucoPose[0:3, 0:3]).as_rotvec().tolist())
        # self.pose_writer.writerow(lst)

        # current camera to object transform (cTo)
        cTo = self.arucoPose
        cTo_t = cTo[0:3, 3].reshape((3,1))
        cto_x = np.array([               # skew symmetric matrix of translation (cTo.t)
            [0, (-1 * cTo_t[2][0]), cTo_t[1][0]],
            [cTo_t[2][0], 0, (-1 * cTo_t[0][0])],
            [(-1 * cTo_t[1][0]), cTo_t[0][0], 0]
        ])
        cTo_thetaU = Rotation.from_matrix(cTo[0:3, 0:3]).as_rotvec().reshape(3,1)        # aixs rotation vector

        # desired camera to object transform (dcTo)
        dcTo = np.eye(6)
        dcTo[0:3, 3] = np.array([0, 0,  0.25])                       # desired trasform should be 30cm above the aruco board
        dcTo_t = dcTo[0:3, 3].reshape((3,1))

        # Adaptive gain (lambda_adapt)
        errorArray = np.empty((6,1))
        errorArray[0:3, :] = (dcTo_t - cTo_t)
        errorArray[3:6, :] = cTo_thetaU
        self.setAdaptiveGain(0.3, 0.2, 30.0, errorArray)           # default - [1.666, 0.666, 1.666] 
        # tuning adaptive gain parameter using constant lambda
        # self.lambdaVar = 0.3                              # uncomment and tune lambda 0, and inf

        if (np.max(np.abs(errorArray)) < 0.01):
            print()
            print(time.perf_counter() - self.startTime)
            print()
            exit(0)

        # compute velocity
        Vc = -1 * self.lambdaVar * ((dcTo_t - cTo_t) + (cto_x @ cTo_thetaU))
        Wc = -1 * self.lambdaVar * cTo_thetaU

        camVel[0:3, :] = Vc
        camVel[3:6, :] = Wc

        return camVel       # 6x1
    
    def computeEEVel(self):
        # check aruco pose detection
        if (self.arucoPose is not None):
            # compute camera velocity
            camVel = self.computeCamVel()

            # log cam vel
            # self.camvel_writer.writerow(camVel.flatten())

            # camera velocity to end effector velocity
            # using adjoint transformation (Ad_eTc)
            self.ee_vel = (self.ADeTc @ camVel).flatten()           # (6,)
            # self.ee_vel[3:6] = [0.0, 0.0, 0.0]                      # comment to use angular velocities
            # print(np.round(camVel[3:].flatten(), 4))
            # print(np.round(self.ee_vel[3:], 4))
            # print()
            
            self.ee_vel[3] *= -1
            self.ee_vel[4] *= -1
            self.ee_vel[5] *= -1
            

            # log ee pose
            self.ee_pose_writer.writerow(self.bot.read_current_cartesian_pose())

            # log ee_vel and camvel
            """
            self.writer.writerow(self.ee_vel)
            self.writer1.writerow(camVel.flatten())
            temp = []
            temp.extend(self.arucoPose.t.flatten().tolist())
            temp.extend(Rotation.from_matrix(self.arucoPose.R).as_quat().flatten().tolist())
            self.writer2.writerow(temp)
            """
            
            """
            # current pose of end effector
            curEEPose = np.array(self.bot.read_current_cartesian_pose())    # [x, y, z, w, p, r]
            # compute transform bTe
            bTe = sm.SE3(curEEPose[0], curEEPose[1], curEEPose[2])
            rot = Rotation.from_euler('xyz', [curEEPose[3], curEEPose[4], curEEPose[5]], degrees=True)
            rotm = rot.as_matrix()
            bTe.R = rotm
            # compute transform bTc
            bTc = bTe @ self.eTc
            trans = bTc.t
            euls = Rotation.from_matrix(bTc.R).as_euler('zyx', degrees=True)     # [w, p, r]
            curCamPose = np.array([trans[0], trans[1], trans[2], euls[0], euls[1], euls[2]])    # [x, y, z, w, p, r]
            # print(Rotation.from_matrix(bTe.R).as_euler('xyz', degrees=True))
            # print(bTc)
            # compute transfrom cTo
            cTo = self.arucoPose
            # NOTE: the above is in camera frame
            # compute transform bTo
            bTo = bTc @ cTo
            # 30cm above the aruco detection
            bTo.t[2] += 300.0
            # current aruco board pose array (from base brame)
            trans = bTo.t
            euls = Rotation.from_matrix(bTo.R).as_euler('zyx', degrees=True)     # [w, p, r]
            curArucoPose = np.array([trans[0], trans[1], trans[2], euls[0], euls[1], euls[2]])    # [x, y, z, w, p, r]
            # compute position error
            position_error = np.subtract(curArucoPose, curCamPose)       # [[d_x, d_y, d_z, d_r, d_p, d_w]]
            # adding proportional to position error to get velocity
            kp = 0.0001
            temp_ee_vel = position_error * self.KPX + position_error * self.KIX
            # temp_ee_vel[0] *= -1
            # temp_ee_vel[0], temp_ee_vel[1] = temp_ee_vel[1], temp_ee_vel[0]
            # print(np.round(temp_ee_vel, 4))
            # for now - lets servo only on x, y, and z
            self.ee_vel[0] = temp_ee_vel[0]
            self.ee_vel[1] = temp_ee_vel[1]
            self.ee_vel[2] = temp_ee_vel[2]
            self.ee_vel[3] = temp_ee_vel[3]
            self.ee_vel[4] = temp_ee_vel[4]
            self.ee_vel[5] = temp_ee_vel[5]
            # self.vel_error = np.subtract(self.curCamVel, self.prevCamVel)
            # self.vel_error = temp_ee_vel
            # self.ee_vel[0] = (self.vel_error[0] * self.KPX) + (self.vel_error[0] * self.KIX) + (self.vel_error[0] * self.KDX)
            # self.ee_vel[1] = (self.vel_error[1] * self.KPY) + (self.vel_error[1] * self.KIY) + (self.vel_error[1] * self.KDY)
            # self.ee_vel[2] = (self.vel_error[2] * self.KPZ) + (self.vel_error[2] * self.KIZ) + (self.vel_error[2] * self.KDZ)
            # self.ee_vel[0] *= -1    # invert x-axis
            # self.ee_vel[0], self.ee_vel[1] = self.ee_vel[1], self.ee_vel[0]
            # self.prevCamVel = self.curCamVel.copy()
            """
        else:
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

    def TF_publisher(self):
        if (self.arucoPose is not None):
            # broadcast cTo tf
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = 'camera_link'
            tf.child_frame_id = "aruco"

            tf.transform.translation.x = self.arucoPose.t[0]
            tf.transform.translation.y = self.arucoPose.t[1]
            tf.transform.translation.z = self.arucoPose.t[2] - 0.3        # NOTE: this may be wrong

            quat = Rotation.from_matrix(self.arucoPose.R).as_quat()
            tf.transform.rotation.x = quat[0]
            tf.transform.rotation.y = quat[1]
            tf.transform.rotation.z = quat[2]
            tf.transform.rotation.w = quat[3]

            self.tf_broadcaster.sendTransform(tf)


    def main_timer_cb(self):
        if (self.triggered):
            # compute EE velocity
            self.computeEEVel()
            # self.get_logger().info(f"EE Vel: {np.round(self.ee_vel, 4)}")

            # publish camera to object tfs to visualize
            # self.TF_publisher()       # depreicated - now this is done in aruco board detection itself

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
            # joint_vels[1] *= -1
            # joint_vels[3] = 0.0

            # compute target joint position from joint velocities
            target_rad_arr = self.integrateVel(qpos=rad_arr, qvel=joint_vels)
            
            # adding coupling - J[3]' = J[3] - J[2]
            target_rad_arr[2] = target_rad_arr[2] - target_rad_arr[1]
            target_joint_pose = np.rad2deg(target_rad_arr).tolist()

            # log joint position (in deg)
            # self.q_writer.writerow(target_joint_pose)

            # write register and sync-movement
            # self.get_logger().info(f"Computed Joint Position: {np.round(target_joint_pose, 4)}")
            self.bot.write_joint_pose(target_joint_pose, blocking=False)

def main():
    rclpy.init()

    try:
        node = PBVS_aruco()
        rclpy.spin(node)
    except Exception as e:
        print(f"Shutting down PBVS node:\nException: {e}")
        node.destroy_node()
        # rclpy.shutdown()

if __name__ == "__main__":
    main()