#!/usr/bin/env python3

# NOTE:
# only ponnel servoing - PBVS

"""
ROS2 Node (Visual servoing node)
Script contains the control loop for visual servoing, both IBVS and PBVS approach can be added here.
Ths script will be triggered from BT (py_trees for now).
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from geometry_msgs.msg import Pose
from visual_servoing_pkg.msg import ArucoCorner
from std_msgs.msg import String, Int64MultiArray
from rclpy.callback_groups import ReentrantCallbackGroup

import cv2
import pickle
import pinocchio
import numpy as np
from scipy.spatial.transform import Rotation

from ComDependencies.robot_controller import robot



class visualServoingNode(Node):
    def __init__(self):
        super().__init__("visual_servoing_node")

        # variables
        self.dt = 0.3
        self.servoTask = None
        self.lambdaVar = 0.3
        self.approxIntMat = None
        self.pixelVel = None
        self.converged = False
        self.aruco_conv = False
        self.socket_conv = False
        self.camVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.eeVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # robot setup (real + model)
        self.bot = robot("192.168.1.9")
        self.robotModel = pinocchio.buildModelsFromUrdf("/home/logesh/fanuc_ws/src/fanuc_ros2_drivers/src/fanuc_description/urdf/lrmate200id4s.urdf")[0]
        self.robotData = pinocchio.createDatas(self.robotModel)[0]
        self.eeFrameId = self.robotModel.getFrameId("tool0")

        # camera parameters
        self.declareCameraParameters()

        # aruco parameters
        self.initializePlugHolderParameters()

        # socket (or holes) parameters
        self.initializeExtBoxParameters()

        # filters initialization
        # moving average (for velocity)
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


        # ros2 comm parameters/variables
        self.data_read_group = ReentrantCallbackGroup()
        # sub
        self.plug_bolt_sub = self.create_subscription(Int64MultiArray, "/bolt_coord", self.plug_bolt_sub_cb, 10, callback_group=self.data_read_group)
        self.plug_holder_pose_sub = self.create_subscription(Pose, "/holder_pose", self.plug_holder_pose_sub_cb, 10, callback_group=self.data_read_group)
        self.matchPoints_sub = self.create_subscription(Int64MultiArray, "/screws_coord", self.matched_points_cb, 10, callback_group=self.data_read_group)
        self.holes_pose_sub = self.create_subscription(Pose, "/ponnel_pose", self.ponnel_pose_cb, 10, callback_group=self.data_read_group)
        self.servo_task_sub = self.create_subscription(String, "/servo_task", self.servo_task_cb, 1, callback_group=self.data_read_group)
        # pub
        self.conv_status_pub = self.create_publisher(String, "/converged_status", 1)
        # timers
        self.main_timer = self.create_timer(1/100, self.controlLoop)


    def declareCameraParameters(self):
        """Function to declare camera parameters like intrinsic's and extrinsic's"""
        
        # intrinsic's
        self.K = pickle.load(open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/camera_matrix_rs.pkl", "rb"))
        self.Kinv = np.linalg.inv(self.K)
        self.camDist = pickle.load(open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/dist_coef_rs.pkl", "rb"))

        # extrinsic's
        mat = pickle.load(open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/eye_in_hand_rs.pkl", "rb"))
        rot = Rotation.from_matrix(mat[0:3, 0:3]).as_matrix()
        self.eTc = np.eye(4)
        self.eTc[0:3, 3] = mat[0:3, 3]
        self.eTc[0:3, 0:3] = rot

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

    def initializePlugHolderParameters(self):
        """
        Function to initialize and declare parameters related to aruco servoing
        Also computes desired interaction matrix for aruco desired points
        """
        # Order:
        # [top_left, top_right, bottom_right, bottom_left]

        self.plugHolderTar_Z = 0.177
        self.plugHolderPose = None
        self.curPlugBolts = None
        self.boltsDepth = np.array([-1.0, -1.0, -1.0, -1.0])        
        self.desPlugBolts = np.array([
            [202, 186],
            [441, 174],
            [440, 310],
            [200, 305],
        ])

        # plug-holder dim
        width = 0.070
        length = 0.040
        self.holder_objectPoints = np.array([
            [-width/2, -(length/2) + 0.005, 0],         # 5mm added for point1
            [width/2, -length/2, 0],
            [width/2, length/2, 0],
            [-width/2, length/2, 0]
        ])

        # compute aruco desired interaction matrix - 8x6 matrix
        tempLst = []
        for u, v in self.desPlugBolts:
            tempLst.append(
                self.computeInteractionMatrix(u, v, self.plugHolderTar_Z)
            )
        self.plug_desiredIntMat = np.vstack(tempLst)
        self.plug_currentIntMat = np.empty((8, 6))

    def initializeExtBoxParameters(self):
        """
        Function to initialize and declare parameters related to extension box screws servoing
        Also computes desired interaction matrix for ext-box screws desired points
        """
        
        self.boxTarZ = 0.2011
        self.ponnelPose = None
        self.curScrews = None
        self.screwsDepth = np.array([-1.0, -1.0, -1.0, -1.0])
        self.desiredScrew = np.array([
            [184, 147],
            [450, 147],
            [449, 347],
            [184, 349],
        ])

        # ext-box screws object point
        width = 0.088
        length = 0.068
        self.box_object_points = np.array([
            [-width/2, -length/2, 0],
            [width/2, -length/2, 0],
            [width/2, length/2, 0],
            [-width/2, length/2, 0]
        ])

        # compute socket holes desired interaction matrix - 10x6 matrix
        tempLst = []
        for u, v in self.desiredScrew:
            tempLst.append(
                self.computeInteractionMatrix(u, v, self.boxTarZ)
            )
        self.box_desiredIntMat = np.vstack(tempLst)
        self.box_currentIntMat = np.empty((8, 6))

    def servo_task_cb(self, msg):
        """Callback function to read servo task"""

        if (msg.data is not None):
            self.servoTask = msg.data
        else:
            self.servoTask = None

    def plug_bolt_sub_cb(self, msg):
        """Callback function to read current plug holder bolts"""
        
        if (msg.data is not None and (msg.data[0] != -1)):
            # extract plug-holder bolts's coordinates
            self.curPlugBolts = np.array(msg.data).reshape((4, 2))
        else:
            self.curPlugBolts = None

    def plug_holder_pose_sub_cb(self, msg):
        """Callback function to read current plug holder pose"""

        if (msg.position is not None and msg.position.x != -1.0):
            # pose extraction
            self.plugHolderPose = np.eye(4)
            self.plugHolderPose[0:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
            rotm = Rotation.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]).as_matrix()
            self.plugHolderPose[0:3, 0:3] = rotm
        else:
            self.plugHolderPose = None

    def ponnel_pose_cb(self, msg):
        """Callback function to read current ponnel pose"""

        if (msg.position is not None and msg.position.x != -1.0):
            # pose extraction
            self.ponnelPose = np.eye(4)
            self.ponnelPose[0:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
            rotm = Rotation.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]).as_matrix()
            self.ponnelPose[0:3, 0:3] = rotm
        else:
            self.ponnelPose = None

    def matched_points_cb(self, msg):
        """Callback function to extract matched points (extension screws)"""

        if (msg.data is not None and (msg.data[0] != -1)):
            # extract ext-box screws's coordinates
            self.curScrews = np.array(msg.data).reshape((4, 2))
        else:
            self.curScrews = None
            # self.get_logger().warn(f"Matched points published are not enough!")

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

    def computeTargetDepth(self, objectPoints, objectPose, depthArray):
        """
        Funtion to compute depth (Z) of target object in camera frame
        
        Args:
           - objectPoints: object points of the point in object frame
           - objectPose: homogeneous transform of object in camera frame
           - depthArray: array to populate the target depth (computed) 
        """

        # compute depth of each object point
        cTo = objectPose                        # camera to object transform
        for idx, point in enumerate(objectPoints):
            oTp = np.eye(4)                     # pose of object point in object frame
            oTp[0:3, 3] = point
            cTp = cTo @ oTp                     # camera to object point
            # store depth (in camera frame)
            depthArray[idx] = cTp[2, 3]    # depth (Z value) 

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
    
    def computePixelVel(self, desFeatureArray, curFeatureArray):
        """
        Function compute pixel velocity from current and desired pixel coordinates
        
        Args:
            - desFeatureArray: list or array containing desired feature points in pixel
            - curFeatureArray: list or array containing current feature points in pixel 
        """

        tempLst = []
        for refPoint, curPoint in zip(desFeatureArray, curFeatureArray):
                tempLst.append(
                    self.computeImgPointVel(refPoint, curPoint).flatten().tolist()
                )
        pixelVel = np.array(tempLst).reshape((len(desFeatureArray) * 2), 1)
        return pixelVel

    def computeCurrentInteractionMat(self, curFeatureArray, featurDepth):
        """
        Function to compute current image jacobian matrix based on the feature points detected.
        
        Args:
            - curFeatureArray: list or array containing current feature points in pixel
            - featureDepth: list or array of depth (Z) of every feature point
        """

        # compute feature jacobian
        tempLst = []
        for (u, v), Z in zip(curFeatureArray, featurDepth):
            tempLst.append(
                self.computeInteractionMatrix(u, v, Z)
            )
        curIntMat = np.vstack(tempLst)
        return curIntMat

    def setAdaptiveGain(self, lam_0, lam_inf, lam_s0):
        """
        Function to compute adaptive gain based on the error vector and other paramters
        This approach was inspired from `ViSP team`.
        Args:
            - lam_0 : float, gain value at error = 0    (i.e., where error is small)
            - lam_inf : float, gain value at error = infinity (i.e., where error is very large)
            - lam_s0 : float, gain at slope in 0 (i.e., idk)
            - pixelVel : np.ndarray(), dtype=float64, error vector
        """
        
        # parameters
        a = lam_0 - lam_inf
        b = lam_s0 / a
        c = lam_inf
        
        # compute infinite norm of error vector (i.e., getting abs max of error vector)
        x_norm = np.linalg.norm(self.pixelVel, np.inf)
        """
        x_norm = 0.0
        for err in self.pixelVel:
            abs_err = abs(err[0])
            if (abs_err > x_norm):
                x_norm = abs_err 
        """
                
        # adaptive gain (lam_adapt)
        lam_adapt = (a * np.exp(-1 * b * x_norm)) + c

        # set adaptive gain to `self.lambdaVar`
        self.lambdaVar = lam_adapt

    def maVelFilter(self, vels):
        """
        Function to perform filtering on computed velocity.
        using simple `Moving Average` filter approach here. 
        """
        # TODO: use matrix multiplication to do this - instead of brute for approach

        fVels = np.array([0.0 for i in range(6)])
        for idx in range(6):    # iterate over joints
            self.maFilterArr[idx][self.maIdx] = vels[idx]
            fVels[idx] = np.sum(self.maFilterArr[idx]) / self.noPoints
        
        self.maIdx += 1
        self.maIdx %= self.noPoints

        return fVels

    def integrateVel(self, qpos, qvel):
        # update joint position by integrating velocity
        for i in range(6):
            qpos[i] += (self.dt * qvel[i])
        
        return qpos

    def computeCamVel(self):
        """Function to compute camera velocity from pixel velocity"""
        
        # handle none cases
        if (self.servoTask == "servo_socket" and self.ponnelPose is None):
            self.camVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return

        if (self.servoTask == "servo_socket" and self.ponnelPose is not None):
            # pbvs for ponnel peg and hole insertion

            # base to end effector transform
            curPose = self.bot.read_current_cartesian_pose()
            bTe = np.eye(4)
            bTe[0:3, 3] = np.array(curPose[0:3]) / 1000.0
            bTe[0:3, 0:3] = Rotation.from_euler("xyz", np.array(curPose[3:6]), degrees=True).as_matrix()

            cTo = self.ponnelPose
            
            dcTo = np.eye(4)
            dcTo[2, 3] = 0.2
            
            dcTc = dcTo @ np.linalg.inv(cTo)

            camvel = np.zeros((6))
            camvel[0:3] = - 0.3 * (dcTc[0:3, 0:3].T @ dcTc[0:3, 3])
            camvel[3:6] = - 0.5 * Rotation.from_matrix(dcTc[0:3, 0:3]).as_rotvec()

            self.camVel = camvel

        else:

            self.camVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # compute lambda for current pixelVel
        # Adaptive gain (lambda_adapt)
        # self.setAdaptiveGain(0.5, 0.3, 30.0)                # default - [1.666, 0.666, 1.666] 
        # tuning adaptive gain parameter using constant lambda
        # self.lambdaVar = 0.3                              # uncomment and tune lambda 0, and inf

        # self.camVel[0:3] = Vc.flatten()
        # self.camVel[3:6] = Wc.flatten()

        # check convergence
        if (np.max(np.abs(self.camVel.flatten())) < 0.006):
            self.converged = True
            self.camVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def computeEEVel(self):
        """Function to compute end effector velocity (form camera velocity)"""
        
        # compute camera velocity from pixel feature velocities
        self.computeCamVel()

        # camera velocity to end effector velocity
        # using adjoint transformation (Ad_eTc)
        self.eeVel = (self.ADeTc @ self.camVel).flatten() 

    def controlLoop(self):
        """Main timer function to compute eeVel at that time stamp and move the arm"""

        # handle
        if (self.servoTask is None or self.servoTask == "no_servo"):
            # servoing not started yet or not to servo now
            self.eeVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # also inform bt about stop of convergence
            msg = String()
            msg.data = "0"
            self.conv_status_pub.publish(msg)

            return
        
        # handle convergence
        if (self.converged == True):
            self.get_logger().info(f"Visual servoing converged ({self.servoTask}): {self.converged}")
            self.eeVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            if (self.servoTask == "servo_aruco"):
                self.aruco_conv = True
                self.socket_conv = False
            elif (self.servoTask == "servo_socket"):
                self.socket_conv = True
                self.aruco_conv = False
            else:
                self.aruco_conv = False
                self.socket_conv = False
            self.servoTask = "no_servo"
        else:
            self.aruco_conv = False
            self.socket_conv = False
            # self.servoTask = "no_servo"

        # publish convergence status
        conv_msg = String()
        if (self.aruco_conv):
            conv_msg.data = "1_aruco"
        elif (self.socket_conv):
            conv_msg.data = "1_socket"
        else:
            conv_msg.data = "0"
        self.conv_status_pub.publish(conv_msg)

        # main control logic
        if (self.servoTask == "servo_aruco" and self.plugHolderPose is None):
            self.get_logger().warn(f"Aruco pose is none")
            self.eeVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif (self.servoTask == "servo_socket" and self.ponnelPose is None):
            self.get_logger().warn(f"Socket pose is none")
            self.eeVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif (self.servoTask == "servo_aruco" or self.servoTask == "servo_socket"):
            # compute ee velocity
            self.converged = False
            self.aruco_conv = False
            self.socket_conv = False
            self.computeEEVel()
        else:
            self.converged = False
            self.eeVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # velocity filter (TODO: Kalman filter instead on moving average)
        self.eeVel = self.maVelFilter(self.eeVel)
        print(np.round(self.eeVel, 4))

        # read the current cartesion position
        cur_joint_pose = self.bot.read_current_joint_position()
        # current joint position (deg to rad) + J23 coupling
        rad_arr = np.deg2rad(cur_joint_pose)
        # remove coupling - J[3]' = J[3] + J[2]
        rad_arr[2] = rad_arr[2] + rad_arr[1]

        # ee velocity to joint velocity
        current_jacobian = pinocchio.computeFrameJacobian(self.robotModel, self.robotData, np.array(rad_arr), self.eeFrameId)   # 6x6 
        joint_vels = (np.linalg.pinv(current_jacobian) @ np.array([self.eeVel]).T)  # 6x6 @ 6x1 => 6x1
        joint_vels = joint_vels.flatten()      # [Vj1, Vj2, Vj3, Vj4, Vj5, Vj6]

        # compute target joint angle from target velocity
        target_rad_arr = self.integrateVel(qpos=rad_arr, qvel=joint_vels)
        
        # adding coupling - J[3]' = J[3] - J[2]
        target_rad_arr[2] = target_rad_arr[2] - target_rad_arr[1]
        target_joint_pose = np.rad2deg(target_rad_arr).tolist()

        # write register and sync-movement
        # self.get_logger().info(f"Computed Joint Position: {np.round(target_joint_pose, 4)}")
        self.bot.write_joint_pose(target_joint_pose, blocking=False)


def main():
    rclpy.init()

    try:
        node = visualServoingNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Shutting down IBVS node:\nException: {e}")
        node.destroy_node()
        # rclpy.shutdown()

if __name__ == "__main__":
    main()