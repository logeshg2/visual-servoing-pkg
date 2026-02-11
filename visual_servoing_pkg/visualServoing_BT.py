#!/usr/bin/env python3

"""
ROS2 Node (Visual servoing node)
Script contains the control loop for visual servoing, both IBVS and PBVS approach can be added here.
Ths script will be triggered from BT (py_trees for now).
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from geometry_msgs.msg import Pose
from std_msgs.msg import Int64MultiArray
from visual_servoing_pkg.msg import ArucoCorner
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
        self.dt = 1.0
        self.lambdaVar = 0.3
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
        self.initializeArucoParameters()

        # socket (or holes) parameters
        self.initializeSocketHolesParameters()

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
        self.aruco_corner_sub = self.create_subscription(ArucoCorner, "/aruco_corners", self.corners_sub_cb, 10, callback_group=self.data_read_group)
        self.aruco_pose_sub = self.create_subscription(Pose, "/aruco_pose", self.aruco_pose_sub_cb, 10, callback_group=self.data_read_group)
        self.matchPoints_sub = self.create_subscription(Int64MultiArray, "/holes_coord", self.matched_points_cb, 10, callback_group=self.data_read_group)
        self.holes_pose_sub = self.create_subscription(Pose, "/holes_pose", self.socket_pose_cb, 10, callback_group=self.data_read_group)
        # pub
        ##
        # services
        self.inc_srv_trig = self.create_service(SetBool, '/trigger_servoing', self.trigger_servoing_cb)
        # timers
        self.main_timer = self.create_timer(1/100, self.controlLoop)


    def declareCameraParameters(self):
        """Function to declare camera parameters like intrinsic's and extrinsic's"""
        
        # intrinsic's
        self.K = pickle.load(open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/camera_matrix_rs.pkl", "rb"))
        self.Kinv = np.linalg.inv(self.K)
        self.camDist = pickle.load(open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/dist_coef_rs.pkl"))

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

    def initializeArucoParameters(self):
        """
        Function to initialize and declare parameters related to aruco servoing
        Also computes desired interaction matrix for aruco desired points
        """
        # Order:
        # [top_left, top_right, bottom_right, bottom_left]

        self.arucoTar_Z = 0.163
        self.arucoPose = None
        self.curArucoCorners = None
        self.cornerDepth = np.array([-1.0, -1.0, -1.0, -1.0])        
        self.desArucoCorners = np.array([
            [270, 196],
            [363, 196],
            [363, 288],
            [270, 289]
        ])

        # aruco object points
        self.markerLength = 0.025
        self.aruco_object_points = np.array([
            [-self.markerLength / 2, -self.markerLength / 2, 0],
            [self.markerLength / 2, -self.markerLength / 2, 0],
            [self.markerLength / 2, self.markerLength / 2, 0],
            [-self.markerLength / 2, self.markerLength / 2, 0]
        ])

        # compute aruco desired interaction matrix - 8x6 matrix
        tempLst = []
        for u, v in self.desArucoCorners:
            tempLst.append(
                self.computeInteractionMatrix(u, v, self.arucoTar_Z)
            )
        self.aruco_desiredIntMat = np.vstack(tempLst)
        self.aruco_currentIntMat = np.empty((8, 6))

    def initializeSocketHolesParameters(self):
        """
        Function to initialize and declare parameters related to socket hole servoing
        Also computes desired interaction matrix for socket holes desired points
        """
        
        self.holesTarZ = 0.208
        self.socketPose = None
        self.curSocketHoles = None
        self.holesDepth = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
        self.desSocketHoles = np.array([
            [339, 213],
            [317, 245],
            [365, 243],
            [315, 272],
            [369, 270]
        ])

        # socket holes object point
        self.socket_object_points = np.array([
            [0.0, -0.01075, 0],                     # NOTE: in here we need the pose of this point (1) - in pose estimation we omit this
            [-0.00825, 0.0, 0],
            [0.00825, 0.0, 0],
            [-0.00955, 0.01075, 0],
            [0.00955, 0.01075, 0]
        ])

        # compute socket holes desired interaction matrix - 10x6 matrix
        tempLst = []
        for u, v in self.desSocketHoles:
            tempLst.append(
                self.computeInteractionMatrix(u, v, self.holesTarZ)
            )
        self.holes_desiredIntMat = np.vstack(tempLst)
        self.holes_currentIntMat = np.empty((10, 6))

    def corners_sub_cb(self, msg):
        """Callback function to read current aruco corners"""
        
        if (msg.top_left is not None and msg.top_left[0] != -1):
            self.curArucoCorners = np.array([
                msg.top_left,
                msg.top_right,
                msg.bottom_right,
                msg.bottom_left
            ])
        else:
            self.curArucoCorners = None

    def aruco_pose_sub_cb(self, msg):
        """Callback function to read current aruco pose"""

        if (msg.position is not None and msg.position.x != -1.0):
            # pose extraction
            self.arucoPose = np.eye(4)
            self.arucoPose[0:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
            rotm = Rotation.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]).as_matrix()
            self.arucoPose[0:3, 0:3] = rotm
        else:
            self.arucoPose = None

    def socket_pose_cb(self, msg):
        """Callback function to read current socket holes pose"""

        if (msg.position is not None and msg.position.x != -1.0):
            # pose extraction
            self.socketPose = np.eye(4)
            self.socketPose[0:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
            rotm = Rotation.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]).as_matrix()
            self.socketPose[0:3, 0:3] = rotm
        else:
            self.socketPose = None

    def matched_points_cb(self, msg):
        """Callback function to extract matched points (socket holes)"""

        if (msg.data is not None and (msg.data[0] != -1)):
            # extract plug holes coordinates
            self.curSocketHoles = np.array(msg.data).reshape((5, 2))
        else:
            self.curSocketHoles = None
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
    
    def setAdaptiveGain(self, lam_0, lam_inf, lam_s0, pixelVel):
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
        x_norm = 0.0
        for err in pixelVel:
            abs_err = abs(err[0])
            if (abs_err > x_norm):
                x_norm = abs_err 

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


    def controlLoop(self):
        """Main timer function to compute eeVel at that time stamp and move the arm"""

        pass



def main():
    rclpy.init()


if __name__ == "__main__":
    main()