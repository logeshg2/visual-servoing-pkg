#!/usr/bin/env python3

"""
Behaviour Script to perform visual servoinng based and compute target velocity.
Only IBVS approach is implemented here.
"""

"""
TODO:
1. Separate SDK - like package for visual servoing -> this behaviour is specificall for plug insertion (need to write generic one)
2. need to write generic pixel vel computation + interaction matrix computation
3. need to implement plug hole ibvs also (right now only aruco ibvs is implemented)
4. 
"""


import pickle
import py_trees
import numpy as np
from move_arm import ControlType

class VisualServoing(py_trees.behaviour.Behaviour):
    def __init__(self):
        self.name = "visual_servoing"
        super(VisualServoing, self).__init__(self.name)

        # feature variables
        self.desArucoCorners = np.empty((4, 2))
        self.curArucoCorners = np.empty((4, 2))
        self.arucoTar_Z = None
        self.cornerDepth = None

        # yolo plug hole variables
        self.curHoles = None
        self.depthImg = None

        # aruco parameters
        self.arucoPose = None
        self.markerLength = None
        self.object_points = np.empty((4, 3))

        # interaction matrix (for both aruco + plug hole)
        self.desiredIntMat_1 = None     # _1 - aruco
        self.currentIntMat_1 = None
        self.desiredIntMat_2 = None     # _2 - plug hole
        self.currentIntMat_2 = None

        # camera calibration parameters
        self.K = pickle.load(open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/camera_matrix_rs.pkl", "rb"))
        self.Kinv = np.linalg.inv(self.K)

        # control varialbes
        self.lambdaVar = None
        self.triggered = False
        self.converged = False
        self.servoAruco = True
        self.servoSocket = False
        self.camVel = None
        self.controlMode = ControlType.camVelCtrl

        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self):
        # target feature points
        # aruco
        self.desArucoCorners = np.array(
            [270, 196],
            [363, 196],
            [363, 288],
            [270, 289]
        )
        self.arucoTar_Z = 0.163
        self.cornerDepth = np.array([-1.0, -1.0, -1.0, -1.0])       # [topLeft, topRight, bottomRight, bottomLeft]
        # yolo holes



        # aruco parameters
        self.markerLength = 0.025
        self.object_points = np.array([
            [-self.markerLength / 2, -self.markerLength / 2, 0],
            [self.markerLength / 2, -self.markerLength / 2, 0],
            [self.markerLength / 2, self.markerLength / 2, 0],
            [-self.markerLength / 2, self.markerLength / 2, 0]
        ])

        # desired interaction computation (constant)
        self.lambdaVar = 0.3
        self.camVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.computeDesiredInteractionMat()

    def computeDesiredInteractionMat(self):
        """
        Function to compute desired aruco corner points interaction matrix + interaction matrix from plug hole
        This matrix computed will be used for `Approximation of Interaction Matrix`.
        """

        # aruco
        # desire aruco points (corners) interaction matrix
        p1_jac = self.computeInteractionMatrix(self.desArucoCorners[0][0], self.desArucoCorners[0][1], self.arucoTar_Z)
        p2_jac = self.computeInteractionMatrix(self.desArucoCorners[1][0], self.desArucoCorners[1][1], self.arucoTar_Z)
        p3_jac = self.computeInteractionMatrix(self.desArucoCorners[2][0], self.desArucoCorners[2][1], self.arucoTar_Z)
        p4_jac = self.computeInteractionMatrix(self.desArucoCorners[3][0], self.desArucoCorners[3][1], self.arucoTar_Z)
        # points jacobian (for all for points) - 8x6 matrix
        # desired points interaction matrix
        self.desiredIntMat_1 = np.vstack([p1_jac, p2_jac, p3_jac, p4_jac])

        # plug_hole
        ##

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
        
        - e_vec : np.ndarray(), dtype=float64, error vector
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

    def initialise(self):
        """Reading dynamic variables and parameters from blackboard"""

        # read from blackboard
        self.triggered = self.blackboard.get("triggered")
        self.arucoPose = self.blackboard.get("arucoPose")
        self.curHoles = self.blackboard.get("curHoles")
        self.depthImg = self.blackboard.get("depthImg")
        self.curArucoCorners[0] = self.blackboard.get("cur_top_left")
        self.curArucoCorners[1] = self.blackboard.get("cur_top_right")
        self.curArucoCorners[2] = self.blackboard.get("cur_bottom_right")
        self.curArucoCorners[3] = self.blackboard.get("cur_bottom_left")

        # cam velocity based control
        self.controlMode = ControlType.camVelCtrl
        self.blackboard.set("controlMode", self.controlMode)

    def update(self):
        """Compute and perform IBVS"""

        # if not triggered (wait)
        if (not self.triggered):
            return py_trees.common.Status.SUCCESS
        
        if (self.servoAruco and not self.arucoPose):
            return py_trees.common.Status.FAILURE
        
        # servo on aruco
        if (self.arucoPose):
            # compute depth corners
            self.computeCornerDepth()

        # compute pixel velocity
        top_left_pix_vel = self.computeImgPointVel(self.desArucoCorners[0], self.curArucoCorners[0])
        top_right_pix_vel = self.computeImgPointVel(self.desArucoCorners[1], self.curArucoCorners[1])
        bottom_right_pix_vel = self.computeImgPointVel(self.desArucoCorners[2], self.curArucoCorners[2])
        bottom_left_pix_vel = self.computeImgPointVel(self.desArucoCorners[3], self.curArucoCorners[3])
        # flatten pixel velocities (here - 8x1 vector)
        pixelVel = np.array([top_left_pix_vel.flatten(), top_right_pix_vel.flatten(), bottom_right_pix_vel.flatten(), bottom_left_pix_vel.flatten()])
        pixelVel = np.array([pixelVel.flatten()]).reshape((8, 1))    # 8x1

        # compute interation matrix (combain all for points interation matrix)
        top_left_jacob = self.computeInteractionMatrix(self.curArucoCorners[0][0], self.curArucoCorners[0][1], self.cornerDepth[0])
        top_right_jacob = self.computeInteractionMatrix(self.curArucoCorners[1][0], self.curArucoCorners[1][1], self.cornerDepth[1])
        bottom_right_jacob = self.computeInteractionMatrix(self.curArucoCorners[2][0], self.curArucoCorners[2][1], self.cornerDepth[2])
        bottom_left_jacob = self.computeInteractionMatrix(self.curArucoCorners[3][0], self.curArucoCorners[3][1], self.cornerDepth[3])
        self.currentIntMat_1 = np.vstack([top_left_jacob, top_right_jacob, bottom_right_jacob, bottom_left_jacob])

        # Approximation of Interaction Matrix   (8x6)
        approxIntMat = (self.currentIntMat_1 + self.desiredIntMat_1) / 2

        # Adaptive gain (lambda_adapt)
        self.setAdaptiveGain(0.5, 0.3, 30.0, pixelVel)           # default - [1.666, 0.666, 1.666] 
        # tuning adaptive gain parameter using constant lambda
        # self.lambdaVar = 0.3                              # uncomment and tune lambda 0, and inf

        # compute camVel 
        self.camVel = -1 * self.lambdaVar * (np.linalg.pinv(approxIntMat) @ pixelVel)        # (6x1) = (6x8) @ (8x1)
        self.camVel = self.camVel.flatten()     # (6,)

        # update blackboard
        self.blackboard.set("controlMode", ControlType.camVelCtrl)
        self.blackboard.set("camVel", self.camVel)

    def terminate(self):
        pass