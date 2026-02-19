#!/usr/bin/env python3

"""
Behaviour script to handle the motion like triggering next operation to perform.
"""

import py_trees
import numpy as np
from move_arm import ControlType
from scipy.spatial.transform import Rotation


class Handler(py_trees.behaviour.Behaviour):
    def __init__(self, realRobot):
        self.name = "opr_handler"
        super(Handler, self).__init__(self.name)

        self.bot = realRobot
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self):

        # operation variables
        self.opr_count = 0
        self.servoTask = None
        self.triggered = False
        self.servoAruco = False
        self.servoSocket = False
        self.converged_aruco = False
        self.converged_socket = False
        self.moveToTracking = False
        self.performAlignment = False

        # declare static transforms
        self.declareStaticTransforms()

        # set blackboard parameters
        self.blackboard.set("triggered", self.triggered)
        self.blackboard.set("servoAruco", self.servoAruco)
        self.blackboard.set("servoSocket", self.servoSocket)
        self.blackboard.set("moveToTracking", self.moveToTracking)
        self.blackboard.set("performAlignment", self.performAlignment)

    def declareStaticTransforms(self):
        """Function to declare static tranforms (that is movement after convergence)"""

        # aruco conv to pick above
        self.acTpa = np.eye(4)
        self.acTpa[0:3, 3] = np.array([0.02653083, -0.03003649, 0.08005753])
        self.acTpa[0:3, 0:3] = Rotation.from_euler("xyz", [3.62705053, 1.10947527, 6.38303279], degrees=True).as_matrix()

        # pick above to pick
        self.paTp = np.eye(4)
        self.paTp[0:3, 3] = np.array([0.0, 0.0, 0.015])

        # pick to pick above (safe height)
        self.pTpa_safe = np.eye(4)
        self.pTpa_safe[0:3, 3] = np.array([0.0, 0.0, -0.04])


        # box conv to insert above
        self.bcTia = np.eye(4)
        self.bcTia[0:3, 3] = np.array([0.06289426, -0.02071626,  0.07633251])
        self.bcTia[0:3, 0:3] = Rotation.from_euler("xyz", [3.81658362e-04, -1.56423169e-03,  3.92872137e-05], degrees=True).as_matrix()

        # insert above to insert
        self.iaTi = np.eye(4)
        self.iaTi[0:3, 3] = np.array([0.0, 0.0, 0.029])

        # insert to insert above (safe height)
        self.iTia_safe = np.eye(4)
        self.iTia_safe[0:3, 3] = np.array([0.0, 0.0, -0.05])

    def initialise(self):
        """Read blackboard"""

        self.servoTask = self.blackboard.get("servoTask")
        self.converged_aruco = self.blackboard.get("converged_aruco")
        self.converged_socket = self.blackboard.get("converged_socket")
        self.triggered = self.blackboard.get("triggered")
        self.servoAruco = self.blackboard.get("servoAruco")
        self.servoSocket = self.blackboard.get("servoSocket")
        self.moveToTracking = self.blackboard.get("moveToTracking")
        self.performAlignment = self.blackboard.get("performAlignment")

    def update(self):
        """Trigger operation | movements"""
        # TODO : proper logic

        if (not self.triggered):
            return py_trees.common.Status.FAILURE
        elif (not self.performAlignment) and ((self.triggered and (not self.converged_aruco and not self.converged_socket)) and (self.servoAruco or self.servoSocket)):
            # success if servoing is in progress
            self.blackboard.set("controlMode", ControlType.camVelCtrl)
            return py_trees.common.Status.SUCCESS
        elif (self.triggered and self.performAlignment):
            # perform alignment and perform operation (pick | insert)
            if (self.servoAruco and not self.servoSocket):
                # picking operation
                if (self.opr_count == 0):
                    # current base to ee pose
                    curPose = self.bot.read_current_cartesian_pose()
                    bTe = np.eye(4)
                    bTe[0:3, 3] = np.array(curPose[0:3]) / 1000.0
                    bTe[0:3, 0:3] = Rotation.from_euler("xyz", np.array(curPose[3:6]), degrees=True).as_matrix()

                    # target ee pose (after static transformation)
                    bTe_tar = bTe @ self.acTpa

                    # compute pose as list
                    cartPose = np.empty(6)
                    cartPose[0:3] = bTe_tar[0:3, 3] * 1000
                    cartPose[3:6] = Rotation.from_matrix(bTe_tar[0:3, 0:3]).as_euler("xyz", degrees=True)

                    # perform motion
                    self.blackboard.set("tarCartPos", cartPose)
                    self.blackboard.set("controlMode", ControlType.cartPosCtrl)

                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS
                
                elif (self.opr_count == 1):
                    # current base to ee pose
                    curPose = self.bot.read_current_cartesian_pose()
                    bTe = np.eye(4)
                    bTe[0:3, 3] = np.array(curPose[0:3]) / 1000.0
                    bTe[0:3, 0:3] = Rotation.from_euler("xyz", np.array(curPose[3:6]), degrees=True).as_matrix()
                    
                    # target ee pose (after static transformation)
                    bTe_tar = bTe @ self.paTp

                    # compute pose as list
                    cartPose = np.empty(6)
                    cartPose[0:3] = bTe_tar[0:3, 3] * 1000
                    cartPose[3:6] = Rotation.from_matrix(bTe_tar[0:3, 0:3]).as_euler("xyz", degrees=True)

                    # perform motion
                    self.blackboard.set("tarCartPos", cartPose)
                    self.blackboard.set("controlMode", ControlType.cartPosCtrl)

                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS

                elif (self.opr_count == 2):
                    # close gripper
                    self.bot.air_gripper_control("close")
                    
                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS
                
                elif (self.opr_count == 3):
                    # current base to ee pose
                    curPose = self.bot.read_current_cartesian_pose()
                    bTe = np.eye(4)
                    bTe[0:3, 3] = np.array(curPose[0:3]) / 1000.0
                    bTe[0:3, 0:3] = Rotation.from_euler("xyz", np.array(curPose[3:6]), degrees=True).as_matrix()
                    
                    # target ee pose (after static transformation)
                    bTe_tar = bTe @ self.pTpa_safe

                    # compute pose as list
                    cartPose = np.empty(6)
                    cartPose[0:3] = bTe_tar[0:3, 3] * 1000
                    cartPose[3:6] = Rotation.from_matrix(bTe_tar[0:3, 0:3]).as_euler("xyz", degrees=True)
                
                    # perform motion
                    self.blackboard.set("tarCartPos", cartPose)
                    self.blackboard.set("controlMode", ControlType.cartPosCtrl)

                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS

                elif (self.opr_count == 4):
                    # go back to tracking position
                    tracking_pos = self.blackboard.get("tracking_pose")
                    self.blackboard.set("controlMode", ControlType.cartPosCtrl)
                    self.blackboard.set("tarCartPos", tracking_pos)

                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS
                
                else:
                    # picking operation is over
                    self.performAlignment = False
                    # self.converged = False
                    self.blackboard.set("performAlignment", self.performAlignment)
                    # self.blackboard.set("converged", self.converged)

            else:
                # insertion operation
                if (self.opr_count == 0):
                    # current base to ee pose
                    curPose = self.bot.read_current_cartesian_pose()
                    bTe = np.eye(4)
                    bTe[0:3, 3] = np.array(curPose[0:3]) / 1000.0
                    bTe[0:3, 0:3] = Rotation.from_euler("xyz", np.array(curPose[3:6]), degrees=True).as_matrix()

                    # target ee pose (after static transformation)
                    bTe_tar = bTe @ self.bcTia

                    # compute pose as list
                    cartPose = np.empty(6)
                    cartPose[0:3] = bTe_tar[0:3, 3] * 1000
                    cartPose[3:6] = Rotation.from_matrix(bTe_tar[0:3, 0:3]).as_euler("xyz", degrees=True)

                    # perform motion
                    self.blackboard.set("tarCartPos", cartPose)
                    self.blackboard.set("controlMode", ControlType.cartPosCtrl)

                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS
                
                elif (self.opr_count == 1):
                    # perform insertion
                    # current base to ee pose
                    curPose = self.bot.read_current_cartesian_pose()
                    bTe = np.eye(4)
                    bTe[0:3, 3] = np.array(curPose[0:3]) / 1000.0
                    bTe[0:3, 0:3] = Rotation.from_euler("xyz", np.array(curPose[3:6]), degrees=True).as_matrix()
                    
                    # target ee pose (after static transformation)
                    bTe_tar = bTe @ self.iaTi

                    # compute pose as list
                    cartPose = np.empty(6)
                    cartPose[0:3] = bTe_tar[0:3, 3] * 1000
                    cartPose[3:6] = Rotation.from_matrix(bTe_tar[0:3, 0:3]).as_euler("xyz", degrees=True)

                    # perform motion
                    self.blackboard.set("tarCartPos", cartPose)
                    self.blackboard.set("controlMode", ControlType.cartPosCtrl)

                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS
                
                elif (self.opr_count == 2):
                    # open gripper
                    self.bot.air_gripper_control("open")
                    
                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS
                
                elif (self.opr_count == 3):
                    # current base to ee pose
                    curPose = self.bot.read_current_cartesian_pose()
                    bTe = np.eye(4)
                    bTe[0:3, 3] = np.array(curPose[0:3]) / 1000.0
                    bTe[0:3, 0:3] = Rotation.from_euler("xyz", np.array(curPose[3:6]), degrees=True).as_matrix()
                    
                    # target ee pose (after static transformation)
                    bTe_tar = bTe @ self.iTia_safe

                    # compute pose as list
                    cartPose = np.empty(6)
                    cartPose[0:3] = bTe_tar[0:3, 3] * 1000
                    cartPose[3:6] = Rotation.from_matrix(bTe_tar[0:3, 0:3]).as_euler("xyz", degrees=True)
                
                    # perform motion
                    self.blackboard.set("tarCartPos", cartPose)
                    self.blackboard.set("controlMode", ControlType.cartPosCtrl)

                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS

                elif (self.opr_count == 4):
                    # go back to tracking position
                    tracking_pos = self.blackboard.get("tracking_pose")
                    self.blackboard.set("controlMode", ControlType.cartPosCtrl)
                    self.blackboard.set("tarCartPos", tracking_pos)

                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS

                else:
                    # picking operation is over
                    self.performAlignment = False
                    # self.converged = False
                    self.blackboard.set("performAlignment", self.performAlignment)
                    # self.blackboard.set("converged", self.converged)

                    # operation over
                    self.logger.info(f"Plug insertion operation is over.")
                    exit(0)


        if ((self.triggered and not self.moveToTracking) and (not self.servoAruco and not self.servoSocket)):
            # move to tracking position (initial movement)
            tracking_pos = self.blackboard.get("tracking_pose")
            self.blackboard.set("controlMode", ControlType.cartPosCtrl)
            self.blackboard.set("tarCartPos", tracking_pos)
            self.blackboard.set("moveToTracking", True)

            # open gripper
            self.bot.air_gripper_control("open")
            
            return py_trees.common.Status.SUCCESS
        elif ((self.triggered and self.moveToTracking) and not self.servoAruco):
            # trigger aruco servoing
            self.servoAruco = True
            self.servoTask = "servo_aruco"
            self.blackboard.set("servoTask", self.servoTask)
            self.blackboard.set("moveToTracking", False)
            self.blackboard.set("servoAruco", True)
            self.blackboard.set("controlMode", ControlType.camVelCtrl)

            return py_trees.common.Status.SUCCESS
        elif ((self.triggered and self.servoAruco) and self.converged_aruco):
            # trigger alignment and picking - convergence
            self.opr_count = 0      # reset operation counter
            self.performAlignment = True
            self.blackboard.set("performAlignment", self.performAlignment)
            self.servoTask = "no_servo"
            self.blackboard.set("servoTask", self.servoTask)
            
            # return running or failure to perform alignment procedure (in the same scipt)
            return py_trees.common.Status.FAILURE
        elif ((self.triggered and self.servoAruco)):
            # trigger socket holes servoing
            self.servoSocket = True
            self.servoAruco = False
            self.blackboard.set("servoSocket", self.servoSocket)
            self.blackboard.set("servoAruco", self.servoAruco)
            self.blackboard.set("controlMode", ControlType.camVelCtrl)
            self.servoTask = "servo_socket"
            self.blackboard.set("servoTask", self.servoTask)

            return py_trees.common.Status.SUCCESS
        elif ((self.triggered and self.servoSocket) and self.converged_socket):
            # trigger alignment and insertion - convergence
            self.opr_count = 0      # reset operation counter
            self.performAlignment = True
            self.blackboard.set("performAlignment", self.performAlignment)
            self.servoTask = "no_servo"
            self.blackboard.set("servoTask", self.servoTask)
            
            # return running or failure to perform alignment procedure (in the same scipt)
            return py_trees.common.Status.FAILURE
        else:
            # not defined
            self.servoTask = "no_servo"
            self.blackboard.set("servoTask", self.servoTask)
            return py_trees.common.Status.FAILURE

    # def terminate(self):
    #     pass