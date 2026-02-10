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
        self.triggered = False
        self.servoAruco = False
        self.servoSocket = False
        self.converged = False
        self.moveToTracking = False
        self.performAlignment = False
        self.arucoPose = None
        self.eTc = None

        # set blackboard parameters
        self.blackboard.set("triggered", self.triggered)
        self.blackboard.set("servoAruco", self.servoAruco)
        self.blackboard.set("servoSocket", self.servoSocket)
        self.blackboard.set("converged", self.converged)
        self.blackboard.set("moveToTracking", self.moveToTracking)
        self.blackboard.set("performAlignment", self.performAlignment)

    def initialise(self):
        """Read blackboard"""

        self.eTc = self.blackboard.get("eTc")
        self.arucoPose = self.blackboard.get("arucoPose")
        self.converged = self.blackboard.get("converged")
        self.triggered = self.blackboard.get("triggered")
        self.servoAruco = self.blackboard.get("servoAruco")
        self.servoSocket = self.blackboard.get("servoSocket")
        self.moveToTracking = self.blackboard.get("moveToTracking")
        self.performAlignment = self.blackboard.get("performAlignment")

    def update(self):
        """Trigger operation | movements"""

        if (not self.triggered):
            return py_trees.common.Status.FAILURE
        elif ((self.triggered and not self.converged) and (self.servoAruco or self.servoSocket)):
            # success if servoing is in progress
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

                    # aruco to grip pose
                    aTo = np.eye(4)
                    aTo[0:3, 3] = np.array([0.004, 0.042, -0.025])
                    aTo[0:3, 0:3] = Rotation.from_euler("xyz", (0, 0, -90), degrees=True).as_matrix()
                    # camera to grip pose (self.arucoPose -> cTa)
                    cTo = self.arucoPose @ aTo

                    # ee to grip pose
                    eTo = self.eTc @ cTo
                    eTo[2, 3] -= 0.110

                    # base to object pose
                    bTo = bTe @ eTo

                    # compute pose as list
                    cartPose = np.empty(6)
                    cartPose[0:3] = bTo[0:3, 3] * 1000
                    cartPose[3:6] = Rotation.from_matrix(bTo[0:3, 0:3]).as_euler("xyz", degrees=True)
                    cartPose[3:5] = np.array([-179.9, 0.0])     # assuming the plug is perpendicular
                    print(cartPose)
                    print()
                    print(self.eTc)
                    print()
                    print(bTo)
                    exit(0)

                    # perform motion
                    self.blackboard.set("tarCartPos", cartPose)
                    self.blackboard.set("controlMode", ControlType.cartPosCtrl)

                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS
                
                elif (self.opr_count == 1):
                    # decrease z to pick (after aligning)
                    curPose = self.bot.read_current_cartesian_pose()
                    curPose[2] -= 17    # move 17 mm down
                    
                    # perform motion
                    self.blackboard.set("tarCartPos", curPose)
                    self.blackboard.set("controlMode", ControlType.cartPosCtrl)

                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS

                elif (self.opr_count == 2):
                    # close gripper
                    self.bot.air_gripper_control("close")
                    
                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS
                
                elif (self.opr_count == 3):
                    # move up in z
                    curPose = self.bot.read_current_cartesian_pose()
                    curPose[2] += 40
                
                    # perform motion
                    self.blackboard.set("tarCartPos", curPose)
                    self.blackboard.set("controlMode", ControlType.cartPosCtrl)

                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS

                elif (self.opr_count == 4):
                    # go back to tracking position
                    tracking_pos = self.blackboard.get("tracking_pose")
                    self.blackboard.set("controlMode", ControlType.jntPosCtrl)
                    self.blackboard.set("tarJntPos", tracking_pos)

                    self.opr_count += 1

                    return py_trees.common.Status.SUCCESS
                
                else:
                    # picking operation is over
                    self.performAlignment = False
                    self.converged = False
                    self.blackboard.set("performAlignment", self.performAlignment)
                    self.blackboard.set("converged", self.converged)

            else:
                # insertion operation
                pass



        if ((self.triggered and not self.moveToTracking) and (not self.servoAruco and not self.servoSocket)):
            # move to tracking position (initial movement)
            tracking_pos = self.blackboard.get("tracking_pose")
            self.blackboard.set("controlMode", ControlType.cartPosCtrl)
            self.blackboard.set("tarCartPos", tracking_pos)
            self.blackboard.set("moveToTracking", True)
            
            return py_trees.common.Status.SUCCESS
        elif ((self.triggered and self.moveToTracking) and not self.servoAruco):
            # trigger aruco servoing
            self.blackboard.set("moveToTracking", False)
            self.blackboard.set("servoAruco", True)
            self.blackboard.set("controlMode", ControlType.camVelCtrl)

            return py_trees.common.Status.SUCCESS
        elif ((self.triggered and self.servoAruco) and self.converged):
            # trigger alignment and picking - convergence
            self.opr_count = 0      # reset operation counter
            self.performAlignment = True
            self.blackboard.set("performAlignment", self.performAlignment)
            
            # return running or failure to perform alignment procedure (in the same scipt)
            return py_trees.common.Status.RUNNING
        elif ((self.triggered and self.servoAruco)):
            # trigger socket holes servoing
            self.servoSocket = True
            self.servoAruco = False
            self.blackboard.set("servoSocket", self.servoSocket)
            self.blackboard.set("controlMode", ControlType.camVelCtrl)

            return py_trees.common.Status.SUCCESS
        elif ((self.triggered and self.servoSocket) and self.converged):
            # trigger alignment and insertion - convergence
            self.opr_count = 0      # reset operation counter
            self.performAlignment = True
            self.blackboard.set("performAlignment", self.performAlignment)
            
            # return running or failure to perform alignment procedure (in the same scipt)
            return py_trees.common.Status.RUNNING
        else:
            # not defined
            return py_trees.common.Status.FAILURE

    # def terminate(self):
    #     pass