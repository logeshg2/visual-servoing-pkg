#!/usr/bin/env python3

"""Contains multiple behaviours for aruco servoing composite or PickSubtree composite."""

import time
import py_trees
import numpy as np
from scipy.spatial.transform import Rotation


class startArucoServo(py_trees.behaviour.Behaviour):
    def __init__(self):
        self.name = "startArucoServo"
        super(startArucoServo, self).__init__(self.name)

        self.servoTask = "servo_aruco"

        self.blackboard = py_trees.blackboard.Blackboard()

    def update(self):

        self.logger.info(f"Starting servo: {self.servoTask}")

        # set servoTask (to 'servo_aruco') - to begin aruco servoing
        self.blackboard.set("servoTask", self.servoTask)

        return py_trees.common.Status.SUCCESS
    

class servoArucoUntilConv(py_trees.behaviour.Behaviour):
    def __init__(self):
        self.name = "servoArucoUntilConv"
        super(servoArucoUntilConv, self).__init__(self.name)

        self.arucoConverged = False

        self.blackboard = py_trees.blackboard.Blackboard()
    
    def initialise(self):
        
        self.arucoConverged = False
    
    def update(self):

        # get aruco conv status from blackboard
        self.arucoConverged = self.blackboard.get("converged_aruco")

        if (self.arucoConverged):
            self.logger.info("Aruco servoing converged")
            return py_trees.common.Status.SUCCESS

        self.logger.info("Waiting to converge")

        return py_trees.common.Status.RUNNING


class stopArucoServo(py_trees.behaviour.Behaviour):
    def __init__(self):
        self.name = "stopArucoServo"
        super(stopArucoServo, self).__init__(self.name)

        self.servoTask = "no_servo"

        self.blackboard = py_trees.blackboard.Blackboard()

    def update(self):

        # set servoTask (to 'no_servo') - to stop aruco servoing
        self.blackboard.set("servoTask", self.servoTask)

        self.logger.info(f"Stopping aruco servoing: {self.servoTask}")

        return py_trees.common.Status.SUCCESS



# pick behaviour's

class moveAbovePick(py_trees.behaviour.Behaviour):
    def __init__(self, realRobot):
        self.name = "moveAbovePick"
        super(moveAbovePick, self).__init__(self.name)

        # aruco conv to pick above
        self.acTpa = np.eye(4)
        self.acTpa[0:3, 3] = np.array([0.07281911, -0.03435075,  0.0975986])
        self.acTpa[0:3, 0:3] = Rotation.from_euler("xyz", [2.25463710e-05, -1.07034064e-02,  2.62626922e-03], degrees=True).as_matrix()

        self.bot = realRobot
        self.isMoving = None
        self.goal_sent = False
        self.targetPosition = None

        self.waitTime = 2.0
        self.startTime = None

        self.blackboard = py_trees.blackboard.Blackboard()
    
    def initialise(self):
        
        self.goal_sent = False
        self.startTime = None
        self.targetPosition = None
        self.isMoving = self.blackboard.get("isMoving")

    def update(self):
        
        self.isMoving = self.blackboard.get("isMoving")

        if (not self.goal_sent):
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

            self.bot.write_cartesian_position(coords=cartPose, blocking=False)
            self.goal_sent = True
            self.targetPosition = np.array(cartPose)
            self.startTime = time.time()

            return py_trees.common.Status.RUNNING
        
        # wait
        if (time.time() - self.startTime) < self.waitTime:
            return py_trees.common.Status.RUNNING

        # check motion
        if (self.isMoving == True):     
            self.logger.info(f"Arm moving to pick above pose")
            return py_trees.common.Status.RUNNING
        
        # check whether the arm reached target position
        curPose = np.array(self.bot.read_current_cartesian_pose())
        if (np.linalg.norm(curPose - self.targetPosition, ord=np.inf) > 2.0):        # giving 2.0 mm tolerance
            self.logger.warning("Position did not reach! again sending goal")
            self.goal_sent = False
            return py_trees.common.Status.RUNNING

        self.logger.info(f"Arm reached pick above pose!")
        return py_trees.common.Status.SUCCESS


class moveDownPick(py_trees.behaviour.Behaviour):
    def __init__(self, realRobot):
        self.name = "moveDownPick"
        super(moveDownPick, self).__init__(self.name)

        # pick above to pick
        self.paTp = np.eye(4)
        self.paTp[0:3, 3] = np.array([0.0, 0.0, 0.0135])

        self.bot = realRobot
        self.isMoving = None
        self.goal_sent = False
        self.targetPosition = None

        self.waitTime = 2.0
        self.startTime = None

        self.blackboard = py_trees.blackboard.Blackboard()
    
    def initialise(self):
        
        self.goal_sent = False
        self.startTime = None
        self.targetPosition = None
        self.isMoving = self.blackboard.get("isMoving")

    def update(self):
        
        self.isMoving = self.blackboard.get("isMoving")

        if (not self.goal_sent):
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

            self.bot.write_cartesian_position(coords=cartPose, blocking=False)
            self.goal_sent = True
            self.targetPosition = np.array(cartPose)
            self.startTime = time.time()

            return py_trees.common.Status.RUNNING
        
        # wait
        if (time.time() - self.startTime) < self.waitTime:
            return py_trees.common.Status.RUNNING

        # check motion
        if (self.isMoving == True):     
            self.logger.info(f"Arm moving to pick pose")
            return py_trees.common.Status.RUNNING
        
        # check whether the arm reached target position
        curPose = np.array(self.bot.read_current_cartesian_pose())
        if (np.linalg.norm(curPose - self.targetPosition, ord=np.inf) > 2.0):        # giving 2.0 mm tolerance
            self.logger.warning("Position did not reach! again sending goal")
            self.goal_sent = False
            return py_trees.common.Status.RUNNING

        self.logger.info(f"Arm reached pick pose!")
        return py_trees.common.Status.SUCCESS


class moveUpSafe(py_trees.behaviour.Behaviour):
    def __init__(self, realRobot):
        self.name = "moveUpSafe"
        super(moveUpSafe, self).__init__(self.name)

        # pick to pick above (safe height)
        self.pTpa_safe = np.eye(4)
        self.pTpa_safe[0:3, 3] = np.array([0.0, 0.0, -0.04])

        self.bot = realRobot
        self.isMoving = None
        self.goal_sent = False
        self.targetPosition = None

        self.waitTime = 2.0
        self.startTime = None

        self.blackboard = py_trees.blackboard.Blackboard()
    
    def initialise(self):
        
        self.goal_sent = False
        self.startTime = None
        self.targetPosition = None
        self.isMoving = self.blackboard.get("isMoving")

    def update(self):
        
        self.isMoving = self.blackboard.get("isMoving")

        if (not self.goal_sent):
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

            self.bot.write_cartesian_position(coords=cartPose, blocking=False)
            self.goal_sent = True
            self.targetPosition = np.array(cartPose)
            self.startTime = time.time()

            return py_trees.common.Status.RUNNING
        
        if (time.time() - self.startTime) < self.waitTime:
            return py_trees.common.Status.RUNNING

        # check motion
        if (self.isMoving == True):     
            self.logger.info(f"Arm moving to pick above safe pose")
            return py_trees.common.Status.RUNNING
        
        # check whether the arm reached target position
        curPose = np.array(self.bot.read_current_cartesian_pose())
        if (np.linalg.norm(curPose - self.targetPosition, ord=np.inf) > 2.0):        # giving 2.0 mm tolerance
            self.logger.warning("Position did not reach! again sending goal")
            self.goal_sent = False
            return py_trees.common.Status.RUNNING

        self.logger.info(f"Arm reached pick above safe pose!")
        return py_trees.common.Status.SUCCESS
