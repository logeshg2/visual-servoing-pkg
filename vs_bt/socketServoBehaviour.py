#!/usr/bin/env python3

"""Contains multiple behaviours for socket servoing composite or InsertSubtree composite."""

import time
import py_trees
import numpy as np
from scipy.spatial.transform import Rotation


class startSocketServo(py_trees.behaviour.Behaviour):
    def __init__(self):
        self.name = "startSocketServo"
        super(startSocketServo, self).__init__(self.name)

        self.servoTask = "servo_socket"

        self.blackboard = py_trees.blackboard.Blackboard()

    def update(self):
        
        self.logger.info(f"Starting Servoing: {self.servoTask}")

        # set servoTask (to 'servo_socket') - to begin socket servoing
        self.blackboard.set("servoTask", self.servoTask)

        return py_trees.common.Status.SUCCESS
    

class servoSocketUntilConv(py_trees.behaviour.Behaviour):
    def __init__(self):
        self.name = "servoSocketUntilConv"
        super(servoSocketUntilConv, self).__init__(self.name)

        self.socketConverged = False

        self.blackboard = py_trees.blackboard.Blackboard()
    
    def initialise(self):
        
        self.socketConverged = False
    
    def update(self):

        # get socket conv status from blackboard
        self.socketConverged = self.blackboard.get("converged_socket")

        if (self.socketConverged):
            self.logger.info("Socket servoing converged")
            return py_trees.common.Status.SUCCESS

        self.logger.info("Waiting to converge")

        return py_trees.common.Status.RUNNING


class stopSocketServo(py_trees.behaviour.Behaviour):
    def __init__(self):
        self.name = "stopSocketServo"
        super(stopSocketServo, self).__init__(self.name)

        self.servoTask = "no_servo"

        self.blackboard = py_trees.blackboard.Blackboard()

    def update(self):

        # set servoTask (to 'no_servo') - to stop socket servoing
        self.blackboard.set("servoTask", self.servoTask)

        self.logger.info(f"Stopping socket servoing: {self.servoTask}")

        return py_trees.common.Status.SUCCESS



# insertion behaviour's

class moveAboveInsert(py_trees.behaviour.Behaviour):
    def __init__(self, realRobot):
        self.name = "moveAboveInsert"
        super(moveAboveInsert, self).__init__(self.name)

        # box conv to insert above
        self.bcTia = np.eye(4)
        self.bcTia[0:3, 3] = np.array([0.06289426, -0.02071626,  0.07633251])
        self.bcTia[0:3, 0:3] = Rotation.from_euler("xyz", [3.81658362e-04, -1.56423169e-03,  3.92872137e-05], degrees=True).as_matrix()

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
            bTe_tar = bTe @ self.bcTia

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
            self.logger.error("Wait for 2 seconds here")
            return py_trees.common.Status.RUNNING

        # check motion
        if (self.isMoving == True):     
            self.logger.info(f"Arm moving to insert above pose")
            return py_trees.common.Status.RUNNING
        
        # check whether the arm reached target position
        curPose = np.array(self.bot.read_current_cartesian_pose())
        if (np.linalg.norm(curPose - self.targetPosition, ord=np.inf) > 2.0):        # giving 2.0 mm tolerance
            self.logger.warning("Position did not reach! again sending goal")
            self.goal_sent = False
            return py_trees.common.Status.RUNNING

        self.logger.info(f"Arm reached insert above pose!")
        return py_trees.common.Status.SUCCESS


class moveDownInsert(py_trees.behaviour.Behaviour):
    def __init__(self, realRobot):
        self.name = "moveDownInsert"
        super(moveDownInsert, self).__init__(self.name)

        # insert above to insert
        self.iaTi = np.eye(4)
        self.iaTi[0:3, 3] = np.array([0.0, 0.0, 0.029])

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
            bTe_tar = bTe @ self.iaTi

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
            self.logger.info(f"Arm moving to insert pose")
            return py_trees.common.Status.RUNNING
        
        # check whether the arm reached target position
        curPose = np.array(self.bot.read_current_cartesian_pose())
        if (np.linalg.norm(curPose - self.targetPosition, ord=np.inf) > 2.0):        # giving 2.0 mm tolerance
            self.logger.warning("Position did not reach! again sending goal")
            self.goal_sent = False
            return py_trees.common.Status.RUNNING

        self.logger.info(f"Arm reached insert pose!")
        return py_trees.common.Status.SUCCESS


class moveUpSafe_insert(py_trees.behaviour.Behaviour):
    def __init__(self, realRobot):
        self.name = "moveUpSafe_insert"
        super(moveUpSafe_insert, self).__init__(self.name)

        # insert to insert above (safe height)
        self.iTia_safe = np.eye(4)
        self.iTia_safe[0:3, 3] = np.array([0.0, 0.0, -0.05])

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
            bTe_tar = bTe @ self.iTia_safe

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
            self.logger.info(f"Arm moving to insert above safe pose")
            return py_trees.common.Status.RUNNING
        
        # check whether the arm reached target position
        curPose = np.array(self.bot.read_current_cartesian_pose())
        if (np.linalg.norm(curPose - self.targetPosition, ord=np.inf) > 2.0):        # giving 2.0 mm tolerance
            self.logger.warning("Position did not reach! again sending goal")
            self.goal_sent = False
            return py_trees.common.Status.RUNNING

        self.logger.info(f"Arm reached insert above safe pose!")
        return py_trees.common.Status.SUCCESS
