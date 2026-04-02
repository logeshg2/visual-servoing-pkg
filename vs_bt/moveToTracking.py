#!/usr/bin/env python3

"""
Behaviour Script to move the arm to tracking location.
Monitors trigger from ros2 service (this is also optional).
"""

import time
import py_trees
import numpy as np

class move2Tracking(py_trees.behaviour.Behaviour):
    def __init__(self, realRobot, look_for_trigger=True):
        self.name = "move_2_tracking"
        super(move2Tracking, self).__init__(self.name)

        self.bot = realRobot
        self.waitTime = 2.0
        self.startTime = None
        self.isMoving = None
        self.triggered = None
        self.goal_sent = False
        self.targetPosition = None
        self.look_for_trigger = look_for_trigger

        self.blackboard = py_trees.blackboard.Blackboard()

    def initialise(self):
        
        self.goal_sent = False
        self.triggered = False
        self.startTime = None
        self.targetPosition = None
        self.isMoving = self.blackboard.get("isMoving")
        self.triggered = self.blackboard.get("triggered")

    def update(self):

        self.isMoving = self.blackboard.get("isMoving")
        self.triggered = self.blackboard.get("triggered")

        if (not self.triggered and self.look_for_trigger):
            self.logger.warning(f"Not triggered yet!")
            
            return py_trees.common.Status.RUNNING

        if (not self.goal_sent):
            self.logger.info(f"Triggered, moving to tracking pose")
            
            # move the arm
            trackingPose = self.blackboard.get("tracking_pose")
            self.bot.write_cartesian_position(coords=trackingPose, blocking=False)
            self.targetPosition = np.array(trackingPose)

            self.startTime = time.time()
            self.goal_sent = True

            return py_trees.common.Status.RUNNING

        # wait for movement to start
        if (time.time() - self.startTime) < self.waitTime:
            return py_trees.common.Status.RUNNING

        # check motion
        if (self.isMoving == True):     
            self.logger.info(f"Arm moving to tracking pose")
            return py_trees.common.Status.RUNNING
        
        # check whether the arm reached target position
        curPose = np.array(self.bot.read_current_cartesian_pose())
        if (np.linalg.norm(curPose - self.targetPosition, ord=np.inf) > 2.0):        # giving 2.0 mm tolerance
            self.logger.warning("Position did not reach! again sending goal")
            self.goal_sent = False
            return py_trees.common.Status.RUNNING

        self.logger.info(f"Arm reached tracking pose!")
        return py_trees.common.Status.SUCCESS
