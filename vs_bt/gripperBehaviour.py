#!/usr/bin/env python3

"""Contains multiple behaviours for gripper control."""

import time
import py_trees


class openGripper(py_trees.behaviour.Behaviour):
    def __init__(self, realRobot):
        self.name = "openGripper"
        super(openGripper, self).__init__(self.name)

        self.waitTime = 1.0       # setting 1 sec wait
        self.startTime = None
        self.goal_sent = False
        self.bot = realRobot

    def initialise(self):
        
        self.startTime = None
        self.goal_sent = False

    def update(self):
        
        if (not self.goal_sent):
            # open gripper
            self.bot.air_gripper_control("open")
            # record start time
            self.startTime = time.time()
            
            self.goal_sent = True

            self.logger.info(f"Opening gripper")

            return py_trees.common.Status.RUNNING
        
        if ((time.time() - self.startTime) >= self.waitTime):
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING
    

class closeGripper(py_trees.behaviour.Behaviour):
    def __init__(self, realRobot):
        self.name = "closeGripper"
        super(closeGripper, self).__init__(self.name)

        self.waitTime = 1.0       # setting 1 sec wait
        self.startTime = None
        self.goal_sent = False
        self.bot = realRobot

    def initialise(self):
        
        self.startTime = None
        self.goal_sent = False

    def update(self):
        
        if (not self.goal_sent):
            # close gripper
            self.bot.air_gripper_control("close")
            # record start time
            self.startTime = time.time()
            
            self.goal_sent = True

            self.logger.info(f"Closing gripper")

            return py_trees.common.Status.RUNNING
        
        if ((time.time() - self.startTime) >= self.waitTime):
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING