#!/usr/bin/env python3

"""
Behaviour Script to read motion parameter and wait until completion.
"""

import py_trees
from move_arm import ControlType

class CheckMovement(py_trees.behaviour.Behaviour):
    def __init__(self, realRobot):
        self.name = "check_motion"
        super(CheckMovement, self).__init__(self.name)

        self.controlMode = None
        self.bot = realRobot
        self.isMoving = False

        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self):
        self.controlMode = self.blackboard.get("controlMode")

    def initialise(self):
        self.controlMode = self.blackboard.get("controlMode")

    def update(self):
        """Read current movement status from SDK and wait or proceed"""

        self.isMoving = self.bot.is_moving()

        # for joint and cartesian position control mode - the motion should complete
        if (self.controlMode == ControlType.jntPosCtrl or self.controlMode == ControlType.cartPosCtrl):
            if (self.isMoving):
                return py_trees.common.Status.RUNNING
            return py_trees.common.Status.SUCCESS
        
        # NOTE: for velocity  controlled modes - we no need to wait for motion to complete
        return py_trees.common.Status.SUCCESS

    def terminate(self):
        pass