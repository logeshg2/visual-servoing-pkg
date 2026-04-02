#!/usr/bin/env python3

"""
Behaviour Script to read motion parameter and wait until completion.
"""

import py_trees

class CheckMovement(py_trees.behaviour.Behaviour):
    def __init__(self, realRobot):
        self.name = "check_motion"
        super(CheckMovement, self).__init__(self.name)

        self.bot = realRobot
        self.isMoving = False

        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self):

        self.blackboard.set("isMoving", self.isMoving)

    def update(self):
        """Read current movement status from SDK and wait or proceed"""

        self.isMoving = self.bot.is_moving()
        self.blackboard.set("isMoving", self.isMoving)
        
        return py_trees.common.Status.RUNNING
