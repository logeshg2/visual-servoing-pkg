#!/usr/bin/env python3

"""
Behaviour Script to read required ros2 messages and write / update to py_trees blackboard.
This be behaviour will be called every time to update the blackboard.
"""

import py_trees
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from cv_bridge import CvBridge
from std_msgs.msg import String
from std_srvs.srv import SetBool
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup


class ReadfromROS(py_trees.behaviour.Behaviour):
    def __init__(self, node):
        self.name = "read_from_ros"
        super(ReadfromROS, self).__init__(self.name)

        # ros2 communication variables (initializing)
        self.node = node
        self.servoTask = "no_servo"
        self.converged = False
        self.triggered = False
        self.data_read_group = ReentrantCallbackGroup()
        self.cv_bridge = CvBridge()

        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self):
        """Setup ros2 nodes"""

        # set blackboard value to default
        self.blackboard.set("triggered", self.triggered)
        self.blackboard.set("converged", self.converged)
        self.blackboard.set("servoTask", self.servoTask)

        # ros2 subscription
        self.conv_status_sub = self.node.create_subscription(String, "/converged_status", self.conv_status_cb, 10, callback_group=self.data_read_group)

        # ros2 publishers
        self.servo_task_pub = self.node.create_publisher(String, "/servo_task", 10)

        # ros2 service
        self.trigger_srv = self.node.create_service(SetBool, '/trigger_servoing', self.trigger_servoing_cb)

    def conv_status_cb(self, msg):
        """Callback function to read convergence status from ros2 vs node"""

        if (msg.data is not None):
            self.converged = True if (msg.data == "1") else False
        else:
            self.converged = False

    def trigger_servoing_cb(self, request, response):
        if (request.data):
            self.triggered = True
        else:
            self.triggered = False

        response.success = True
        response.message = "trigger successful"
        return response

    def initialise(self):
        """Read from bt - that needs to be updated to ros2 nodes"""

        self.servoTask = self.blackboard.get("servoTask")

    def update(self):
        """Update blackboard"""
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)

            # set blackboard value to default
            self.blackboard.set("triggered", self.triggered)
            self.blackboard.set("converged", self.converged)

            # publish servoTask to ros2 VS node
            msg = String()
            msg.data = self.servoTask
            self.servo_task_pub.publish(msg)

            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self.node.get_logger().warn(f"Exceptino while updating blackboard: {e}")
            return py_trees.common.Status.FAILURE

    # def terminate(self):
    #     pass