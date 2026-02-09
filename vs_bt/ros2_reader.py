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
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from std_msgs.msg import Int64MultiArray
from visual_servoing_pkg.msg import ArucoCorner
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup


class ReadfromROS(py_trees.behaviour.Behaviour):
    def __init__(self, node):
        self.name = "read_from_ros"
        super(ReadfromROS, self).__init__(self.name)

        # ros2 communication variables (initializing)
        self.node = node
        self.arucoPose = None
        self.cur_top_left = None
        self.cur_top_right = None
        self.cur_bottom_right = None
        self.cur_bottom_left = None
        self.curHoles = None
        self.depthImg = None
        self.triggered = False
        self.data_read_group = ReentrantCallbackGroup()
        self.cv_bridge = CvBridge()

        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self):
        """Setup ros2 nodes"""

        # set blackboard value to default
        self.blackboard.set("triggered", self.triggered)
        # aruco
        self.blackboard.set("cur_top_left", self.cur_top_left)
        self.blackboard.set("cur_top_right", self.cur_top_right)
        self.blackboard.set("cur_bottom_right", self.cur_bottom_right)
        self.blackboard.set("cur_bottom_left", self.cur_bottom_left)
        self.blackboard.set("arucoPose", self.arucoPose)
        # plug hole
        self.blackboard.set("curHoles", self.curHoles)
        self.blackboard.set("depthImg", self.depthImg)

        # ros2 subscription
        self.aruco_corner_sub = self.node.create_subscription(ArucoCorner, "/aruco_corners", self.corners_sub_cb, 10, callback_group=self.data_read_group)
        self.aruco_pose_sub = self.node.create_subscription(Pose, "/aruco_pose", self.pose_sub_cb, 10, callback_group=self.data_read_group)
        self.matchPoints_sub = self.node.create_subscription(Int64MultiArray, "/holes_coord", self.matched_points_cb, 10, callback_group=self.data_read_group)
        self.depthImg_sub = self.node.create_subscription(Image, "/camera/camera/depth/image_rect_raw", self.depthImg_cb, 10, callback_group=self.data_read_group)

        # ros2 service
        self.trigger_srv = self.node.create_service(SetBool, '/trigger_servoing', self.trigger_servoing_cb)

    def corners_sub_cb(self, msg):
        if (msg.top_left is not None):
            self.cur_top_left = msg.top_left
            self.cur_top_right = msg.top_right
            self.cur_bottom_right = msg.bottom_right
            self.cur_bottom_left = msg.bottom_left
        else:
            self.cur_top_left = None
            self.cur_top_right = None
            self.cur_bottom_right = None
            self.cur_bottom_left = None

    def pose_sub_cb(self, msg):
        if (msg.position is not None and msg.position.x != -1.0):
            # pose extraction
            self.arucoPose = np.eye(4)
            self.arucoPose[0:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
            rotm = Rotation.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]).as_matrix()
            self.arucoPose[0:3, 0:3] = rotm
        else:
            self.arucoPose = None

    def matched_points_cb(self, msg):
        """Callback function to extract matched points from the ros2 custom message"""

        if (msg.data is not None and (msg.data[0] != -1)):
            # extract plug holes coordinates
            self.curHoles = np.array(msg.data).reshape((5, 2))
        else:
            self.curHoles = None
            # self.get_logger().warn(f"Matched points published are not enough!")

    def depthImg_cb(self, msg):
        """Callback function to extract depth image from ros2 image message"""
        
        if (msg is not None):
            self.depthImg = self.cv_bridge.imgmsg_to_cv2(msg)
            self.depthImg = np.float64(self.depthImg) / 1000.0          # in meters
        else:
            self.depthImg = None
            self.node.get_logger().warn(f"Depth image msg is None!")

    def trigger_servoing_cb(self, request, response):
        if (request.data):
            self.triggered = True
        else:
            self.triggered = False
        
        # go to tracking position
        # self.bot.write_cartesian_position(coords=self.tracking_pose, blocking=False)
        # time.sleep(2)
        # while (self.bot.is_moving()):
        #     time.sleep(0.1)

        response.success = True
        response.message = "trigger successful"
        return response

    def initialise(self):
        pass

    def update(self):
        """Update blackboard"""
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)

            # set blackboard value to default
            self.blackboard.set("triggered", self.triggered)

            # aruco
            self.blackboard.set("cur_top_left", self.cur_top_left)
            self.blackboard.set("cur_top_right", self.cur_top_right)
            self.blackboard.set("cur_bottom_right", self.cur_bottom_right)
            self.blackboard.set("cur_bottom_left", self.cur_bottom_left)
            self.blackboard.set("arucoPose", self.arucoPose)

            # plug hole
            self.blackboard.set("curHoles", self.curHoles)
            self.blackboard.set("depthImg", self.depthImg)

            return py_trees.common.Status.RUNNING
        except Exception as e:
            self.node.get_logger().warn(f"Exceptino while updating blackboard: {e}")
            return py_trees.common.Status.FAILURE

    def terminate(self):
        pass