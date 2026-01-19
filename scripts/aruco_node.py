#!/usr/bin/env python3

"""
This is a ros2 node that read current image from the sensor and processes the image.
Aruco is used in this script.
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray
from visual_servoing_pkg.msg import ArucoCorner
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class ArucoNode(Node):
    def __init__(self):
        super().__init__("aruco_node")

        self.declare_parameter('camera_id', 2)
        self.declare_parameter('frame_width', 640.0)
        self.declare_parameter('frame_height', 480.0)
        self.cam_dev_id = self.get_parameter('camera_id').get_parameter_value().integer_value
        self.frame_width = self.get_parameter('frame_width').get_parameter_value().double_value
        self.frame_height = self.get_parameter('frame_height').get_parameter_value().double_value

        # camera device reader in another thread
        self.latest_image = None
        self.cam_obj = cv2.VideoCapture(self.cam_dev_id)
        self.cam_obj.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cam_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.frame_center = np.array([int(self.frame_width // 2), int(self.frame_height // 2)])

        # logging
        self.get_logger().info(f"Camera device id: {self.cam_dev_id}")
        self.get_logger().info(f"Image frame width: {self.frame_width}")
        self.get_logger().info(f"Image frame height: {self.frame_height}")

        # aruco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        self.aruco_center = None
        self.corner = None

        # target aruco points
        self.tar_top_left = np.array([370, 149])
        self.tar_top_right = np.array([376, 259])
        self.tar_bottom_right = np.array([265, 265])
        self.tar_bottom_left = np.array([259, 155])

        # ros2 communication variables
        self.cvBridge = CvBridge()
        self.img_group = MutuallyExclusiveCallbackGroup()
        self.img_publisher = self.create_publisher(Image, "/processed_image", 10, callback_group=self.img_group)
        self.aruco_center_publisher = self.create_publisher(Int64MultiArray, "/aruco_center", 10)
        self.corner_pub = self.create_publisher(ArucoCorner, "/aruco_corners", 10)

        # image reader timer
        self.img_reader_timer = self.create_timer(1/20, self.image_reader_timer, callback_group=self.img_group)     # 20 hz
        self.img_processer_timer = self.create_timer(1/20, self.image_pub_timer, callback_group=self.img_group)

    def process_image(self):
        gray_img = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)
        # Detect the markers
        self.corner, ids, rejected = self.detector.detectMarkers(gray_img)
        if (ids is not None):
            # considering only one marker found
            self.corner = self.corner[0][0]
            topLeft, topRight, bottomRight, bottomLeft = np.array(self.corner, dtype=np.int64)
            cv2.circle(self.latest_image, topLeft, radius=3, thickness=-1, color=(0, 255, 0))
            # cv2.putText(self.latest_image, '0', topLeft, 20, 1.0, (0, 0, 0), 1)
            cv2.circle(self.latest_image, topRight, radius=3, thickness=-1, color=(0, 255, 0))
            # cv2.putText(self.latest_image, '1', topRight, 20, 1.0, (0, 0, 0), 1)
            cv2.circle(self.latest_image, bottomRight, radius=3, thickness=-1, color=(0, 255, 0))
            # cv2.putText(self.latest_image, '2', bottomRight, 20, 1.0, (0, 0, 0), 1)
            cv2.circle(self.latest_image, bottomLeft, radius=3, thickness=-1, color=(0, 255, 0))
            # cv2.putText(self.latest_image, '3', bottomLeft, 20, 1.0, (0, 0, 0), 1)

            # mark the center of aruco marker
            p1 = topLeft
            p3 = bottomRight
            cx = p1[0] + (p3[0] - p1[0]) // 2
            cy = p1[1] + (p3[1] - p1[1]) // 2
            self.aruco_center = np.array([int(cx), int(cy)])
            cv2.circle(self.latest_image, self.aruco_center, radius=5, thickness=-1, color=(255, 0, 0))
        else:
            self.aruco_center = np.array([-1, -1])
            self.corner = None

    def plotTargetMarkers(self):
        # mark center of the frame
        # self.latest_image = cv2.circle(self.latest_image, self.frame_center, radius=10, thickness=-1, color=(0, 0, 255))
        
        # mark the target aruco point in the image frame
        cv2.circle(self.latest_image, self.tar_top_left, radius=3, thickness=-1, color=(0, 0, 255))
        cv2.circle(self.latest_image, self.tar_top_right, radius=3, thickness=-1, color=(0, 0, 255))
        cv2.circle(self.latest_image, self.tar_bottom_right, radius=3, thickness=-1, color=(0, 0, 255))
        cv2.circle(self.latest_image, self.tar_bottom_left, radius=3, thickness=-1, color=(0, 0, 255))
        # center
        p1 = self.tar_top_left
        p3 = self.tar_bottom_right
        cx = p1[0] + (p3[0] - p1[0]) // 2
        cy = p1[1] + (p3[1] - p1[1]) // 2
        cv2.circle(self.latest_image, [cx, cy], radius=4, thickness=-1, color=(0, 255, 255))
        # boundaries
        # cv2.line(self.latest_image, self.tar_top_left, self.tar_top_right, (230, 216, 173), 2)
        # cv2.line(self.latest_image, self.tar_top_right, self.tar_bottom_right, (230, 216, 173), 2)
        # cv2.line(self.latest_image, self.tar_bottom_right, self.tar_bottom_left, (230, 216, 173), 2)
        # cv2.line(self.latest_image, self.tar_bottom_left, self.tar_top_left, (230, 216, 173), 2)


    def image_reader_timer(self):
        # acquire latest image
        if (self.cam_obj.isOpened()):
            check, frame = self.cam_obj.read()
            if (check):
                self.latest_image = frame
            else:
                self.get_logger().warn(f"Image read failed!")
        else:
            self.get_logger().warn(f"Camera object is not opened!")

    def image_pub_timer(self):
        # processed image and publish it
        self.process_image()

        # draw target markers
        self.plotTargetMarkers()        

        # publish aruco center and aruco corners array
        if (self.aruco_center is not None and self.corner is not None):
            # aruco center
            array_msg = Int64MultiArray()
            array_msg.data = self.aruco_center.tolist()
            self.aruco_center_publisher.publish(array_msg)

            # aruco corner
            corner_msg = ArucoCorner()
            corner_msg.top_left = np.array(self.corner[0], np.int64)
            corner_msg.top_right = np.array(self.corner[1], np.int64)
            corner_msg.bottom_right = np.array(self.corner[2], np.int64)
            corner_msg.bottom_left = np.array(self.corner[3], np.int64)
            self.corner_pub.publish(corner_msg)

        # publish processed image
        msg = Image()
        msg= self.cvBridge.cv2_to_imgmsg(self.latest_image, encoding="bgr8")
        self.img_publisher.publish(msg)


def main():
    rclpy.init()

    try:
        node = ArucoNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Shutting down aruco node:\nException: {e}")
        node.destroy_node()
        # rclpy.shutdown()

if __name__ == "__main__":
    main()