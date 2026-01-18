#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class BlackCircleDetection(Node):
    def __init__(self):
        super().__init__("black_circle_detection_node")

        # camera object
        self.camDevice = 0
        self.frameWidth = 1080
        self.frameHeight = 720
        self.cam = cv2.VideoCapture(self.camDevice)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.frameWidth)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frameHeight)

        #
        
        if (not self.cam.isOpened()):
            self.get_logger().error(f"Not able open camera!")
            self.destroy_node()
        # log 
        self.get_logger().info(f"Camera is open: camera device id: {self.camDevice}")
        self.get_logger().info(f"Camera frame width x height: {self.frameWidth} x {self.frameHeight}")

        # ros2 variables
        self.current_frame = None
        self.cv_bridge = CvBridge()
        self.img_group = MutuallyExclusiveCallbackGroup()

        self.img_pub = self.create_publisher(Image, '/processed_image', 10)
        self.img_timer = self.create_timer(1/20, self.image_callback, self.img_group)
        self.process_timer = self.create_timer(1/20, self.image_processor_cb, self.img_group)
    
    def image_callback(self):
        # read image from camera object
        check, frame = self.cam.read()
        if (check):
            # received image frame
            self.current_frame = frame
        else:
            self.current_frame = None

    def image_processor_cb(self):
        if (self.current_frame is not None):
            gray_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

            # threshold
            _, thresh = cv2.threshold(gray_image, 130, 250, cv2.THRESH_BINARY_INV)
            # find contours
            # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # for cnt in contours:
            #     area = cv2.contourArea(cnt)
            #     perimeter = cv2.arcLength(cnt, True)
            #     if perimeter == 0:
            #         continue

            #     circularity = 4 * np.pi * area / (perimeter * perimeter)

            #     if circularity > 0.8 and area > 100:
            #         (x, y), radius = cv2.minEnclosingCircle(cnt)
            #         cv2.circle(self.current_frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

            # pub
            self.img_pub.publish(self.cv_bridge.cv2_to_imgmsg(thresh))

    # def __del__(self):
    #     self.cam.release()


def main():
    rclpy.init()
    node = BlackCircleDetection()
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"Shutting down node: {e}")
        node.destroy_node()

if __name__ == "__main__":
    main()