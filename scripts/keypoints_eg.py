#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class KeypointsDetection(Node):
    def __init__(self):
        super().__init__("keypoints_detection_node")

        # camera object
        self.camDevice = 0
        self.frameWidth = 640
        self.frameHeight = 480
        self.cam = cv2.VideoCapture(self.camDevice)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.frameWidth)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frameHeight)

        # feature detector
        self.orb_detector = cv2.ORB_create()
        # create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # ref image (key computation)
        self.ref_image = cv2.imread("/home/logesh/fanuc_ws/src/visual-servoing-pkg/doc/images/ref_img.png", cv2.IMREAD_GRAYSCALE)
        self.kp_ref, self.des_ref = self.orb_detector.detectAndCompute(self.ref_image, None)
        
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

            # process image
            kp_cur, des_cur = self.orb_detector.detectAndCompute(gray_image, None)
            # match points
            matches = self.bf.match(self.des_ref, des_cur)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
            # Draw first 10 matches.
            processed_image = cv2.drawMatches(self.ref_image, self.kp_ref, gray_image, kp_cur, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # processed_image = cv2.drawKeypoints(gray_image, kp, None, color=(0,255,0), flags=0)

            # pub
            self.img_pub.publish(self.cv_bridge.cv2_to_imgmsg(processed_image))

    # def __del__(self):
    #     self.cam.release()


def main():
    rclpy.init()
    node = KeypointsDetection()
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"Shutting down node: {e}")
        node.destroy_node()

if __name__ == "__main__":
    main()