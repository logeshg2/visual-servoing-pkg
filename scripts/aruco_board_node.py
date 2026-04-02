#!/usr/bin/env python3

"""
This is a ros2 node that read current image from the sensor and processes the image.
Aruco board is detected in this script.
"""

import cv2
import math
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import Pose, TransformStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class ArucoNode(Node):
    def __init__(self):
        super().__init__("aruco_node")

        self.declare_parameter('camera_id', 0)
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

        # camera intrinsic's
        K_fp = open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/camera_matrix.pkl", "rb")
        dist_fp = open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/dist_coef.pkl", "rb")
        self.K = pickle.load(K_fp)
        self.camDist = pickle.load(dist_fp)

        # logging
        self.get_logger().info(f"Camera device id: {self.cam_dev_id}")
        self.get_logger().info(f"Image frame width: {self.frame_width}")
        self.get_logger().info(f"Image frame height: {self.frame_height}")

        # aruco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        parameters = cv2.aruco.DetectorParameters()
        # Create grid board object we're using in our stream
        self.board = cv2.aruco.GridBoard(
                    size=(3, 4),
                    markerLength=0.052,
                    markerSeparation=0.005,
                    dictionary=aruco_dict)
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        self.tvecs = None
        self.rvecs = None

        # ros2 communication variables
        self.cvBridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.img_group = MutuallyExclusiveCallbackGroup()
        self.img_publisher = self.create_publisher(Image, "/processed_image", 10, callback_group=self.img_group)
        self.aruco_pose_pub = self.create_publisher(Pose, "/aruco_pose", 10)

        # image reader timer
        self.img_reader_timer = self.create_timer(1/20, self.image_reader_timer, callback_group=self.img_group)     # 20 hz
        self.img_processer_timer = self.create_timer(1/20, self.image_pub_timer, callback_group=self.img_group)

    def process_image(self):
        gray_img = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)
        
        # Detect the markers
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray_img)
        
        # Refine detected markers
        # Eliminates markers not part of our board, adds missing markers to the board
        corners, ids, rejectedImgPoints, recoveredIds = cv2.aruco.refineDetectedMarkers(
            image=gray_img,
            board=self.board,
            detectedCorners=corners,
            detectedIds=ids,
            rejectedCorners=rejectedImgPoints,
            cameraMatrix=self.K,
            distCoeffs=self.camDist
        )

        # Outline all of the markers detected in our image
        # self.latest_image = cv2.aruco.drawDetectedMarkers(self.latest_image, corners, ids, borderColor=(0, 255, 0))

        # Require 1 markers before drawing axis
        if ids is not None and len(ids) > 0:
            # Estimate the posture of the gridboard, which is a construction of 3D space based on the 2D video
            pose, self.rvecs, self.tvecs = cv2.aruco.estimatePoseBoard(corners, ids, self.board, self.K, self.camDist, self.rvecs, self.tvecs)

            if pose:
                # Draw the camera posture calculated from the gridboard
                self.latest_image = cv2.drawFrameAxes(self.latest_image, self.K, self.camDist, self.rvecs, self.tvecs, 0.1)
        else:
            self.rvecs = None
            self.tvecs = None

    def TF_publisher(self, trans, quat):
        # broadcast cTo tf
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'camera_link'
        tf.child_frame_id = "aruco_target"
        
        tf.transform.translation.x = trans[0]
        tf.transform.translation.y = trans[1]
        tf.transform.translation.z = trans[2] - 0.3        # NOTE: publish target at a certain height

        tf.transform.rotation.x = quat[0]
        tf.transform.rotation.y = quat[1]
        tf.transform.rotation.z = quat[2]
        tf.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(tf)

    def plotTargetMarkers(self):
        # mark center of the frame
        # cv2.circle(self.latest_image, self.frame_center, radius=5, thickness=-1, color=(255, 0, 0))
        cv2.line(self.latest_image, (self.frame_center[0]-10, self.frame_center[1]-10), (self.frame_center[0]+10, self.frame_center[1]+10), (255,0,0), 2)
        cv2.line(self.latest_image, (self.frame_center[0]-10, self.frame_center[1]+10), (self.frame_center[0]+10, self.frame_center[1]-10), (255,0,0), 2)
        # mark xy image plane
        cv2.arrowedLine(self.latest_image, self.frame_center, [self.frame_center[0] + 100, self.frame_center[1]], (0,0,255), 2)     # X
        cv2.arrowedLine(self.latest_image, self.frame_center, [self.frame_center[0], self.frame_center[1] + 100], (0,255,0), 2)     # Y

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

    def eul2Quat(self, roll, pitch, yaw):   # roll, pitch and yaw should be in radians
        quat = np.array([0.0, 0.0, 0.0, 1.0])

        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        quat[0] = sr * cp * cy - cr * sp * sy
        quat[1] = cr * sp * cy + sr * cp * sy
        quat[2] = cr * cp * sy - sr * sp * cy
        quat[3] = cr * cp * cy + sr * sp * sy
        
        return quat

    def image_pub_timer(self):
        # processed image and publish it
        self.process_image()

        # draw target markers
        self.plotTargetMarkers()        

        # publish aruco pose
        if (self.tvecs is not None):
            self.tvecs = np.array(self.tvecs).flatten()
            self.rvecs = np.array(self.rvecs).flatten()
            quat = self.eul2Quat(self.rvecs[0], self.rvecs[1], self.rvecs[2])

            msg = Pose()
            msg.position.x = self.tvecs[0]
            msg.position.y = self.tvecs[1]
            msg.position.z = self.tvecs[2]
            msg.orientation.x = quat[0]
            msg.orientation.y = quat[1]
            msg.orientation.z = quat[2]
            msg.orientation.w = quat[3]

            # publish
            self.aruco_pose_pub.publish(msg)

            # publish tf
            self.TF_publisher(self.tvecs, quat)
        else:
            msg = Pose()
            msg.position.x = -1.0
            self.aruco_pose_pub.publish(msg)


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