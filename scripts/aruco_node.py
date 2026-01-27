#!/usr/bin/env python3

"""
This is a ros2 node that read current image from the sensor and processes the image.
Aruco is used in this script.
"""

import cv2
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Int64MultiArray
from visual_servoing_pkg.msg import ArucoCorner
from geometry_msgs.msg import Pose, TransformStamped
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
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.markerLength = 0.1
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        self.aruco_center = None
        self.corner = None
        self.aruco_pose = None

        # target aruco points
        self.tar_top_left = np.array([370, 149])
        self.tar_top_right = np.array([376, 259])
        self.tar_bottom_right = np.array([265, 265])
        self.tar_bottom_left = np.array([259, 155])

        # ros2 communication variables
        self.cvBridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.img_group = MutuallyExclusiveCallbackGroup()
        self.img_publisher = self.create_publisher(Image, "/processed_image", 10, callback_group=self.img_group)
        self.aruco_center_publisher = self.create_publisher(Int64MultiArray, "/aruco_center", 10)
        self.corner_pub = self.create_publisher(ArucoCorner, "/aruco_corners", 10)
        self.aruco_pose_pub = self.create_publisher(Pose, "/aruco_pose", 10)

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

            # aruco pose
            object_points = np.array([
                [-self.markerLength / 2, -self.markerLength / 2, 0],
                [self.markerLength / 2, -self.markerLength / 2, 0],
                [self.markerLength / 2, self.markerLength / 2, 0],
                [-self.markerLength / 2, self.markerLength / 2, 0]
            ], dtype=np.float32)
            _, rvec, tvec = cv2.solvePnP(object_points, np.array(self.corner), self.K, self.camDist, False, flags=cv2.SOLVEPNP_EPNP )
            if (_):
                cv2.drawFrameAxes(self.latest_image, self.K, self.camDist, rvec, tvec, 0.1, 3)
                rvec = rvec.flatten()
                tvec = tvec.flatten()
                quat = Rotation.from_rotvec(rvec).as_quat()
                self.aruco_pose = {'tvec': tvec, 'quat': quat}
            else:
                self.aruco_pose = None
        else:
            self.aruco_center = np.array([-1, -1])
            self.corner = None
            self.aruco_pose = None

    def plotTargetMarkers(self):
        # mark center of the frame
        # cv2.circle(self.latest_image, self.frame_center, radius=5, thickness=-1, color=(255, 0, 0))
        cv2.line(self.latest_image, (self.frame_center[0]-10, self.frame_center[1]-10), (self.frame_center[0]+10, self.frame_center[1]+10), (255,0,0), 2)
        cv2.line(self.latest_image, (self.frame_center[0]-10, self.frame_center[1]+10), (self.frame_center[0]+10, self.frame_center[1]-10), (255,0,0), 2)
        # mark xy image plane
        cv2.arrowedLine(self.latest_image, self.frame_center, [self.frame_center[0] + 100, self.frame_center[1]], (0,0,255), 2)     # X
        cv2.arrowedLine(self.latest_image, self.frame_center, [self.frame_center[0], self.frame_center[1] + 100], (0,255,0), 2)     # Y


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
        # self.plotTargetMarkers()        

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

            # aruco pose + aruco TF
            if (self.aruco_pose is not None):
                msg = Pose()
                msg.position.x = self.aruco_pose['tvec'][0]
                msg.position.y = self.aruco_pose['tvec'][1]
                msg.position.z = self.aruco_pose['tvec'][2]
                msg.orientation.x = self.aruco_pose['quat'][0]
                msg.orientation.y = self.aruco_pose['quat'][1]
                msg.orientation.z = self.aruco_pose['quat'][2]
                msg.orientation.w = self.aruco_pose['quat'][3]
                self.aruco_pose_pub.publish(msg)

                # aruco TF (with respect to camera_link)
                self.TF_publisher(self.aruco_pose['tvec'], self.aruco_pose['quat'])
        else:
            corner_msg = ArucoCorner()
            corner_msg.top_left = np.array([-1, -1])
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