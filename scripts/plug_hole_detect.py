#!/home/logesh/cv_ws/cv_env/bin/python3

"""
This is a ros2 node that read current image from the sensor and processes the image.
The script uses YOLO from Ultralytics pkg with fine-tuned weights to detect holes in the query image.
"""

import cv2
import math
import pickle
import numpy as np
import pyrealsense2 as rs
import matplotlib.cm as cm
from ultralytics import YOLO
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Int64MultiArray
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class HoleDetector(Node):
    def __init__(self):
        super().__init__("hole_detector_node")

        self.color_frame = None
        self.depth_frame = None
        self.collect_refImg = False
        self.frame_width = 640
        self.frame_height = 480
        self.frame_center = np.array([int(self.frame_width // 2), int(self.frame_height // 2)])

        # realsense sdk setup
        self.pipeline = None
        self.realsenseDev = None
        self.align = None
        self.K = None
        self.camDist = None
        # realsense intrinsic's
        K_fp = open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/camera_matrix_rs.pkl", "rb")
        dist_fp = open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/dist_coef_rs.pkl", "rb")
        self.K = pickle.load(K_fp)
        self.camDist = np.float64(pickle.load(dist_fp))
        # realsense setup
        # self.configureRS()

        # logging
        # self.get_logger().info(f"Camera device: {self.realsenseDev}")
        self.get_logger().info(f"Image frame width: {self.frame_width}")
        self.get_logger().info(f"Image frame height: {self.frame_height}")

        # yolo hole detector model setup
        self.holes = None
        self.model = YOLO("/home/logesh/fanuc_ws/src/ObjectPose-simple/weights/hole_best.pt")
        self.desiredHoles = np.array([
            [337, 221],
            [319, 245],
            [356, 244],
            [318, 266],
            [359, 265]
        ])

        # ros2 communication variables
        self.cvBridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.img_group = MutuallyExclusiveCallbackGroup()
        self.img_publisher = self.create_publisher(Image, "/processed_image_1", 10, callback_group=self.img_group)
        self.depthImg_publisher = self.create_publisher(Image, "/depth_image", 10, callback_group=self.img_group)
        self.matchPoints_publisher = self.create_publisher(Int64MultiArray, "/holes_coord", 10, callback_group=self.img_group)
        self.rs_color_sub = self.create_subscription(Image, "/camera/camera/color/image_raw", self.rs_color_cb, 10, callback_group=self.img_group)
        self.rs_depth_sub = self.create_subscription(Image, "/camera/camera/depth/image_rect_raw", self.rs_depth_cb, 10, callback_group=self.img_group)

        # image reader timer
        # self.img_reader_timer = self.create_timer(1/20, self.image_reader_timer, callback_group=self.img_group)     # 20 hz
        self.img_processer_timer = self.create_timer(1/20, self.image_pub_timer, callback_group=self.img_group)

    def rs_color_cb(self, msg):
        try:
            # acquire color image
            if (msg is not None):
                self.color_frame = self.cvBridge.imgmsg_to_cv2(msg, 'bgr8')
            else:
                self.color_frame = None
        except Exception as e:
            self.get_logger().warn(f"RS color image aquisition exception: {e}")
            self.color_frame = None

    def rs_depth_cb(self, msg):
        try:
            # acquire color image
            if (msg is not None):
                self.depth_frame = self.cvBridge.imgmsg_to_cv2(msg)
            else:
                self.depth_frame = None
        except Exception as e:
            self.get_logger().warn(f"RS depth image aquisition exception: {e}")
            self.depth_frame = None

    def configureRS(self):
        """Function to configure  realsense device to perform image acquisition"""
         # realsense camera setting (color and depth frame aligned)
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, self.frame_width, self.frame_height, rs.format.z16, 30)
        rs_config.enable_stream(rs.stream.color, self.frame_width, self.frame_height, rs.format.bgr8, 30)
        profile = self.pipeline.start(rs_config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        # alignment object
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        # realsense product details
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = rs_config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.realsenseDev = str(device.get_info(rs.camera_info.product_line))
        # filtering and hole filling parameter
        self.hole_filling = rs.hole_filling_filter()
        # get intrinsic's of camera frame
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_intrinsics = depth_profile.get_intrinsics()
        color_intrinsics = color_profile.get_intrinsics()
        # put into matrix
        depth_camera_matrix = np.array([
            [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
            [0, depth_intrinsics.fy, depth_intrinsics.ppy],
            [0, 0, 1]
        ])
        self.K = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ])
        self.camDist = color_intrinsics.coeffs

    def solveCorrespondence(self, xywh_arr):
        """Function to compute and solve correspondence problem - match points between two frames."""

        # TODO: optimized approach for 2d points correspondence
        
        # find close points p2-p4 and p3-p5 
        closest_idx = []
        min_dist = 10000
        for idx, xywh in enumerate(xywh_arr):
            xy1 = xywh[0:2]
            min_dist = 10000
            temp = [idx, -1]
            for idx_1, xywh_1 in enumerate(xywh_arr):
                if (idx == idx_1):
                    continue
                xy2 = xywh_1[0:2]
                dist = math.dist(xy1, xy2)
                if (dist < min_dist):
                    min_dist = dist
                    temp[1] = idx_1
            closest_idx.append(temp)
        
        """
        # print(closest_idx)
        # draw line
        for idx_arr in closest_idx:
            pt1 = list(map(int, xywh_arr[idx_arr[0]][0:2]))
            pt2 = list(map(int, xywh_arr[idx_arr[1]][0:2]))
            # print(pt1)
            # print(pt2)
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        """

        # find P1
        P1_idx = -1
        set_temp = set(np.array(closest_idx).flatten().tolist())
        for point in set_temp:
            if np.array(closest_idx).flatten().tolist().count(point) == 1:
                P1_idx = point
                break

        # find P2 and P3 ( two least distant from P1 )
        P2_idx = -1
        min_dist = 10000
        # P2
        for idx, xywh in enumerate(xywh_arr):
            if (idx == P1_idx): # no need to compare for P1
                continue
            dist = math.dist(xywh_arr[P1_idx][0:2], xywh_arr[idx][0:2])
            if (dist < min_dist):
                min_dist = dist
                P2_idx = idx
        # P3
        P3_idx = -1
        min_dist = 10000
        for idx, xywh in enumerate(xywh_arr):
            if (idx == P1_idx or idx == P2_idx): # no need to compare for P1
                continue
            dist = math.dist(xywh_arr[P1_idx][0:2], xywh_arr[idx][0:2])
            if (dist < min_dist):
                min_dist = dist
                P3_idx = idx

        # P2 and P3 correspondence (using angle)
        # P2 and P3 mid point
        mid_x = (xywh_arr[P2_idx][0] + xywh_arr[P3_idx][0]) / 2
        mid_y = (xywh_arr[P2_idx][1] + xywh_arr[P3_idx][1]) / 2
        mid_P2P3 = np.array([mid_x, mid_y])

        # compute line vector 
        # (v - vector of P1-mid_P2P3)
        v = mid_P2P3 - xywh_arr[P1_idx][0:2]
        # w - vector of P1-P2
        w = xywh_arr[P2_idx][0:2] - xywh_arr[P1_idx][0:2]
        # compute angle between vector (v & w) -> (line_P1_Mid and line_P1_P2)
        theta_rad = np.arctan2((v[0]*w[1]) - (v[1]*w[0]), (v[0]*w[0]) + (v[1]*w[1]))
        theta_deg = np.degrees(theta_rad)
        # NOTE: if theta is negative - the point is P3 (so swap P2 and P3 idices)
        if (theta_deg < 0):
            P2_idx, P3_idx = P3_idx, P2_idx

        # find P4 and P5 ( least distant from P2 and P3 )
        P4_idx = -1
        min_dist = 10000
        # P4
        for idx, xywh in enumerate(xywh_arr):
            if ((idx == P1_idx or idx == P2_idx) or (idx == P3_idx)): # no need to compare for P1, P2, P3
                continue
            dist = math.dist(xywh_arr[P2_idx][0:2], xywh_arr[idx][0:2]) # P4 is closer to P2
            if (dist < min_dist):
                min_dist = dist
                P4_idx = idx
        # P5
        P5_idx = -1
        min_dist = 10000
        for idx, xywh in enumerate(xywh_arr):
            if ((idx == P1_idx or idx == P2_idx) or (idx == P3_idx or idx == P4_idx)): # no need to compare for P1, P2, P3, P4
                continue
            dist = math.dist(xywh_arr[P3_idx][0:2], xywh_arr[idx][0:2]) # P5 is closer to P3
            if (dist < min_dist):
                min_dist = dist
                P5_idx = idx

        # matched correspondance
        self.holes = np.int64([
            xywh_arr[P1_idx][0:2],
            xywh_arr[P2_idx][0:2],
            xywh_arr[P3_idx][0:2],
            xywh_arr[P4_idx][0:2],
            xywh_arr[P5_idx][0:2]
        ])

    def processImage(self):
        """Function to process image and find feature points"""
        
        if (self.color_frame is None):
            self.holes = None
            return
        
        try:
            # detect the holes
            result = self.model.predict(self.color_frame, stream=False, save=False)[0]
            xywh_arr = result.boxes.xywh.cpu().numpy()
            
            # plot the holes
            for idx, xywh in enumerate(xywh_arr):
                cv2.circle(self.color_frame, (int(xywh[0]), int(xywh[1])), 2, (0, 255, 0), -1)
            
            # plot desired holes coord
            for idx, xywh in enumerate(self.desiredHoles):
                cv2.circle(self.color_frame, (int(xywh[0]), int(xywh[1])), 2, (0, 0, 255), -1)

            if (len(xywh_arr) == 5):
                # solve correspondence problem
                self.solveCorrespondence(xywh_arr)
            else:
                # no enough point to compute
                self.holes = None
                pass

        except Exception as e:
            self.get_logger().warn(f"Exception occured: {e}")
            self.holes = None

    def image_reader_timer(self):
        try:
            # acquire latest image
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            self.color_frame = np.asanyarray(aligned_frames.get_color_frame().get_data())
            # self.depth_frame = np.asanyarray(aligned_frames.get_depth_frame().get_data())

            # apply filter to fill the holes in depth image
            frame = aligned_frames.get_depth_frame()
            self.depth_frame = np.asanyarray(self.hole_filling.process(frame).get_data())

        except Exception as e:
            self.color_frame = None
            self.depth_frame = None
            self.get_logger().error(f"Frame extraction issue! {e}")

    def image_pub_timer(self):
        """Timer function to call feature detection and publish the same"""

        # process the image (feature detection and processing)
        self.processImage()

        # publish matched poin (if available)
        if (self.holes is not None):
            msg1 = Int64MultiArray()
            msg1.data = self.holes.flatten().tolist()
            self.matchPoints_publisher.publish(msg1)
        else:
            msg1 = Int64MultiArray()
            msg1.data = [-1]
            self.matchPoints_publisher.publish(msg1)

        if (self.color_frame is None):
            return

        # publish processed image
        msg = self.cvBridge.cv2_to_imgmsg(self.color_frame, encoding="bgr8")
        self.img_publisher.publish(msg)
        # publish depth image
        msg = self.cvBridge.cv2_to_imgmsg(self.depth_frame)
        self.depthImg_publisher.publish(msg)


def main():
    rclpy.init()

    # try:
    node = HoleDetector()
    rclpy.spin(node)
    # except Exception as e:
        # print(f"Shutting down node:\nException: {e}")
        # node.destroy_node()
        # node.pipeline.stop()
        # rclpy.shutdown()

if __name__ == "__main__":
    main()