#!/home/logesh/cv_ws/cv_env/bin/python3

"""
This is a ros2 node that read current image from the sensor and processes the image.
The script uses YOLO from Ultralytics pkg with fine-tuned weights to detect bolts in the query image.
Using PnP, pose of the ponnel is also identified.
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
from geometry_msgs.msg import Pose
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Int64MultiArray
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class labels:
    bolt = 0
    box = 1
    feature = 2

SAVE_MISSED = False

class PonnelDetector(Node):
    def __init__(self):
        super().__init__("ponnel_detector_node")

        self.color_frame = None
        self.depth_frame = None
        self.collect_refImg = False
        self.frame_width = 640
        self.frame_height = 480
        self.frame_center = np.array([int(self.frame_width // 2), int(self.frame_height // 2)])
        self.imgCount = 0

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

        # yolo screw + box model setup
        self.screws = None
        self.boxPose = None
        self.model = YOLO("/home/logesh/fanuc_ws/src/ObjectPose-simple/weights/ponnel_best.pt")
        self.desiredScrews = np.array([
            [202, 186],
            [441, 174],
            [440, 310],
            [200, 305],
        ])
        self.prev_rvec = None
        self.prev_tvec = None
        # ponnel - dim
        self.ponnel_object_points = np.array([
            [-(94/2)+5, -(94/2), 0.0],
            [(94/2), -(94/2), 0.0],
            [(94/2), (94/2), 0.0],
            [-(94/2), (94/2), 0.0]
        ], dtype=np.float32) / 1000.0

        # ros2 communication variables
        self.cvBridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.img_group = MutuallyExclusiveCallbackGroup()
        self.img_publisher = self.create_publisher(Image, "/processed_image_2", 10, callback_group=self.img_group)
        # self.depthImg_publisher = self.create_publisher(Image, "/depth_image", 10, callback_group=self.img_group)
        self.matchPoints_publisher = self.create_publisher(Int64MultiArray, "/bolt_coord", 10, callback_group=self.img_group)
        self.rs_color_sub = self.create_subscription(Image, "/camera/camera/color/image_raw", self.rs_color_cb, 10, callback_group=self.img_group)
        self.rs_depth_sub = self.create_subscription(Image, "/camera/camera/depth/image_rect_raw", self.rs_depth_cb, 10, callback_group=self.img_group)
        self.boxPose_pub = self.create_publisher(Pose, "/ponnel_pose", 10)

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

    def solveCorrespondence(self, classes, xyxy_arr, xywh_arr):
        """Function to compute and solve correspondence problem - match points between two frames."""

        boxArr = []
        screwArr = []
        tempScrewArr = []

        # classification
        for idx, cls in enumerate(classes):
            if (cls == labels.box):
                boxArr.append(xyxy_arr[idx])
            elif (cls == labels.feature):
                screwArr.append(xywh_arr[idx][0:2])

        if (len(boxArr) != 0 and len(screwArr) != 0):
            # take the first box from boxes
            bx, by, bx_, by_ = boxArr[0]
            # remove screw detection outside box area
            for idx, (x, y) in enumerate(screwArr):
                if ((x >= bx and x <= bx_) and (y >= by and y <= by_)):
                    tempScrewArr.append([x, y])

        screwArr = None

        # correspondance problem
        # NOTE: atleast 5 screw's has to be detected
        if (len(tempScrewArr) >= 5):
            screw_xy = tempScrewArr[0:5]

            pt1_idx = -1
            pt2_idx = -1
            pt3_idx = -1
            pt4_idx = -1

            # find point 1 (two bolts which are close to each other)
            # pt1 is the mid of these two bolts
            minDist = 1000
            pair_idx = [-1, -1]
            for idx_, s_xy_ in enumerate(screw_xy):
                for idx, s_xy in enumerate(screw_xy):
                    if idx == idx_:
                        continue
                    dist = math.dist(s_xy, s_xy_)
                    if (dist < minDist):
                        minDist = dist
                        pair_idx = [idx_, idx]
            pt1_idx = pair_idx
            pt1_corrd = [-1, -1]
            pt1_corrd[0] = ((screw_xy[pt1_idx[0]][0] + screw_xy[pt1_idx[1]][0]) // 2)
            pt1_corrd[1] = ((screw_xy[pt1_idx[0]][1] + screw_xy[pt1_idx[1]][1]) // 2)
            
            # find point 3 (far away from point 1)
            maxDist = -1
            for idx, s_xy in enumerate(screw_xy):
                if (idx in pt1_idx):
                    continue
                dist = math.dist(s_xy, pt1_corrd)
                if (dist > maxDist):
                    maxDist = dist
                    pt3_idx = idx
            
            # find point 2 and 4 (using angle between them)
            for idx, s_xy in enumerate(screw_xy):
                if (idx in pt1_idx or idx == pt3_idx):
                    continue
                if (pt2_idx == -1):
                    pt2_idx = idx
                else:
                    pt4_idx = idx

            # mid of point 2 and 4
            mid_2_4 = (np.array(screw_xy[pt2_idx]) + np.array(screw_xy[pt4_idx])) // 2
            # compute line vector 
            # v - vector of P1-mid_2_4
            v = np.array(mid_2_4) - np.array(pt1_corrd)
            # w - vector of P1-P2
            w = np.array(screw_xy[pt2_idx]) - np.array(pt1_corrd)
            # compute angle between vector (v & w) -> (line_P1_Mid and line_P1_P2)
            theta_rad = np.arctan2((v[0]*w[1]) - (v[1]*w[0]), (v[0]*w[0]) + (v[1]*w[1]))
            theta_deg = np.degrees(theta_rad)
            # NOTE: if theta is positive - the point is P4 (so swap P2 and P4 idices)
            if (theta_deg >= 0):
                pt2_idx, pt4_idx = pt4_idx, pt2_idx

            
            # repopulate screwArr
            screwArr = np.int64([pt1_corrd, screw_xy[pt2_idx], screw_xy[pt3_idx], screw_xy[pt4_idx]])

            for idx, xy in enumerate(screwArr):
                cv2.circle(self.color_frame, (int(xy[0]), int(xy[1])), 5, (0, 255, 0), -1)

        self.screws = screwArr

            
    def processImage(self):
        """Function to process image and find feature points"""
        
        if (self.color_frame is None):
            self.screws = None
            return
        
        try:
            # detect the holes
            result = self.model.predict(self.color_frame, stream=False, save=False, conf=0.3, imgsz=320)[0]
            classes = result.boxes.cls.cpu().numpy()
            xyxy_arr = result.boxes.xyxy.cpu().numpy()
            xywh_arr = result.boxes.xywh.cpu().numpy()
            
            if (SAVE_MISSED):
                tempImg = self.color_frame.copy()

            # plot desired holes coord
            for idx, xywh in enumerate(self.desiredScrews):
                cv2.circle(self.color_frame, (int(xywh[0]), int(xywh[1])), 2, (0, 0, 255), -1)

            if (len(classes) >= 6):
                # solve correspondence problem
                self.solveCorrespondence(classes, xyxy_arr, xywh_arr)

                # compute pose
                imagePoints = np.float64(self.screws)        # remove point 1

                # 1st iteration
                if (self.prev_rvec is None):
                    success, rvec, tvec = cv2.solvePnP(self.ponnel_object_points, imagePoints, self.K, self.camDist, flags=cv2.SOLVEPNP_IPPE)
                else:
                    success, rvec, tvec = cv2.solvePnP(self.ponnel_object_points, imagePoints, self.K, self.camDist, self.prev_rvec, self.prev_tvec, useExtrinsicGuess=True,flags=cv2.SOLVEPNP_IPPE)
                self.prev_rvec = rvec
                self.prev_tvec = tvec
                
                if tvec[2] < 0.05:   # less than 5 cm
                    print("Rejected unstable pose")
                    success = False

                if (success):
                    cv2.drawFrameAxes(self.color_frame, self.K, self.camDist, rvec, tvec, 0.05, 3)
                    rvec = rvec.flatten()
                    tvec = tvec.flatten()
                    quat = Rotation.from_rotvec(rvec).as_quat()
                    self.boxPose = {'tvec': tvec, 'quat': quat}
                else:
                    self.boxPose = None
            else:

                # save data (missed detection)
                if (SAVE_MISSED):
                    # only save every 10 image
                    if (self.imgCount % 10 == 0):
                        cv2.imwrite(f"/home/logesh/no_detect_data/img_{self.imgCount}.png", tempImg)
                    self.imgCount += 1

                # no enough point to compute
                self.screws = None
                self.boxPose = None
                pass

        except Exception as e:
            self.get_logger().warn(f"Exception occured: {e}")
            self.screws = None
            self.boxPose = None

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

        # plot reference camera frame
        # mark center of the frame
        cv2.line(self.color_frame, (self.frame_center[0]-10, self.frame_center[1]-10), (self.frame_center[0]+10, self.frame_center[1]+10), (255,0,0), 2)
        cv2.line(self.color_frame, (self.frame_center[0]-10, self.frame_center[1]+10), (self.frame_center[0]+10, self.frame_center[1]-10), (255,0,0), 2)
        # mark xy image plane
        cv2.arrowedLine(self.color_frame, self.frame_center, [self.frame_center[0] + 100, self.frame_center[1]], (0,0,255), 2)     # X
        cv2.arrowedLine(self.color_frame, self.frame_center, [self.frame_center[0], self.frame_center[1] + 100], (0,255,0), 2)     # Y

        # publish matched point (if available)
        if (self.screws is not None):
            msg1 = Int64MultiArray()
            msg1.data = self.screws.flatten().tolist()
            self.matchPoints_publisher.publish(msg1)

            # box pose
            if (self.boxPose is not None):
                msg = Pose()
                msg.position.x = self.boxPose['tvec'][0]
                msg.position.y = self.boxPose['tvec'][1]
                msg.position.z = self.boxPose['tvec'][2]
                msg.orientation.x = self.boxPose['quat'][0]
                msg.orientation.y = self.boxPose['quat'][1]
                msg.orientation.z = self.boxPose['quat'][2]
                msg.orientation.w = self.boxPose['quat'][3]
                self.boxPose_pub.publish(msg)
            else:
                msg = Pose()
                msg.position.x = -1.0
                self.boxPose_pub.publish(msg)
        else:
            msg1 = Int64MultiArray()
            msg1.data = [-1]
            self.matchPoints_publisher.publish(msg1)

            msg = Pose()
            msg.position.x = -1.0
            self.boxPose_pub.publish(msg)

        if (self.color_frame is None):
            return

        # publish processed image
        msg = self.cvBridge.cv2_to_imgmsg(self.color_frame, encoding="bgr8")
        self.img_publisher.publish(msg)
        # publish depth image
        # msg = self.cvBridge.cv2_to_imgmsg(self.depth_frame)
        # self.depthImg_publisher.publish(msg)


def main():
    rclpy.init()

    try:
        node = PonnelDetector()
        rclpy.spin(node)
    except Exception as e:
        print(f"Shutting down node:\nException: {e}")
        node.destroy_node()
        # node.pipeline.stop()
        # rclpy.shutdown()

if __name__ == "__main__":
    main()