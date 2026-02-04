#!/home/logesh/cv_ws/cv_env/bin/python3

"""
This is a ros2 node that read current image from the sensor and processes the image.
The script uses SuperGlue pkg with pretrained weights to detect features in the query image.
This packages uses PyTorch + GPU for feature detection and matching.
"""

import cv2
import torch
import numpy as np
import pyrealsense2 as rs
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster
from visual_servoing_pkg.msg import MatchedPoints
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from models.matching import Matching
from models.utils import make_matching_plot_fast, frame2tensor

torch.set_grad_enabled(False)

class ArucoNode(Node):
    def __init__(self):
        super().__init__("aruco_node")

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
        # realsense setup
        self.configureRS()

        # logging
        self.get_logger().info(f"Camera device: {self.realsenseDev}")
        self.get_logger().info(f"Image frame width: {self.frame_width}")
        self.get_logger().info(f"Image frame height: {self.frame_height}")

        # superglue model setup
        self.device = None
        self.refImg = None
        self.depthRefImg = None         # depth of each feature points in the refImage
        self.depthCurImg = None         # depth of each feature points in the curImage
        self.stackImage = None
        self.refData = None
        self.matchingModel = None
        self.match0 = None
        self.match1 = None
        self.setupSuperGlue()
        
        # log
        self.get_logger().info(f"Found device (for inference): {self.device}")

        # ros2 communication variables
        self.cvBridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.img_group = MutuallyExclusiveCallbackGroup()
        self.img_publisher = self.create_publisher(Image, "/processed_image", 10, callback_group=self.img_group)
        self.depthImg_publisher = self.create_publisher(Image, "/depth_image", 10, callback_group=self.img_group)
        self.matchPoints_publisher = self.create_publisher(MatchedPoints, "/matched_points", 10, callback_group=self.img_group)

        # image reader timer
        self.img_reader_timer = self.create_timer(1/20, self.image_reader_timer, callback_group=self.img_group)     # 20 hz
        self.img_processer_timer = self.create_timer(1/20, self.image_pub_timer, callback_group=self.img_group)


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

    def setupSuperGlue(self):
        """
        Function to setup superglue feature detection model.
        Function also extracts the reference image features and stores it.
        """

        # model setup
        self.device = "cuda" if (torch.cuda.is_available()) else "cpu"
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        self.matchingModel = Matching(config).eval().to(self.device)
        keys = ['keypoints', 'scores', 'descriptors']

        # load reference image
        self.refImg = cv2.imread("/home/logesh/fanuc_ws/src/visual-servoing-pkg/doc/images/ref_img_socket.png", cv2.IMREAD_GRAYSCALE)
        self.refImg = cv2.resize(self.refImg, (640, 480))
        refTensor = frame2tensor(self.refImg, self.device)
        # compute feature extraction - reference image
        self.refData = self.matchingModel.superpoint({'image': refTensor})
        self.refData = {k+'0': self.refData[k] for k in keys}
        self.refData['image0'] = refTensor

    def processImage(self):
        """Function to process image and find feature points"""
        
        if (self.color_frame is None):
            self.match0 = None
            self.match1 = None
            return
        
        try:
            cur_grayImg = cv2.cvtColor(self.color_frame, cv2.COLOR_BGR2GRAY)
            # tensor frame
            cur_tensorImg = frame2tensor(cur_grayImg, self.device)
            # feature prediction and matching 
            pred = self.matchingModel({**self.refData, 'image1': cur_tensorImg})
            kpts0 = self.refData['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()
            # compute valid pairs
            valid = matches > -1
            self.match0 = kpts0[valid]
            self.match1 = kpts1[matches[valid]]
            color = cm.jet(confidence[valid])
            
            # stacked image
            """
            # img1 = cv2.circle(self.refImg, list(map(int, mkpts0[0])), 10, (255, 0, 0), -1)
            # img2 = cv2.circle(cur_grayImg, list(map(int, mkpts1[0])), 10, (255, 0, 0), -1)
            self.stackImage = make_matching_plot_fast(
                self.refImg, cur_grayImg, kpts0, kpts1, self.match0, self.match1, color, "",
                path=None, show_keypoints=True)
            # self.stackImage = np.vstack([img1, img2])
            """

            # plot matched points in current frame (used SuperGlue utils)
            color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
            for (x1, y1), c in zip(self.match1, color):
                c = c.tolist()
                # display line end-points as circles
                cv2.circle(self.color_frame, (int(x1), int(y1)), 2, c, -1, lineType=cv2.LINE_AA)
        except Exception as e:
            self.get_logger().warn(f"Exception occured: {e}")
            self.match0 = None
            self.match1 = None

    def image_reader_timer(self):
        try:
            # acquire latest image
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            self.color_frame = np.asanyarray(aligned_frames.get_color_frame().get_data())
            self.depth_frame = np.asanyarray(aligned_frames.get_depth_frame().get_data())

            # collect refernce image
            if (self.collect_refImg and (self.color_frame is not None and self.depth_frame is not None)):
                self.collect_refImg = False
                cv2.imwrite("../doc/images/ref_img_socket.png", self.color_frame)
                np.savez("../doc/images/ref_img_socket_depth.npz", depthArr=np.float64(self.depth_frame))
        except Exception as e:
            self.color_frame = None
            self.depth_frame = None
            self.get_logger().error(f"Frame extraction issue! {e}")

    def image_pub_timer(self):
        """Timer function to call feature detection and publish the same"""

        # process the image (feature detection and processing)
        self.processImage()
        match_len = 0 if (self.match0 is None) else len(self.match0)
        self.get_logger().info(f"Number of matches found: {match_len}")

        # publish matched poin (if available)
        if (self.match0 is not None and self.match1 is not None):
            msg1 = MatchedPoints()
            msg1.rows, msg1.cols = self.match0.shape
            msg1.match_0 = np.int64(self.match0).flatten().tolist()
            msg1.match_1 = np.int64(self.match1).flatten().tolist()
            self.matchPoints_publisher.publish(msg1)
        else:
            msg1 = MatchedPoints()
            msg1.rows = -1
            msg1.cols = -1
            self.matchPoints_publisher.publish(msg1)

        # publish processed image
        msg = self.cvBridge.cv2_to_imgmsg(self.color_frame, encoding="bgr8")
        self.img_publisher.publish(msg)
        # publish depth image
        msg = self.cvBridge.cv2_to_imgmsg(self.depth_frame)
        self.depthImg_publisher.publish(msg)


def main():
    rclpy.init()

    try:
        node = ArucoNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Shutting down aruco node:\nException: {e}")
        node.destroy_node()
        node.pipeline.stop()
        # rclpy.shutdown()

if __name__ == "__main__":
    main()