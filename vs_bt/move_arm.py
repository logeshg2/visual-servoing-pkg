#!/usr/bin/env python3

"""
Behaviour Script to move the arm to a specific joint or cartesian pose.
Target cartesian or joint pose is read from py_trees blackboard.
The behaviour accepts the following controls:
    1. Camera Velocity
    2. End Effector Velocity
    3. Joint Velcoity
    4. Joint Position
    5. Cartesian Position
"""


"""
TODO:
1. Correct default for joint position and cartestion position to blackboard.
2. Damping based control for velocity controller types (proper use of robot jacobian)
3. May be need to apply filter to the velocities generated (NOTE)
4. Check correctness of integration (jnt velocity)
5. 
"""


import pickle
import py_trees
import pinocchio
import numpy as np
from scipy.spatial.transform import Rotation


class ControlType:
    camVelCtrl = 0
    eeVelCtrl = 1
    jntVelCtrl = 2
    jntPosCtrl = 3
    cartPosCtrl = 4


class MoveArm(py_trees.behaviour.Behaviour):
    def __init__(self, realRobot, control_mode: ControlType=ControlType.camVelCtrl):

        self.name = "move_arm"
        self.controlMode = control_mode
        super(MoveArm, self).__init__(self.name)

        # robot parameters
        self.bot = realRobot
        self.eeFrameId = None
        self.robotModel = None
        self.robotData = None
        self.tracking_pose = None
        self.urdfPath = "/home/logesh/fanuc_ws/src/fanuc_ros2_drivers/src/fanuc_description/urdf/lrmate200id4s.urdf"

        # robot control parameters
        self.dt = 1.0
        self.camVel = None
        self.eeVel = None
        self.jntVel = None
        self.tarJntPos = None
        self.tarCartPos = None

        # camera calibration parameters
        self.K = pickle.load(open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/camera_matrix_rs.pkl", "rb"))
        self.Kinv = np.linalg.inv(self.K)
        # camera extrinsic for realsense (eye in hand config)
        mat = pickle.load(open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/eye_in_hand_rs.pkl", "rb"))
        rot = Rotation.from_matrix(mat[0:3, 0:3]).as_matrix()
        self.eTc = np.eye(4)
        self.eTc[0:3, 3] = mat[0:3, 3]
        self.eTc[0:3, 0:3] = rot

        # adjoint transformation (camera frame velocity to end-effector frame velocity transform)
        self.ADeTc = np.zeros((6, 6))
        self.ADeTc[0:3, 0:3] = self.eTc[0:3, 0:3]
        self.ADeTc[3:6, 3:6] = self.eTc[0:3, 0:3]
        eTc_t = self.eTc[0:3, 3]
        etc_x = np.array([               # skew symmetric matrix of translation (eTc.t)
            [0, (-1 * eTc_t[2]), eTc_t[1]],
            [eTc_t[2], 0, (-1 * eTc_t[0])],
            [(-1 * eTc_t[1]), eTc_t[0], 0]
        ])
        self.ADeTc[3:6, 0:3] = etc_x @ self.eTc[0:3, 0:3]

        self.blackboard = py_trees.blackboard.Blackboard()
        self.logger = py_trees.logging.Logger()

    def setup(self):
        """Setup robot arm movement parameters"""
        
        # hardware robot initialization
        self.bot.air_gripper_control('open')
        self.tracking_pose = np.array([60.0, 240.0, 120.0, 179.65, 0.69, 67.63])

        # robot model
        self.robotModel = pinocchio.buildModelsFromUrdf(self.urdfPath)[0]
        self.robotData = pinocchio.createDatas(self.robotModel)[0]
        self.eeFrameId = self.robotModel.getFrameId("tool0")

        # initialize robot control parameters
        self.camVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.eeVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.jntVel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.tarJntPos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.tarCartPos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # set default values to blackboard
        self.blackboard.set("eTc", self.eTc)
        self.blackboard.set("controlMode", self.controlMode)
        self.blackboard.set("camVel", self.camVel)
        self.blackboard.set("eeVel", self.eeVel)
        self.blackboard.set("jntVel", self.jntVel)
        self.blackboard.set("tarJntPos", self.tarJntPos)
        self.blackboard.set("tarCartPos", self.tarCartPos)

    def initialise(self):
        """Initialize or Reset control parameters before movement"""

        self.controlMode = self.blackboard.get("controlMode")
        
        if (self.controlMode == ControlType.camVelCtrl):
            self.camVel = self.blackboard.get("camVel")
        elif (self.controlMode == ControlType.eeVelCtrl):
            self.eeVel = self.blackboard.get("eeVel")
        elif (self.controlMode == ControlType.jntVelCtrl):
            self.jntVel = self.blackboard.get("jntVel")
        elif (self.controlMode == ControlType.jntPosCtrl):
            self.tarJntPos = self.blackboard.get("tarJntPos")
        elif (self.controlMode == ControlType.cartPosCtrl):
            self.tarCartPos = self.blackboard.get("tarCartPos")

    def integrateVel(self, qpos, qvel):
        # update joint position by integrating velocity
        for i in range(6):
            qpos[i] += (self.dt * qvel[i])
        
        return qpos

    def update(self):
        """Perform robot motion based on controlMode"""
        
        try:
            # read the current joint position
            cur_joint_pose = self.bot.read_current_joint_position()
            # current joint position (deg to rad) + J23 coupling
            rad_arr = np.deg2rad(cur_joint_pose)
            # remove coupling - J[3]' = J[3] + J[2]
            rad_arr[2] = rad_arr[2] + rad_arr[1]

            # compute robot jacobian
            robotJacobian = pinocchio.computeFrameJacobian(self.robotModel, self.robotData, np.array(rad_arr), self.eeFrameId)

            # velocity mode's
            # camVel - eeVel - jntVel (at end jntVel is computed)
            if (self.controlMode == ControlType.camVelCtrl):
                # camera velocity to ee velocity (using adjoint transformation - Ad_eTc)
                self.camVel = self.camVel.reshape((6, 1))
                self.eeVel = self.ADeTc @ self.camVel           # (6x1)
                self.logger.info(f"{np.round(self.eeVel, 4)}")
                
                # ee velocity to joint velocity (using robot jacobian)
                self.jntVel = (np.linalg.pinv(robotJacobian) @ self.eeVel)          # (6x1)

            elif (self.controlMode == ControlType.eeVelCtrl):
                # ee velocity to joint velocity (using robot jacobian)
                self.jntVel = (np.linalg.pinv(robotJacobian) @ self.eeVel)          # (6x1)
            
            elif (self.controlMode == ControlType.jntVelCtrl):
                # do nothing - already in jnt velocity
                pass

            # perform motion (or trigger movement)
            if ((self.controlMode == ControlType.camVelCtrl or self.controlMode == ControlType.eeVelCtrl) or self.controlMode == ControlType.jntVelCtrl):
                # integrate velocity to position
                # compute target joint angle from target velocity
                self.jntVel = self.jntVel.flatten()                                 # (6, )
                target_rad_arr = self.integrateVel(qpos=rad_arr, qvel=self.jntVel)

                # adding coupling - J[3]' = J[3] - J[2]
                target_rad_arr[2] = target_rad_arr[2] - target_rad_arr[1]
                target_joint_pose = np.rad2deg(target_rad_arr).tolist()

                # write register and sync-movement
                self.bot.write_joint_pose(target_joint_pose, blocking=False)
            
            elif (self.controlMode == ControlType.jntPosCtrl):
                # joint position control
                self.bot.write_joint_pose(joint_position_array=self.tarJntPos, blocking=False)
            
            elif (self.controlMode == ControlType.cartPosCtrl):
                # cartesian position control
                self.bot.write_cartesian_position(coords=self.tarCartPos, blocking=False)
            
            return py_trees.common.Status.SUCCESS
        
        except Exception as e:
            self.logger.error(f"Exception with triggering motion: {e}")
            return py_trees.common.Status.FAILURE

    def terminate(self):
        pass