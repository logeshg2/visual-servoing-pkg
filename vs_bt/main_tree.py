#!/usr/bin/env python3

"""Script contains the main tree which supervises other functionalities and behaviours"""

import rclpy
import py_trees
import numpy as np
from handler import Handler
from ros2_reader import ReadfromROS
from check_motion import CheckMovement
from move_arm import MoveArm, ControlType
from visual_servoing import VisualServoing
from ComDependencies.robot_controller import robot


tracking_pose = np.array([60.0, 240.0, 120.0, 179.65, 0.69, 67.63])

def setBlackboard(blackboard):
    """Initialize default parameters in blackboard"""
    
    blackboard.set("tracking_pose", tracking_pose)

def main():
    rclpy.init()
    node = rclpy.create_node("simple_node")
    realRobot = robot("192.168.1.9")

    # initialize root
    root = py_trees.composites.Sequence("root", memory=False)
    read_from_ros = ReadfromROS(node)
    course_handler = Handler(realRobot)
    check_movement = CheckMovement(realRobot)
    # visual_servoing = VisualServoing()
    move_arm = MoveArm(realRobot, ControlType.camVelCtrl)
    # construct root + tree
    root.add_children([read_from_ros, check_movement, course_handler, move_arm])

    # initialize tree
    behaviour_tree = py_trees.trees.BehaviourTree(root=root)
    print(py_trees.display.unicode_tree(root=root))
    behaviour_tree.setup(timeout=15)

    # initialize blackboard
    blackboard = py_trees.blackboard.Blackboard()
    setBlackboard(blackboard)

    def print_tree(tree: py_trees.trees.BehaviourTree) -> None:
        """Print the behaviour tree and its current status."""
        print(py_trees.display.unicode_tree(root=tree.root, show_status=True))

    try:
        behaviour_tree.tick_tock(
            period_ms=5,
            number_of_iterations=py_trees.trees.CONTINUOUS_TICK_TOCK,
            pre_tick_handler=None,
            post_tick_handler=print_tree,
        )
    except KeyboardInterrupt:
        behaviour_tree.interrupt()


if __name__ == "__main__":
    main()