#!/usr/bin/env python3

"""Script contains the main tree which supervises other functionalities and behaviours"""

import rclpy
import py_trees
import numpy as np

from ros2_reader import ReadfromROS
from check_motion import CheckMovement
from moveToTracking import move2Tracking
from gripperBehaviour import openGripper, closeGripper
from arucoServoBehaviour import (
    startArucoServo,
    servoArucoUntilConv,
    stopArucoServo,
    # pick behaviours
    moveAbovePick,
    moveDownPick,
    moveUpSafe
)
from socketServoBehaviour import (
    startSocketServo,
    servoSocketUntilConv,
    stopSocketServo,
    # insert behaviours
    moveAboveInsert,
    moveDownInsert,
    moveUpSafe_insert
)

from ComDependencies.robot_controller import robot


tracking_pose = np.array([60.0, 240.0, 120.0, 179.65, 0.69, 67.63])

def setBlackboard(blackboard):
    """Initialize default parameters in blackboard"""
    
    blackboard.set("tracking_pose", tracking_pose)


def create_root(rosNode, realRobot):
    """Function that create a root and initializes the entire behaviour tree."""

    # pick and insert sub-trees
    pickSubtree = py_trees.composites.Sequence("Pick Subtree", memory=True)
    pickStaticMovements = py_trees.composites.Sequence("Pick Static Movements", memory=True)
    insertSubtree = py_trees.composites.Sequence("Insert Subtree", memory=True)
    insertStaticMovements = py_trees.composites.Sequence("Insert Static Movements", memory=True)

    # pick subtree
    pickStaticMovements.add_children([
        moveAbovePick(realRobot),
        moveDownPick(realRobot),
        closeGripper(realRobot),
        moveUpSafe(realRobot),
        move2Tracking(realRobot, look_for_trigger=False)
    ])
    pickSubtree.add_children([
        startArucoServo(),
        servoArucoUntilConv(),
        stopArucoServo(),
        pickStaticMovements
    ])

    # insert subtree
    insertStaticMovements.add_children([
        moveAboveInsert(realRobot),
        moveDownInsert(realRobot),
        openGripper(realRobot),
        moveUpSafe_insert(realRobot),
        move2Tracking(realRobot, look_for_trigger=False)
    ])
    insertSubtree.add_children([
        startSocketServo(),
        servoSocketUntilConv(),
        stopSocketServo(),
        insertStaticMovements
    ])

    # high level composites
    systemMonitor = py_trees.composites.Parallel("System Monitor", policy=py_trees.common.ParallelPolicy.SuccessOnAll())
    taskSequence = py_trees.composites.Sequence("Task Sequence", memory=True)

    # system monitor
    systemMonitor.add_children([
        ReadfromROS(rosNode),
        CheckMovement(realRobot)
    ])

    # task sequence
    taskSequence.add_children([
        move2Tracking(realRobot, look_for_trigger=True),
        openGripper(realRobot),
        pickSubtree,
        insertSubtree
    ])

    # main root
    root = py_trees.composites.Parallel("Root", policy=py_trees.common.ParallelPolicy.SuccessOnOne())   # if the task sequence is SUCCESS - stops BT
    
    root.add_children([
        systemMonitor,
        taskSequence
    ])

    return root


def main():
    rclpy.init()
    rosNode = rclpy.create_node("simple_node")
    realRobot = robot("192.168.1.9")

    # create root (+ entire BT)
    root = create_root(rosNode, realRobot)

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
            period_ms=20,
            number_of_iterations=py_trees.trees.CONTINUOUS_TICK_TOCK,
            pre_tick_handler=None,
            post_tick_handler=print_tree,
        )
    except KeyboardInterrupt:
        behaviour_tree.interrupt()


if __name__ == "__main__":
    main()