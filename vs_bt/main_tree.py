#!/usr/bin/env python3

"""Script contains the main tree which supervises other functionalities and behaviours"""

import rclpy
import py_trees
from ros2_reader import ReadfromROS
from ComDependencies.robot_controller import robot


def setBlackboard(blackboard):
    pass

def main():
    rclpy.init()
    node = rclpy.create_node("simple_node")
    realRobot = robot("192.168.1.9")

    # initialize root
    root = py_trees.composites.Selector("root", memory=False)
    read_from_ros = ReadfromROS(node)
    root.add_child(read_from_ros)

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
            period_ms=500,
            number_of_iterations=py_trees.trees.CONTINUOUS_TICK_TOCK,
            pre_tick_handler=None,
            post_tick_handler=print_tree,
        )
    except KeyboardInterrupt:
        behaviour_tree.interrupt()


if __name__ == "__main__":
    main()