""" Static transform publisher acquired via OpenCV hand-eye calibration - Fingers Camera """
""" EYE-IN-HAND: link_6 -> camera_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "tool0",
                "--child-frame-id",
                "camera_link",
                "--x",
                "0.06747056",
                "--y",
                "-0.02257421",
                "--z",
                "0.13446383",
                "--qx",
                "-0.02234217",
                "--qy",
                "-0.02735988",
                "--qz",
                "0.59807405",
                "--qw",
                "0.80066204",
            ],
        ),
    ]
    return LaunchDescription(nodes)
