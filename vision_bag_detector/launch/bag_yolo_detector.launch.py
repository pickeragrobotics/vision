from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("vision_bag_detector"), "config", "bag_yolo_detector.params.yaml"]
                ),
                description="Optional ROS 2 parameter file for the bag_yolo_detector node.",
            ),
            DeclareLaunchArgument(
                "bag_path",
                default_value="/workspace/data/real_sense_bags/20260312_100746.bag",
                description="Path to a ROS 2 bag URI or a native RealSense .bag file.",
            ),
            DeclareLaunchArgument(
                "yolo_weights_path",
                default_value="/workspace/data/weights/avocado_detection_v2.pt",
                description="Path to the YOLO weights file.",
            ),
            Node(
                package="vision_bag_detector",
                executable="bag_yolo_detector",
                name="bag_yolo_detector",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "bag_path": LaunchConfiguration("bag_path"),
                        "yolo_weights_path": LaunchConfiguration("yolo_weights_path"),
                    },
                ],
            ),
        ]
    )
