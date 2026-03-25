from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    yolo_weights_path = LaunchConfiguration("yolo_weights_path")
    flip_image = LaunchConfiguration("flip_image")
    debug_mode = LaunchConfiguration("debug_mode")
    debug_image_topic = LaunchConfiguration("debug_image_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    detection_3d_topic = LaunchConfiguration("detection_3d_topic")
    start_delay_sec = LaunchConfiguration("start_delay_sec")

    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace="camera",
        name="camera",
        output="screen",
        parameters=[
            {
                "device_type": "D405",
                "enable_color": True,
                "enable_depth": True,
                "align_depth.enable": True,
                "enable_sync": True,
            }
        ],
        # remappings=[
        #     ("/camera/aligned_depth_to_color/image_raw", "/camera/depth/image_raw"),
        # ],
    )

    detector_node = Node(
        package="vision_bag_detector",
        executable="camera_yolo_detector",
        name="camera_yolo_detector",
        output="screen",
        parameters=[
            params_file,
            {
                "yolo_weights_path": yolo_weights_path,
                "flip_image": ParameterValue(flip_image, value_type=bool),
                "debug_mode": ParameterValue(debug_mode, value_type=bool),
                "debug_image_topic": ParameterValue(debug_image_topic, value_type=str),
                "camera_info_topic": ParameterValue(camera_info_topic, value_type=str),
                "detection_3d_topic": ParameterValue(detection_3d_topic, value_type=str),
            },
        ],
    )

    delayed_detector_start = RegisterEventHandler(
        OnProcessStart(
            target_action=realsense_node,
            on_start=[TimerAction(period=start_delay_sec, actions=[detector_node])],
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("vision_bag_detector"),
                        "config",
                        "camera_yolo_detector.params.yaml",
                    ]
                ),
                description="Optional ROS 2 parameter file for the camera_yolo_detector node.",
            ),
            DeclareLaunchArgument(
                "yolo_weights_path",
                default_value="/workspace/data/weights/avocado_detection_v2.pt",
                description="Path to the YOLO weights file.",
            ),
            DeclareLaunchArgument(
                "flip_image",
                default_value="true",
                description="Rotate the input image stream by 180 degrees before processing.",
            ),
            DeclareLaunchArgument(
                "camera_info_topic",
                default_value="/camera/camera/color/camera_info",
                description="CameraInfo topic used to extract camera intrinsics.",
            ),
            DeclareLaunchArgument(
                "detection_3d_topic",
                default_value="/detections_3d",
                description="Topic used for published 3D detections.",
            ),
            DeclareLaunchArgument(
                "debug_mode",
                default_value="true",
                description="Enable annotated detection image publishing.",
            ),
            DeclareLaunchArgument(
                "debug_image_topic",
                default_value="/debug/detections_image",
                description="Topic used for annotated detection images.",
            ),
            DeclareLaunchArgument(
                "start_delay_sec",
                default_value="2.0",
                description="Delay before starting the detector node after the RealSense driver starts.",
            ),
            realsense_node,
            delayed_detector_start,
        ]
    )
