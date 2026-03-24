from dataclasses import dataclass
from pathlib import Path

from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node


@dataclass(frozen=True)
class DetectorConfig:
    bag_path: str
    bag_reader_backend: str
    bag_storage_id: str
    color_topic: str
    depth_topic: str
    flip_image: bool
    detection_topic: str
    debug_mode: bool
    debug_image_topic: str
    confidence_threshold: float
    sync_tolerance_sec: float
    depth_scale_meters: float
    yolo_weights_path: str
    yolo_device: str
    processing_period_sec: float
    log_every_n_frames: int

    @classmethod
    def from_node(cls, node: Node) -> "DetectorConfig":
        node.declare_parameter(
            "bag_path",
            "",
            ParameterDescriptor(description="Path to a ROS 2 bag URI or a native RealSense .bag file."),
        )
        node.declare_parameter(
            "bag_reader_backend",
            "auto",
            ParameterDescriptor(
                description="Bag reader backend. Valid values: auto, rosbag2, realsense."
            ),
        )
        node.declare_parameter(
            "bag_storage_id",
            "sqlite3",
            ParameterDescriptor(description="Storage plugin identifier used for rosbag2 input."),
        )
        node.declare_parameter(
            "color_topic",
            "/camera/color/image_raw",
            ParameterDescriptor(description="Color image topic to read from a rosbag2 bag."),
        )
        node.declare_parameter(
            "depth_topic",
            "/camera/aligned_depth_to_color/image_raw",
            ParameterDescriptor(description="Depth image topic to read from a rosbag2 bag."),
        )
        node.declare_parameter(
            "flip_image",
            True,
            ParameterDescriptor(
                description="Rotate color and aligned depth images by 180 degrees before processing."
            ),
        )
        node.declare_parameter(
            "detection_topic",
            "/detections",
            ParameterDescriptor(description="Output topic for published vision_msgs/Detection2DArray messages."),
        )
        node.declare_parameter(
            "debug_mode",
            False,
            ParameterDescriptor(
                description="When true, publish annotated detection images for visualization."
            ),
        )
        node.declare_parameter(
            "debug_image_topic",
            "/debug/detections_image",
            ParameterDescriptor(
                description="Output topic for annotated detection images when debug_mode is enabled."
            ),
        )
        node.declare_parameter(
            "confidence_threshold",
            0.25,
            ParameterDescriptor(description="Minimum confidence score accepted from the YOLO model."),
        )
        node.declare_parameter(
            "sync_tolerance_sec",
            0.03,
            ParameterDescriptor(description="Maximum time delta allowed between color and depth frames."),
        )
        node.declare_parameter(
            "depth_scale_meters",
            0.001,
            ParameterDescriptor(description="Meters-per-unit scale for integer depth images when no native scale is available."),
        )
        node.declare_parameter(
            "yolo_weights_path",
            "",
            ParameterDescriptor(description="Path to the YOLO weights file."),
        )
        node.declare_parameter(
            "yolo_device",
            "",
            ParameterDescriptor(description="Optional Ultralytics device string such as cpu, 0, or 0,1."),
        )
        node.declare_parameter(
            "processing_period_sec",
            0.001,
            ParameterDescriptor(description="Timer period used to process the next synchronized frame pair."),
        )
        node.declare_parameter(
            "log_every_n_frames",
            30,
            ParameterDescriptor(description="Emit an info log every N processed frames."),
        )

        config = cls(
            bag_path=node.get_parameter("bag_path").get_parameter_value().string_value,
            bag_reader_backend=node.get_parameter("bag_reader_backend").get_parameter_value().string_value,
            bag_storage_id=node.get_parameter("bag_storage_id").get_parameter_value().string_value,
            color_topic=node.get_parameter("color_topic").get_parameter_value().string_value,
            depth_topic=node.get_parameter("depth_topic").get_parameter_value().string_value,
            flip_image=node.get_parameter("flip_image").value,
            detection_topic=node.get_parameter("detection_topic").get_parameter_value().string_value,
            debug_mode=node.get_parameter("debug_mode").value,
            debug_image_topic=node.get_parameter("debug_image_topic").get_parameter_value().string_value,
            confidence_threshold=node.get_parameter("confidence_threshold").value,
            sync_tolerance_sec=node.get_parameter("sync_tolerance_sec").value,
            depth_scale_meters=node.get_parameter("depth_scale_meters").value,
            yolo_weights_path=node.get_parameter("yolo_weights_path").get_parameter_value().string_value,
            yolo_device=node.get_parameter("yolo_device").get_parameter_value().string_value,
            processing_period_sec=node.get_parameter("processing_period_sec").value,
            log_every_n_frames=node.get_parameter("log_every_n_frames").value,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.bag_path:
            raise ValueError("Parameter 'bag_path' must be provided.")
        if not self.yolo_weights_path:
            raise ValueError("Parameter 'yolo_weights_path' must be provided.")
        if not Path(self.bag_path).exists():
            raise FileNotFoundError(f"Bag path does not exist: {self.bag_path}")
        if not Path(self.yolo_weights_path).exists():
            raise FileNotFoundError(f"YOLO weights path does not exist: {self.yolo_weights_path}")
        if self.bag_reader_backend not in {"auto", "rosbag2", "realsense"}:
            raise ValueError(
                "Parameter 'bag_reader_backend' must be one of: auto, rosbag2, realsense."
            )
        if not self.debug_image_topic:
            raise ValueError("Parameter 'debug_image_topic' must not be empty.")
        if not 0.0 <= float(self.confidence_threshold) <= 1.0:
            raise ValueError("Parameter 'confidence_threshold' must be between 0.0 and 1.0.")
        if float(self.sync_tolerance_sec) < 0.0:
            raise ValueError("Parameter 'sync_tolerance_sec' must be non-negative.")
        if float(self.depth_scale_meters) <= 0.0:
            raise ValueError("Parameter 'depth_scale_meters' must be positive.")
        if float(self.processing_period_sec) <= 0.0:
            raise ValueError("Parameter 'processing_period_sec' must be positive.")
        if int(self.log_every_n_frames) <= 0:
            raise ValueError("Parameter 'log_every_n_frames' must be positive.")
