from __future__ import annotations

from typing import Optional

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray, Detection3DArray

from vision_detector.bag_reader import CameraIntrinsics, FramePair, ImageFrame
from vision_detector.config import LiveDetectorConfig
from vision_detector.debug_utils import draw_detections
from vision_detector.depth_utils import (
    compute_bbox_center_pixel,
    deproject_pixel_to_3d,
    get_depth_at_pixel_m,
)
from vision_detector.image_utils import convert_to_rgb, flip_if_needed
from vision_detector.message_utils import DetectionMessageBuilder
from vision_detector.yolo_detector import DetectionResult, YoloDetector


class CameraYoloDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_yolo_detector")

        self._config = LiveDetectorConfig.from_node(self)
        self._cv_bridge = CvBridge()
        self._debug_bridge = CvBridge() if self._config.debug_mode else None
        self._publisher = self.create_publisher(
            Detection2DArray, self._config.detection_topic, 10
        )
        self._publisher_3d = self.create_publisher(
            Detection3DArray, self._config.detection_3d_topic, 10
        )
        self._debug_publisher = None
        if self._config.debug_mode:
            self._debug_publisher = self.create_publisher(
                Image, self._config.debug_image_topic, 10
            )

        self._message_builder = DetectionMessageBuilder()
        self._detector = YoloDetector(
            weights_path=self._config.yolo_weights_path,
            confidence_threshold=self._config.confidence_threshold,
            device=self._config.yolo_device,
        )

        self._latest_color_frame: Optional[ImageFrame] = None
        self._latest_depth_frame: Optional[ImageFrame] = None
        self._camera_intrinsics: Optional[CameraIntrinsics] = None
        self._last_processed_color_stamp_ns: Optional[int] = None
        self._missing_intrinsics_warned = False

        self._processed_frames = 0
        self._published_detections = 0

        self.get_logger().info(
            f"Subscribing to color, depth, and camera info topics..."
            f" color topic '{self._config.color_topic}',"
            f" depth topic '{self._config.depth_topic}',"
            f" camera info topic '{self._config.camera_info_topic}'"
        )

        self.create_subscription(
            Image,
            self._config.color_topic,
            self._on_color_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self._config.depth_topic,
            self._on_depth_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            self._config.camera_info_topic,
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self._processing_timer = self.create_timer(
            self._config.processing_period_sec, self._process_latest_frame
        )

        self.get_logger().info(
            "Camera YOLO detector initialized. "
            f"Subscribing to '{self._config.color_topic}', '{self._config.depth_topic}', "
            f"and '{self._config.camera_info_topic}'. "
            f"Publishing detections to '{self._config.detection_topic}' "
            f"and '{self._config.detection_3d_topic}'."
        )
        if self._config.flip_image:
            self.get_logger().info("Input image rotation is enabled (180 degrees).")
        if self._config.debug_mode:
            self.get_logger().info(
                f"Debug image publishing enabled on '{self._config.debug_image_topic}'."
            )

    def _on_color_image(self, message: Image) -> None:
        color_image = self._cv_bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        self.get_logger().debug(
            f"Received color image with timestamp {message.header.stamp.sec}."
        )   
        self._latest_color_frame = ImageFrame(
            stamp=message.header.stamp,
            stamp_ns=self._stamp_to_ns(message.header.stamp.sec, message.header.stamp.nanosec),
            frame_id=message.header.frame_id,
            image=color_image,
            encoding="bgr8",
        )

    def _on_depth_image(self, message: Image) -> None:
        depth_image = self._cv_bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        self._latest_depth_frame = ImageFrame(
            stamp=message.header.stamp,
            stamp_ns=self._stamp_to_ns(message.header.stamp.sec, message.header.stamp.nanosec),
            frame_id=message.header.frame_id,
            image=depth_image,
            encoding=message.encoding,
        )

    def _on_camera_info(self, message: CameraInfo) -> None:
        self._camera_intrinsics = CameraIntrinsics(
            fx=float(message.k[0]),
            fy=float(message.k[4]),
            cx=float(message.k[2]),
            cy=float(message.k[5]),
            frame_id=message.header.frame_id,
        )

    def _process_latest_frame(self) -> None:
        if self._latest_color_frame is None:
            return
        if self._last_processed_color_stamp_ns == self._latest_color_frame.stamp_ns:
            return

        frame_pair = self._build_frame_pair()
        if frame_pair is None:
            return

        self._apply_image_orientation(frame_pair)
        detections = self._run_inference(frame_pair)
        detection_array_2d = self._message_builder.build_2d(frame_pair.color, detections)
        detection_array_3d = self._message_builder.build_3d(frame_pair.color, detections)
        self._publisher.publish(detection_array_2d)
        self._publisher_3d.publish(detection_array_3d)
        self._publish_debug_image(frame_pair, detections)

        self._processed_frames += 1
        self._published_detections += len(detections)
        self._last_processed_color_stamp_ns = frame_pair.color.stamp_ns

        if self._processed_frames % self._config.log_every_n_frames == 0:
            self.get_logger().info(
                f"Processed {self._processed_frames} frames, "
                f"published {self._published_detections} detections so far."
            )

    def _build_frame_pair(self) -> Optional[FramePair]:
        if self._latest_color_frame is None:
            return None

        matched_depth = None
        if self._latest_depth_frame is not None:
            depth_delta_ns = abs(
                self._latest_depth_frame.stamp_ns - self._latest_color_frame.stamp_ns
            )
            max_delta_ns = int(self._config.sync_tolerance_sec * 1_000_000_000)
            if depth_delta_ns <= max_delta_ns:
                matched_depth = self._clone_image_frame(self._latest_depth_frame)

        return FramePair(
            color=self._clone_image_frame(self._latest_color_frame),
            depth=matched_depth,
            camera_intrinsics=self._camera_intrinsics,
        )

    def _run_inference(self, frame_pair: FramePair) -> list[DetectionResult]:
        detections = self._detector.predict(frame_pair.color.image)
        if frame_pair.depth is None:
            return detections

        for detection in detections:
            detection.center_pixel_xy = compute_bbox_center_pixel(
                bbox_xyxy=detection.bbox_xyxy,
                image_shape=frame_pair.color.image.shape[:2],
            )
            detection.depth_m = get_depth_at_pixel_m(
                depth_frame=frame_pair.depth,
                pixel_xy=detection.center_pixel_xy,
                fallback_depth_scale_meters=self._config.depth_scale_meters,
            )
            if detection.depth_m is None or frame_pair.camera_intrinsics is None:
                continue

            detection.position_xyz = deproject_pixel_to_3d(
                pixel_xy=self._pixel_for_projection(
                    pixel_xy=detection.center_pixel_xy,
                    image_shape=frame_pair.color.image.shape[:2],
                ),
                depth_m=detection.depth_m,
                camera_intrinsics=frame_pair.camera_intrinsics,
            )

        if detections and frame_pair.camera_intrinsics is None and not self._missing_intrinsics_warned:
            self.get_logger().warning(
                "Camera intrinsics are not available, so 3D detections will be skipped."
            )
            self._missing_intrinsics_warned = True
        return detections

    def _publish_debug_image(
        self, frame_pair: FramePair, detections: list[DetectionResult]
    ) -> None:
        if self._debug_publisher is None or self._debug_bridge is None:
            return

        annotated_image = draw_detections(frame_pair.color.image, detections)
        debug_image = self._debug_bridge.cv2_to_imgmsg(annotated_image, encoding="rgb8")
        debug_image.header.stamp = frame_pair.color.stamp
        debug_image.header.frame_id = frame_pair.color.frame_id
        self._debug_publisher.publish(debug_image)

    def _apply_image_orientation(self, frame_pair: FramePair) -> None:
        frame_pair.color.image, frame_pair.color.encoding = convert_to_rgb(
            frame_pair.color.image, frame_pair.color.encoding
        )
        frame_pair.color.image = flip_if_needed(frame_pair.color.image, self._config.flip_image)
        if frame_pair.depth is not None:
            frame_pair.depth.image = flip_if_needed(
                frame_pair.depth.image, self._config.flip_image
            )

    def _pixel_for_projection(
        self, pixel_xy: tuple[int, int], image_shape: tuple[int, int]
    ) -> tuple[int, int]:
        if not self._config.flip_image:
            return pixel_xy

        image_height, image_width = image_shape
        pixel_x, pixel_y = pixel_xy
        return (image_width - 1) - pixel_x, (image_height - 1) - pixel_y

    @staticmethod
    def _stamp_to_ns(sec: int, nanosec: int) -> int:
        return (sec * 1_000_000_000) + nanosec

    @staticmethod
    def _clone_image_frame(frame: ImageFrame) -> ImageFrame:
        return ImageFrame(
            stamp=frame.stamp,
            stamp_ns=frame.stamp_ns,
            frame_id=frame.frame_id,
            image=frame.image.copy(),
            encoding=frame.encoding,
            depth_scale_meters=frame.depth_scale_meters,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = CameraYoloDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
            try:
                rclpy.shutdown()
            except rclpy._rclpy_pybind11.RCLError:
                pass
        if rclpy.ok():
            rclpy.shutdown()
