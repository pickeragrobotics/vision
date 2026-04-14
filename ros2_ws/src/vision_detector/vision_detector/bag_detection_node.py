from __future__ import annotations

from typing import Optional

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray, Detection3DArray

from vision_detector.bag_reader import FramePair, create_frame_reader
from vision_detector.config import DetectorConfig
from vision_detector.debug_utils import draw_detections
from vision_detector.depth_utils import (
    compute_bbox_center_pixel,
    deproject_pixel_to_3d,
    get_depth_at_pixel_m,
    estimate_detection_physical_volume_m,
)
from vision_detector.image_utils import convert_to_rgb, flip_if_needed
from vision_detector.message_utils import DetectionMessageBuilder
from vision_detector.yolo_detector import DetectionResult, YoloDetector


class BagYoloDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("bag_yolo_detector")

        self._config = DetectorConfig.from_node(self)
        self._publisher = self.create_publisher(
            Detection2DArray, self._config.detection_topic, 10
        )
        self._publisher_3d = self.create_publisher(
            Detection3DArray, self._config.detection_3d_topic, 10
        )
        self._color_image_publisher = self.create_publisher(
            Image, self._config.color_topic, 10
        )
        self._depth_image_publisher = self.create_publisher(
            Image, self._config.depth_topic, 10
        )
        self._camera_info_publisher = self.create_publisher(
            CameraInfo, self._config.camera_info_topic, 10
        )
        self._debug_publisher = None
        self._cv_bridge = CvBridge()
        self._missing_intrinsics_warned = False
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
        self._frame_reader = self._create_frame_reader()
        self._frame_iterator = iter(self._frame_reader)
        self._processed_frames = 0
        self._published_detections = 0
        self._completed_loops = 0
        self._shutdown_timer: Optional[object] = None
        self._processing_timer = self.create_timer(
            self._config.processing_period_sec, self._process_next_frame
        )

        self.get_logger().info(
            "Bag YOLO detector initialized. "
            f"Publishing detections to '{self._config.detection_topic}' "
            f"and '{self._config.detection_3d_topic}', "
            f"and replaying input images to '{self._config.color_topic}' "
            f"and '{self._config.depth_topic}'."
        )
        if self._config.flip_image:
            self.get_logger().info("Input image rotation is enabled (180 degrees).")
        if self._config.debug_mode:
            self.get_logger().info(
                f"Debug image publishing enabled on '{self._config.debug_image_topic}'."
            )

    def _filter_detections_by_physical_size(
        self, detections: list[DetectionResult], frame_pair: FramePair
    ) -> list[DetectionResult]:
        if not detections:
            return detections
        if frame_pair.depth is None or frame_pair.camera_intrinsics is None:
            return detections

        filtered_detections = []
        min_volume_m = float(self._config.min_volume_m)
        max_volume_m = float(self._config.max_volume_m)

        for detection in detections:
            physical_volume = estimate_detection_physical_volume_m(
                detection=detection,
                frame_pair=frame_pair,
                depth_scale_meters=self._config.depth_scale_meters,
            )
            if physical_volume is None:
                filtered_detections.append(detection)
                continue

            if min_volume_m <= physical_volume <= max_volume_m:
                filtered_detections.append(detection)
                continue

            self.get_logger().info(
                "Filtered detection "
                f"'{detection.class_name}' confidence={detection.confidence:.2f}: "
                f"estimated volume={physical_volume:.7f}m³ "
                f"outside valid range [{min_volume_m:.3f}, {max_volume_m:.3f}]m."
            )

        return filtered_detections

    def _process_next_frame(self) -> None:
        try:
            frame_pair = next(self._frame_iterator)
        except StopIteration:
            self._restart_frame_reader()
            return
        except Exception as exc:
            self.get_logger().error(f"Pipeline failed: {exc}")
            self._processing_timer.cancel()
            if self._shutdown_timer is None:
                self._shutdown_timer = self.create_timer(0.2, self._shutdown)
            return

        self._publish_input_frames(frame_pair)
        self._apply_image_orientation(frame_pair)
        detections = self._run_inference(frame_pair)
        detection_array_2d = self._message_builder.build_2d(frame_pair.color, detections)
        detection_array_3d = self._message_builder.build_3d(frame_pair.color, detections)
        self._publisher.publish(detection_array_2d)
        self._publisher_3d.publish(detection_array_3d)
        self._publish_debug_image(frame_pair, detections)

        self._processed_frames += 1
        self._published_detections += len(detections)

        if self._processed_frames % self._config.log_every_n_frames == 0:
            self.get_logger().info(
                f"Processed {self._processed_frames} frames, "
                f"published {self._published_detections} detections so far "
                f"across {self._completed_loops} completed loop(s)."
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
        return self._filter_detections_by_physical_size(detections, frame_pair)

    def _create_frame_reader(self):
        return create_frame_reader(
            bag_path=self._config.bag_path,
            bag_reader_backend=self._config.bag_reader_backend,
            bag_storage_id=self._config.bag_storage_id,
            color_topic=self._config.color_topic,
            depth_topic=self._config.depth_topic,
            camera_info_topic=self._config.camera_info_topic,
            sync_tolerance_sec=self._config.sync_tolerance_sec,
            logger=self.get_logger(),
        )

    def _restart_frame_reader(self) -> None:
        self._completed_loops += 1
        self.get_logger().info(
            f"Reached end of bag playback after {self._processed_frames} frames and "
            f"{self._published_detections} detections. Restarting loop {self._completed_loops}."
        )
        close_reader = getattr(self._frame_reader, "close", None)
        if callable(close_reader):
            close_reader()
        self._frame_reader = self._create_frame_reader()
        self._frame_iterator = iter(self._frame_reader)

    def _publish_input_frames(self, frame_pair: FramePair) -> None:
        color_image = self._cv_bridge.cv2_to_imgmsg(
            frame_pair.color.image, encoding=frame_pair.color.encoding
        )
        color_image.header.stamp = frame_pair.color.stamp
        color_image.header.frame_id = frame_pair.color.frame_id
        self._color_image_publisher.publish(color_image)

        if frame_pair.depth is not None:
            depth_image = self._cv_bridge.cv2_to_imgmsg(
                frame_pair.depth.image, encoding=frame_pair.depth.encoding
            )
            depth_image.header.stamp = frame_pair.depth.stamp
            depth_image.header.frame_id = frame_pair.depth.frame_id
            self._depth_image_publisher.publish(depth_image)

        if frame_pair.camera_intrinsics is not None:
            camera_info = self._build_camera_info(frame_pair)
            self._camera_info_publisher.publish(camera_info)

    def _build_camera_info(self, frame_pair: FramePair) -> CameraInfo:
        intrinsics = frame_pair.camera_intrinsics
        if intrinsics is None:
            raise ValueError("Camera intrinsics are required to build CameraInfo.")

        image_height, image_width = frame_pair.color.image.shape[:2]
        camera_info = CameraInfo()
        camera_info.header.stamp = frame_pair.color.stamp
        camera_info.header.frame_id = intrinsics.frame_id
        camera_info.height = image_height
        camera_info.width = image_width
        camera_info.distortion_model = "plumb_bob"
        camera_info.d = []
        camera_info.k = [
            intrinsics.fx,
            0.0,
            intrinsics.cx,
            0.0,
            intrinsics.fy,
            intrinsics.cy,
            0.0,
            0.0,
            1.0,
        ]
        camera_info.r = [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        camera_info.p = [
            intrinsics.fx,
            0.0,
            intrinsics.cx,
            0.0,
            0.0,
            intrinsics.fy,
            intrinsics.cy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return camera_info

    def _publish_debug_image(
        self, frame_pair: FramePair, detections: list[DetectionResult]
    ) -> None:
        if self._debug_publisher is None:
            return

        annotated_image = draw_detections(frame_pair.color.image, detections)
        debug_image = self._cv_bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
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

    def _shutdown(self) -> None:
        if self._shutdown_timer is not None:
            self._shutdown_timer.cancel()
        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = BagYoloDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            close_reader = getattr(node._frame_reader, "close", None)
            if callable(close_reader):
                close_reader()
            node.destroy_node()
            try:
                rclpy.shutdown()
            except rclpy._rclpy_pybind11.RCLError:
                pass
        if rclpy.ok():
            rclpy.shutdown()
