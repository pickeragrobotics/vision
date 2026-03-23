from __future__ import annotations

from typing import Iterable

from geometry_msgs.msg import Pose2D
from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose, Point2D

from vision_bag_detector.bag_reader import ImageFrame
from vision_bag_detector.yolo_detector import DetectionResult


class DetectionMessageBuilder:
    _pose2d_debug_printed = False

    @staticmethod
    def _build_bbox_center(
        bbox_pose2d_type: type, center_x: float, center_y: float
    ) -> object:
        pose_fields = bbox_pose2d_type.get_fields_and_field_types()
        center_x = float(center_x)
        center_y = float(center_y)

        if {"x", "y", "theta"}.issubset(pose_fields):
            return bbox_pose2d_type(x=center_x, y=center_y, theta=0.0)

        if {"position", "theta"}.issubset(pose_fields):
            return bbox_pose2d_type(position=Point2D(x=center_x, y=center_y), theta=0.0)

        raise TypeError(f"Unsupported BoundingBox2D center fields: {pose_fields}")

    def build(
        self, color_frame: ImageFrame, detections: Iterable[DetectionResult]
    ) -> Detection2DArray:
        header = Header(stamp=color_frame.stamp, frame_id=color_frame.frame_id)
        detection_array = Detection2DArray(header=header)
        bbox_pose2d_type = BoundingBox2D().center.__class__

        if not self._pose2d_debug_printed:
            print("Pose2D module:", Pose2D.__module__)
            print("Pose2D type:", type(Pose2D()))
            print("BoundingBox2D.center module:", bbox_pose2d_type.__module__)
            print("BoundingBox2D.center type:", bbox_pose2d_type)
            print("BoundingBox2D.center fields:", bbox_pose2d_type.get_fields_and_field_types())
            print("Same type:", bbox_pose2d_type is Pose2D)
            self._pose2d_debug_printed = True

        for index, detection in enumerate(detections):
            x_min, y_min, x_max, y_max = detection.bbox_xyxy
            width = float(max(0.0, x_max - x_min))
            height = float(max(0.0, y_max - y_min))
            center_x = float(x_min + (width / 2.0))
            center_y = float(y_min + (height / 2.0))

            detection_msg = Detection2D()
            detection_msg.header = header
            detection_msg.id = f"{color_frame.stamp_ns}-{index}"
            bbox = BoundingBox2D()
            # Build the exact center message type expected by the installed vision_msgs package.
            bbox.center = self._build_bbox_center(bbox_pose2d_type, center_x, center_y)
            assert (
                bbox.center.__class__ is bbox_pose2d_type
            ), f"Mismatch: {bbox.center.__class__} vs {bbox_pose2d_type}"
            bbox.size_x = width
            bbox.size_y = height
            detection_msg.bbox = bbox

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = detection.class_name
            hypothesis.hypothesis.score = detection.confidence

            # The median aligned depth is stored in pose.position.z when available.
            if detection.depth_m is not None:
                hypothesis.pose.pose.position.z = detection.depth_m

            detection_msg.results.append(hypothesis)
            detection_array.detections.append(detection_msg)

        return detection_array
