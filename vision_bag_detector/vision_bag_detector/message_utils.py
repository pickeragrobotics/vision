from __future__ import annotations

from typing import Iterable

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import Header
from vision_msgs.msg import (
    BoundingBox2D,
    BoundingBox3D,
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
    Point2D,
)

from vision_bag_detector.bag_reader import ImageFrame
from vision_bag_detector.yolo_detector import DetectionResult


class DetectionMessageBuilder:
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

    @staticmethod
    def _build_header(color_frame: ImageFrame) -> Header:
        return Header(stamp=color_frame.stamp, frame_id=color_frame.frame_id)

    @staticmethod
    def _build_pose(x_m: float, y_m: float, z_m: float) -> Pose:
        return Pose(
            position=Point(x=float(x_m), y=float(y_m), z=float(z_m)),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )

    def build(
        self, color_frame: ImageFrame, detections: Iterable[DetectionResult]
    ) -> Detection2DArray:
        return self.build_2d(color_frame, detections)

    def build_2d(
        self, color_frame: ImageFrame, detections: Iterable[DetectionResult]
    ) -> Detection2DArray:
        header = self._build_header(color_frame)
        detection_array = Detection2DArray(header=header)
        bbox_pose2d_type = BoundingBox2D().center.__class__

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

            if detection.depth_m is not None:
                hypothesis.pose.pose.position.z = detection.depth_m

            detection_msg.results.append(hypothesis)
            detection_array.detections.append(detection_msg)

        return detection_array

    def build_3d(
        self, color_frame: ImageFrame, detections: Iterable[DetectionResult]
    ) -> Detection3DArray:
        header = self._build_header(color_frame)
        detection_array = Detection3DArray(header=header)

        for index, detection in enumerate(detections):
            if detection.position_xyz is None:
                continue

            x_m, y_m, z_m = detection.position_xyz
            pose = self._build_pose(x_m, y_m, z_m)

            detection_msg = Detection3D()
            detection_msg.header = header
            detection_msg.id = f"{color_frame.stamp_ns}-{index}"

            bbox = BoundingBox3D()
            bbox.center = pose
            bbox.size = Vector3(x=0.0, y=0.0, z=0.0)
            detection_msg.bbox = bbox

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(detection.class_id)
            hypothesis.hypothesis.score = detection.confidence
            hypothesis.pose.pose = pose
            detection_msg.results.append(hypothesis)

            detection_array.detections.append(detection_msg)

        return detection_array
