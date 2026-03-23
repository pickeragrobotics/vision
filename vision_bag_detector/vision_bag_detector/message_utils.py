from __future__ import annotations

from typing import Iterable

from geometry_msgs.msg import Pose2D
from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose

from vision_bag_detector.bag_reader import ImageFrame
from vision_bag_detector.yolo_detector import DetectionResult


class DetectionMessageBuilder:
    def build(
        self, color_frame: ImageFrame, detections: Iterable[DetectionResult]
    ) -> Detection2DArray:
        header = Header(stamp=color_frame.stamp, frame_id=color_frame.frame_id)
        detection_array = Detection2DArray(header=header)

        for index, detection in enumerate(detections):
            x_min, y_min, x_max, y_max = detection.bbox_xyxy
            width = max(0.0, x_max - x_min)
            height = max(0.0, y_max - y_min)
            center_x = x_min + (width / 2.0)
            center_y = y_min + (height / 2.0)

            detection_msg = Detection2D()
            detection_msg.header = header
            detection_msg.id = f"{color_frame.stamp_ns}-{index}"
            detection_msg.bbox = BoundingBox2D(
                center=Pose2D(x=center_x, y=center_y, theta=0.0),
                size_x=width,
                size_y=height,
            )

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = detection.class_name
            hypothesis.hypothesis.score = detection.confidence

            # The median aligned depth is stored in pose.position.z when available.
            if detection.depth_m is not None:
                hypothesis.pose.pose.position.z = detection.depth_m

            detection_msg.results.append(hypothesis)
            detection_array.detections.append(detection_msg)

        return detection_array
