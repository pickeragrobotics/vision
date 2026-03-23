from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from vision_bag_detector.yolo_detector import DetectionResult


def draw_detections(image_bgr: np.ndarray, detections: Sequence[DetectionResult]) -> np.ndarray:
    annotated = image_bgr.copy()
    image_height, image_width = annotated.shape[:2]
    line_thickness = max(2, int(round(min(image_height, image_width) / 300)))
    font_scale = max(0.5, min(image_height, image_width) / 900.0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for detection in detections:
        x_min, y_min, x_max, y_max = detection.bbox_xyxy
        left = max(0, min(image_width - 1, int(round(x_min))))
        top = max(0, min(image_height - 1, int(round(y_min))))
        right = max(0, min(image_width - 1, int(round(x_max))))
        bottom = max(0, min(image_height - 1, int(round(y_max))))

        if right <= left or bottom <= top:
            continue

        color = _color_for_detection(detection.class_id)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.rectangle(annotated, (left, top), (right, bottom), color, line_thickness)

        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, max(1, line_thickness - 1)
        )
        text_top = max(0, top - text_height - baseline - 6)
        text_bottom = min(image_height - 1, text_top + text_height + baseline + 6)
        text_right = min(image_width - 1, left + text_width + 8)
        cv2.rectangle(annotated, (left, text_top), (text_right, text_bottom), color, -1)
        cv2.putText(
            annotated,
            label,
            (left + 4, text_bottom - baseline - 3),
            font,
            font_scale,
            (20, 20, 20),
            max(1, line_thickness - 1),
            cv2.LINE_AA,
        )

    return annotated


def _color_for_detection(class_id: int) -> tuple[int, int, int]:
    palette = (
        (40, 220, 120),
        (30, 180, 255),
        (0, 215, 255),
        (255, 170, 0),
        (235, 99, 71),
        (180, 105, 255),
    )
    return palette[class_id % len(palette)]
