from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class DetectionResult:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    depth_m: float | None = None
    center_pixel_xy: Tuple[int, int] | None = None
    position_xyz: Tuple[float, float, float] | None = None


class YoloDetector:
    def __init__(self, weights_path: str, confidence_threshold: float, device: str = "") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install it with 'pip install ultralytics'."
            ) from exc

        self._model = YOLO(weights_path)
        self._confidence_threshold = confidence_threshold
        self._device = device

    def predict(self, color_image_bgr: np.ndarray) -> List[DetectionResult]:
        prediction_kwargs = {
            "conf": self._confidence_threshold,
            "verbose": False,
        }
        if self._device:
            prediction_kwargs["device"] = self._device

        predictions = self._model.predict(color_image_bgr, **prediction_kwargs)
        if not predictions:
            return []

        prediction = predictions[0]
        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            return []

        class_names = self._model.names
        detections: List[DetectionResult] = []

        for index in range(len(boxes)):
            x_min, y_min, x_max, y_max = boxes.xyxy[index].tolist()
            confidence = float(boxes.conf[index].item())
            class_id = int(boxes.cls[index].item())
            if isinstance(class_names, dict):
                class_name = str(class_names.get(class_id, class_id))
            else:
                class_name = str(class_names[class_id])

            detections.append(
                DetectionResult(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=(x_min, y_min, x_max, y_max),
                )
            )

        return detections
