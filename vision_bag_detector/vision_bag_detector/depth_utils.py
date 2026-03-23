from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from vision_bag_detector.bag_reader import ImageFrame


def clamp_bbox_to_image(
    bbox_xyxy: Tuple[float, float, float, float], image_shape: Tuple[int, int]
) -> Optional[Tuple[int, int, int, int]]:
    height, width = image_shape
    x_min, y_min, x_max, y_max = bbox_xyxy
    x_min = max(0, min(int(round(x_min)), width - 1))
    y_min = max(0, min(int(round(y_min)), height - 1))
    x_max = max(0, min(int(round(x_max)), width))
    y_max = max(0, min(int(round(y_max)), height))
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def compute_median_depth_m(
    depth_frame: Optional[ImageFrame],
    bbox_xyxy: Tuple[float, float, float, float],
    fallback_depth_scale_meters: float,
) -> Optional[float]:
    if depth_frame is None:
        return None

    clamped_bbox = clamp_bbox_to_image(bbox_xyxy, depth_frame.image.shape[:2])
    if clamped_bbox is None:
        return None

    x_min, y_min, x_max, y_max = clamped_bbox
    box_width = x_max - x_min
    box_height = y_max - y_min

    # Use a center crop inside the box to reduce background contamination.
    crop_margin_x = max(1, int(box_width * 0.25))
    crop_margin_y = max(1, int(box_height * 0.25))
    inner_x_min = min(x_max - 1, x_min + crop_margin_x)
    inner_y_min = min(y_max - 1, y_min + crop_margin_y)
    inner_x_max = max(inner_x_min + 1, x_max - crop_margin_x)
    inner_y_max = max(inner_y_min + 1, y_max - crop_margin_y)

    roi = depth_frame.image[inner_y_min:inner_y_max, inner_x_min:inner_x_max]
    if roi.size == 0:
        roi = depth_frame.image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None

    if np.issubdtype(roi.dtype, np.integer):
        scale = depth_frame.depth_scale_meters or fallback_depth_scale_meters
        depth_m = roi.astype(np.float32) * float(scale)
    else:
        depth_m = roi.astype(np.float32)

    valid_depths = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
    if valid_depths.size == 0:
        return None

    return float(np.median(valid_depths))
