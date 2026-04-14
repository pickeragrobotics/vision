from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from vision_detector.bag_reader import CameraIntrinsics, ImageFrame, FramePair
from vision_detector.yolo_detector import DetectionResult


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

def estimate_detection_physical_volume_m(
        detection: DetectionResult, frame_pair: FramePair, depth_scale_meters: float
    ) -> Optional[float]:
        intrinsics = frame_pair.camera_intrinsics
        if frame_pair.depth is None or intrinsics is None:
            return None
        if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
            return None

        median_depth_m = compute_median_depth_m(
            depth_frame=frame_pair.depth,
            bbox_xyxy=detection.bbox_xyxy,
            fallback_depth_scale_meters=depth_scale_meters,
        )
        if median_depth_m is None:
            return None

        clamped_bbox = clamp_bbox_to_image(
            detection.bbox_xyxy,
            frame_pair.color.image.shape[:2],
        )
        if clamped_bbox is None:
            return None

        x_min, y_min, x_max, y_max = clamped_bbox
        width_px = max(0, x_max - x_min)
        height_px = max(0, y_max - y_min)
        if width_px == 0 or height_px == 0:
            return None

        width_m = float(width_px) * median_depth_m / intrinsics.fx
        height_m = float(height_px) * median_depth_m / intrinsics.fy
        # Volume of ellipsoid: V = (4/3) * pi * a * b * c
        volume = (4.0 / 3.0) * 3.14159 * (width_m / 2.0) * (height_m / 2.0) * (width_m / 2.0)
        return volume


def compute_bbox_center_pixel(
    bbox_xyxy: Tuple[float, float, float, float], image_shape: Tuple[int, int]
) -> Tuple[int, int]:
    height, width = image_shape
    x_min, y_min, x_max, y_max = bbox_xyxy
    center_x = int(round((float(x_min) + float(x_max)) / 2.0))
    center_y = int(round((float(y_min) + float(y_max)) / 2.0))
    center_x = max(0, min(center_x, width - 1))
    center_y = max(0, min(center_y, height - 1))
    return center_x, center_y


def get_depth_at_pixel_m(
    depth_frame: Optional[ImageFrame],
    pixel_xy: Tuple[int, int],
    fallback_depth_scale_meters: float,
) -> Optional[float]:
    if depth_frame is None:
        return None

    image_height, image_width = depth_frame.image.shape[:2]
    pixel_x, pixel_y = pixel_xy
    if not (0 <= pixel_x < image_width and 0 <= pixel_y < image_height):
        return None

    depth_value = depth_frame.image[pixel_y, pixel_x]
    if np.issubdtype(depth_frame.image.dtype, np.integer):
        scale = depth_frame.depth_scale_meters or fallback_depth_scale_meters
        depth_m = float(depth_value) * float(scale)
    else:
        depth_m = float(depth_value)

    if not np.isfinite(depth_m) or depth_m <= 0.0:
        return None

    return depth_m


def deproject_pixel_to_3d(
    pixel_xy: Tuple[float, float],
    depth_m: float,
    camera_intrinsics: Optional[CameraIntrinsics],
) -> Optional[Tuple[float, float, float]]:
    if camera_intrinsics is None:
        return None
    if camera_intrinsics.fx <= 0.0 or camera_intrinsics.fy <= 0.0:
        return None
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        return None

    pixel_x, pixel_y = pixel_xy
    x_m = (float(pixel_x) - camera_intrinsics.cx) * float(depth_m) / camera_intrinsics.fx
    y_m = (float(pixel_y) - camera_intrinsics.cy) * float(depth_m) / camera_intrinsics.fy
    return x_m, y_m, float(depth_m)
