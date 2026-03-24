from __future__ import annotations

import cv2
import numpy as np


def convert_to_bgr(image: np.ndarray, encoding: str) -> tuple[np.ndarray, str]:
    normalized_encoding = encoding.lower()
    if normalized_encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), "bgr8"
    if normalized_encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR), "bgr8"
    if normalized_encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR), "bgr8"
    if normalized_encoding == "bgr8":
        return image, "bgr8"
    return image, encoding

def convert_to_rgb(image: np.ndarray, encoding: str) -> tuple[np.ndarray, str]:
    normalized_encoding = encoding.lower()
    if normalized_encoding == "bgr8":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "rgb8"
    if normalized_encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB), "rgb8"
    if normalized_encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB), "rgb8"
    if normalized_encoding == "rgb8":
        return image, "rgb8"
    return image, encoding


def flip_if_needed(image: np.ndarray, flip_image: bool) -> np.ndarray:
    if not flip_image:
        return image
    return cv2.rotate(image, cv2.ROTATE_180)
