from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    frame_id: str


@dataclass
class ImageFrame:
    stamp: Time
    stamp_ns: int
    frame_id: str
    image: np.ndarray
    encoding: str
    depth_scale_meters: Optional[float] = None


@dataclass
class FramePair:
    color: ImageFrame
    depth: Optional[ImageFrame]
    camera_intrinsics: Optional[CameraIntrinsics] = None


class Rosbag2ImagePairReader:
    def __init__(
        self,
        bag_path: str,
        storage_id: str,
        color_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        sync_tolerance_sec: float,
        logger: RcutilsLogger,
    ) -> None:
        self._bag_path = bag_path
        self._storage_id = storage_id
        self._color_topic = color_topic
        self._depth_topic = depth_topic
        self._camera_info_topic = camera_info_topic
        self._sync_tolerance_ns = int(sync_tolerance_sec * 1_000_000_000)
        self._logger = logger
        self._bridge = CvBridge()
        self._pending_colors: Deque[ImageFrame] = deque()
        self._pending_depths: List[ImageFrame] = []
        self._reader_exhausted = False
        self._type_map: Dict[str, object] = {}
        self._latest_camera_intrinsics: Optional[CameraIntrinsics] = None
        self._reader: Optional[rosbag2_py.SequentialReader] = None
        self._open()

    def __iter__(self) -> "Rosbag2ImagePairReader":
        return self

    def __next__(self) -> FramePair:
        while True:
            pending_pair = self._try_make_pair()
            if pending_pair is not None:
                return pending_pair

            if self._reader_exhausted:
                if self._pending_colors:
                    color_frame = self._pending_colors.popleft()
                    self._logger.warning(
                        f"No depth frame matched color frame at {color_frame.stamp_ns} ns."
                    )
                    return FramePair(
                        color=color_frame,
                        depth=None,
                        camera_intrinsics=self._latest_camera_intrinsics,
                    )
                self.restart()
                continue

            if self._reader is None:
                raise RuntimeError("rosbag2 reader is not open.")

            if not self._reader.has_next():
                self._reader_exhausted = True
                continue

            topic_name, serialized_message, _ = self._reader.read_next()
            valid_topics = {self._color_topic, self._depth_topic}
            if self._camera_info_topic:
                valid_topics.add(self._camera_info_topic)
            if topic_name not in valid_topics:
                continue

            message_type = self._type_map.get(topic_name)
            if message_type is None:
                continue

            message = deserialize_message(serialized_message, message_type)
            if topic_name == self._camera_info_topic:
                if isinstance(message, CameraInfo):
                    self._latest_camera_intrinsics = self._convert_camera_info_message(message)
                continue

            if not isinstance(message, Image):
                continue

            frame = self._convert_image_message(topic_name, message)
            if topic_name == self._color_topic:
                self._pending_colors.append(frame)
            else:
                self._pending_depths.append(frame)

    def _open(self) -> None:
        self._pending_colors.clear()
        self._pending_depths.clear()
        self._reader_exhausted = False
        self._type_map.clear()
        self._latest_camera_intrinsics = None
        self._reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self._bag_path, storage_id=self._storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        self._reader.open(storage_options, converter_options)

        available_topics = {}
        for topic_metadata in self._reader.get_all_topics_and_types():
            available_topics[topic_metadata.name] = topic_metadata.type
            self._type_map[topic_metadata.name] = get_message(topic_metadata.type)

        if self._color_topic not in available_topics:
            raise ValueError(
                f"Color topic '{self._color_topic}' not found in bag. Available topics: {sorted(available_topics)}"
            )
        if self._depth_topic not in available_topics:
            raise ValueError(
                f"Depth topic '{self._depth_topic}' not found in bag. Available topics: {sorted(available_topics)}"
            )
        if self._camera_info_topic and self._camera_info_topic not in available_topics:
            self._logger.warning(
                f"CameraInfo topic '{self._camera_info_topic}' not found in bag. "
                "3D detections will be skipped until intrinsics are available."
            )

        self._logger.info(
            f"Opened rosbag2 input '{self._bag_path}' with color topic '{self._color_topic}', "
            f"depth topic '{self._depth_topic}', and camera info topic '{self._camera_info_topic}'."
        )

    def close(self) -> None:
        self._pending_colors.clear()
        self._pending_depths.clear()
        self._type_map.clear()
        self._reader_exhausted = False
        self._latest_camera_intrinsics = None
        self._reader = None

    def restart(self) -> None:
        self._logger.info("Reached end of rosbag2 input. Restarting from the beginning.")
        self.close()
        self._open()

    def _convert_image_message(self, topic_name: str, message: Image) -> ImageFrame:
        if topic_name == self._color_topic:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            encoding = "bgr8"
        else:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            encoding = message.encoding

        stamp_ns = (message.header.stamp.sec * 1_000_000_000) + message.header.stamp.nanosec
        return ImageFrame(
            stamp=message.header.stamp,
            stamp_ns=stamp_ns,
            frame_id=message.header.frame_id,
            image=image,
            encoding=encoding,
        )

    def _convert_camera_info_message(self, message: CameraInfo) -> CameraIntrinsics:
        return CameraIntrinsics(
            fx=float(message.k[0]),
            fy=float(message.k[4]),
            cx=float(message.k[2]),
            cy=float(message.k[5]),
            frame_id=message.header.frame_id,
        )

    def _try_make_pair(self) -> Optional[FramePair]:
        if not self._pending_colors:
            return None

        color_frame = self._pending_colors[0]

        while self._pending_depths and (
            self._pending_depths[0].stamp_ns < color_frame.stamp_ns - self._sync_tolerance_ns
        ):
            dropped_depth = self._pending_depths.pop(0)
            self._logger.debug(
                f"Dropping stale depth frame at {dropped_depth.stamp_ns} ns while matching color {color_frame.stamp_ns} ns."
            )

        best_index = None
        best_delta = None
        for index, depth_frame in enumerate(self._pending_depths):
            delta = abs(depth_frame.stamp_ns - color_frame.stamp_ns)
            if best_delta is None or delta < best_delta:
                best_index = index
                best_delta = delta
            if depth_frame.stamp_ns > color_frame.stamp_ns and delta > self._sync_tolerance_ns:
                break

        if best_index is not None and best_delta is not None and best_delta <= self._sync_tolerance_ns:
            self._pending_colors.popleft()
            return FramePair(
                color=color_frame,
                depth=self._pending_depths.pop(best_index),
                camera_intrinsics=self._latest_camera_intrinsics,
            )

        if self._pending_depths:
            earliest_depth = self._pending_depths[0]
            if earliest_depth.stamp_ns > color_frame.stamp_ns + self._sync_tolerance_ns:
                self._pending_colors.popleft()
                self._logger.warning(
                    f"No synchronized depth frame found for color frame at {color_frame.stamp_ns} ns."
                )
                return FramePair(
                    color=color_frame,
                    depth=None,
                    camera_intrinsics=self._latest_camera_intrinsics,
                )

        return None


class RealSenseBagReader:
    def __init__(self, bag_path: str, logger: RcutilsLogger) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is required to read native RealSense .bag files. Install Intel RealSense Python bindings first."
            ) from exc

        self._logger = logger
        self._rs = rs
        self._bag_path = str(Path(bag_path))
        self._pipeline = None
        self._config = None
        self._profile = None
        self._playback = None
        self._align = None
        self._depth_scale_meters = None
        self._camera_intrinsics = None
        self._open()
        self._logger.info(f"Opened native RealSense bag '{bag_path}'.")

    def __iter__(self) -> "RealSenseBagReader":
        return self

    def __next__(self) -> FramePair:
        while True:
            if self._pipeline is None or self._playback is None or self._align is None:
                raise RuntimeError("RealSense reader is not open.")

            try:
                frames = self._pipeline.wait_for_frames(5000)
            except RuntimeError as exc:
                if self._playback.current_status() == self._rs.playback_status.stopped:
                    self.restart()
                    continue
                raise RuntimeError(f"Failed while reading RealSense frames: {exc}") from exc

            aligned_frames = self._align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame:
                continue

            color_stamp = self._frame_time(color_frame)
            color_image = np.asanyarray(color_frame.get_data()).copy()
            color = ImageFrame(
                stamp=color_stamp,
                stamp_ns=(color_stamp.sec * 1_000_000_000) + color_stamp.nanosec,
                frame_id="camera_color_optical_frame",
                image=color_image,
                encoding="bgr8",
            )

            depth = None
            if depth_frame:
                depth_stamp = self._frame_time(depth_frame)
                depth_image = np.asanyarray(depth_frame.get_data()).copy()
                depth = ImageFrame(
                    stamp=depth_stamp,
                    stamp_ns=(depth_stamp.sec * 1_000_000_000) + depth_stamp.nanosec,
                    frame_id="camera_depth_optical_frame",
                    image=depth_image,
                    encoding="16UC1" if depth_image.dtype == np.uint16 else "32FC1",
                    depth_scale_meters=self._depth_scale_meters,
                )

            return FramePair(
                color=color,
                depth=depth,
                camera_intrinsics=self._camera_intrinsics,
            )

    def _open(self) -> None:
        self._pipeline = self._rs.pipeline()
        self._config = self._rs.config()
        self._rs.config.enable_device_from_file(
            self._config, self._bag_path, repeat_playback=True
        )
        self._profile = self._pipeline.start(self._config)
        self._playback = self._profile.get_device().as_playback()
        self._playback.set_real_time(False)
        self._align = self._rs.align(self._rs.stream.color)
        self._depth_scale_meters = self._resolve_depth_scale()
        self._camera_intrinsics = self._resolve_color_intrinsics()

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
        self._pipeline = None
        self._config = None
        self._profile = None
        self._playback = None
        self._align = None
        self._camera_intrinsics = None

    def restart(self) -> None:
        self._logger.info("Reached end of RealSense bag playback. Restarting from the beginning.")
        self.close()
        self._open()

    def _frame_time(self, frame) -> Time:
        stamp_ns = int(frame.get_timestamp() * 1_000_000)
        sec, nanosec = divmod(stamp_ns, 1_000_000_000)
        return Time(sec=sec, nanosec=nanosec)

    def _resolve_depth_scale(self) -> Optional[float]:
        try:
            depth_sensor = self._profile.get_device().first_depth_sensor()
            return float(depth_sensor.get_depth_scale())
        except RuntimeError:
            self._logger.warning(
                "Could not query the RealSense depth scale. Falling back to the configured depth scale."
            )
            return None

    def _resolve_color_intrinsics(self) -> Optional[CameraIntrinsics]:
        try:
            color_stream = self._profile.get_stream(self._rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            return CameraIntrinsics(
                fx=float(intrinsics.fx),
                fy=float(intrinsics.fy),
                cx=float(intrinsics.ppx),
                cy=float(intrinsics.ppy),
                frame_id="camera_color_optical_frame",
            )
        except RuntimeError:
            self._logger.warning(
                "Could not query RealSense color intrinsics. 3D detections will be skipped."
            )
            return None


def create_frame_reader(
    bag_path: str,
    bag_reader_backend: str,
    bag_storage_id: str,
    color_topic: str,
    depth_topic: str,
    camera_info_topic: str,
    sync_tolerance_sec: float,
    logger: RcutilsLogger,
):
    resolved_backend = bag_reader_backend
    if bag_reader_backend == "auto":
        resolved_backend = "realsense" if Path(bag_path).suffix == ".bag" else "rosbag2"

    if resolved_backend == "realsense":
        logger.info("Selected native RealSense bag reader backend.")
        return RealSenseBagReader(bag_path=bag_path, logger=logger)

    logger.info("Selected rosbag2 reader backend.")
    return Rosbag2ImagePairReader(
        bag_path=bag_path,
        storage_id=bag_storage_id,
        color_topic=color_topic,
        depth_topic=depth_topic,
        camera_info_topic=camera_info_topic,
        sync_tolerance_sec=sync_tolerance_sec,
        logger=logger,
    )
