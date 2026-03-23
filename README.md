# vision

This repository now contains a ROS 2 Python package at `vision_bag_detector/` for offline detection on bagged Intel RealSense data.

The package provides:

- A parameterized ROS 2 node built with `rclpy`
- A clean separation between bag reading, YOLO inference, depth handling, and message construction
- Support for both `rosbag2` inputs and native RealSense `.bag` playback
- Detection publishing via `vision_msgs/msg/Detection2DArray`

## Package Layout

- `vision_bag_detector/vision_bag_detector/bag_detection_node.py`: main ROS 2 node
- `vision_bag_detector/vision_bag_detector/bag_reader.py`: rosbag2 and RealSense bag readers
- `vision_bag_detector/vision_bag_detector/yolo_detector.py`: YOLO wrapper
- `vision_bag_detector/vision_bag_detector/depth_utils.py`: depth extraction helpers
- `vision_bag_detector/vision_bag_detector/message_utils.py`: `vision_msgs` conversion
- `vision_bag_detector/config/bag_yolo_detector.params.yaml`: starter parameters
- `vision_bag_detector/launch/bag_yolo_detector.launch.py`: launch entry point

## Dependencies

Install the ROS 2 dependencies with your normal workspace tooling, then install the model/runtime extras you need:

```bash
pip install ultralytics
```

If you want to read native RealSense `.bag` files directly, also install Intel RealSense Python bindings:

```bash
pip install pyrealsense2
```

## Example

```bash
colcon build --packages-select vision_bag_detector
source install/setup.bash
ros2 launch vision_bag_detector bag_yolo_detector.launch.py \
  bag_path:=/path/to/bag \
  yolo_weights_path:=/path/to/yolo.pt
```
