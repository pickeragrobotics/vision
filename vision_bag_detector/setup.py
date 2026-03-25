from setuptools import find_packages, setup


package_name = "vision_bag_detector"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/launch",
            [
                "launch/bag_yolo_detector.launch.py",
                "launch/from_camera_full_pipeline.launch.py",
            ],
        ),
        (
            f"share/{package_name}/config",
            [
                "config/bag_yolo_detector.params.yaml",
                "config/camera_yolo_detector.params.yaml",
            ],
        ),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Vision Team",
    maintainer_email="dev@example.com",
    description="Offline bag reader and YOLO detection publisher for ROS 2.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bag_yolo_detector = vision_bag_detector.bag_detection_node:main",
            "camera_yolo_detector = vision_bag_detector.camera_detection_node:main",
        ],
    },
)
