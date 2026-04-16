from setuptools import find_packages, setup
import os
from glob import glob

# ROS 2 패키지 이름. 여러 곳에서 반복 사용하므로 상수로 분리한다.
package_name = 'dsr_realsense_pick_place'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    # find_packages()는 dsr_realsense_pick_place 디렉터리만 패키지로 찾는다.
    # 패키지 바깥의 .py 파일들(pca_mask_utils 등)은 py_modules로 직접 지정해야
    # colcon build 시 install 디렉터리에 함께 설치되어 import 할 수 있다.
    py_modules=[
        'pca_mask_utils',
        'vision_display_utils',
        'yolo_check',
        'yolo_live_cam',
        'yolo_live_cam_3d_metrics',
        'prepare_yolo_dataset',
        'train_yolo',
    ],
    data_files=[
        # ament 인덱스와 share 디렉터리에 패키지 메타데이터를 설치한다.
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch, config 파일도 설치본에서 바로 찾을 수 있도록 함께 복사한다.
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'ultralytics',
        'PyQt5',
        'pyrealsense2',
    ],
    zip_safe=True,
    maintainer='hyunwook',
    maintainer_email='hyunwook@todo.com',
    description='Doosan E0509 pick and place with RealSense, YOLO, and GUI object selection',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ros2 run 으로 실행할 엔트리포인트 등록
            'object_detector = dsr_realsense_pick_place.object_detector:main',
            'pick_place_node = dsr_realsense_pick_place.pick_place_node:main',
            'gui_node = dsr_realsense_pick_place.gui_node:main',
            'gripper_node = dsr_realsense_pick_place.gripper_node:main',
        ],
    },
)
