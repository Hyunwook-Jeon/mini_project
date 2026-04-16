"""
pick_place.launch.py
--------------------
전체 Pick & Place 시스템 런치:
  1. Doosan E0509 로봇 (MoveIt 포함)
  2. RealSense 카메라
  3. 카메라→로봇 정적 TF
  4. 객체 검출 노드
  5. Pick & Place 상태머신 노드
  6. GUI (선택)

사용법:
  # 가상 모드 (에뮬레이터)
  ros2 launch dsr_realsense_pick_place pick_place.launch.py mode:=virtual

  # 실제 로봇
  ros2 launch dsr_realsense_pick_place pick_place.launch.py mode:=real host:=192.168.1.100

  # RealSense 없이 테스트 (카메라 노드 스킵)
  ros2 launch dsr_realsense_pick_place pick_place.launch.py use_realsense:=false
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


# ── 런치 인수 ──────────────────────────────────────────────────────────
ARGUMENTS = [
    # 로봇
    DeclareLaunchArgument('mode',  default_value='virtual',
                          description='virtual | real'),
    DeclareLaunchArgument('host',  default_value='127.0.0.1',
                          description='로봇 IP (real 모드)'),
    DeclareLaunchArgument('port',  default_value='12345',
                          description='로봇 포트'),
    DeclareLaunchArgument('model', default_value='e0509',
                          description='Doosan 모델명'),
    DeclareLaunchArgument('color', default_value='white',
                          description='로봇 색상'),
    # 카메라
    DeclareLaunchArgument('use_realsense', default_value='true',
                          description='RealSense 카메라 노드 실행 여부'),
    DeclareLaunchArgument('camera_serial', default_value='',
                          description='RealSense 시리얼 번호 (비어있으면 자동)'),
    # TF: 카메라→로봇 베이스 (hand-eye 캘리브레이션 결과 입력)
    # 아래 값은 예시 (실제 측정값으로 교체 필요!)
    # 단위: translation=m, rotation=quaternion (x y z w)
    DeclareLaunchArgument('cam_tf_x', default_value='0.5'),
    DeclareLaunchArgument('cam_tf_y', default_value='0.0'),
    DeclareLaunchArgument('cam_tf_z', default_value='0.6'),
    DeclareLaunchArgument('cam_tf_qx', default_value='0.0'),
    DeclareLaunchArgument('cam_tf_qy', default_value='0.707'),
    DeclareLaunchArgument('cam_tf_qz', default_value='0.0'),
    DeclareLaunchArgument('cam_tf_qw', default_value='0.707'),
    # 기타
    DeclareLaunchArgument('gui', default_value='true',
                          description='PyQt GUI 실행 여부'),
]


def generate_launch_description():

    # 설치된 패키지의 share 디렉터리에서 설정 파일 경로를 찾는다.
    pkg_this = get_package_share_directory('dsr_realsense_pick_place')
    params_file = os.path.join(pkg_this, 'config', 'pick_place_params.yaml')

    # ── 1. Doosan 로봇 bringup (MoveIt 포함) ───────────────────────────
    doosan_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('dsr_bringup2'),
            '/launch/dsr_bringup2_moveit.launch.py'
        ]),
        launch_arguments={
            # 여기서 넘긴 값이 dsr_bringup2 쪽 launch 인수로 그대로 전달된다.
            'mode':  LaunchConfiguration('mode'),
            'host':  LaunchConfiguration('host'),
            'port':  LaunchConfiguration('port'),
            'model': LaunchConfiguration('model'),
            'color': LaunchConfiguration('color'),
            'name':  'dsr01',
        }.items(),
    )

    # ── 2. RealSense 카메라 ────────────────────────────────────────────
    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'),
            '/launch/rs_launch.py'
        ]),
        launch_arguments={
            # 검출은 컬러 이미지 중심으로 하므로 depth를 color 프레임에 맞춘다.
            'align_depth.enable': 'true',
            'pointcloud.enable':  'true',
            'serial_no':          LaunchConfiguration('camera_serial'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_realsense')),
    )

    # ── 3. 카메라 → 로봇 베이스 정적 TF ──────────────────────────────
    # eye-to-hand 구성 (외부 고정 카메라):
    #   base_link  ← [static TF] ←  camera_color_optical_frame
    #
    # ※ 실제 값은 hand-eye 캘리브레이션 후 교체:
    #   ros2 run easy_handeye2 calibrate ...
    #   또는 직접 줄자로 측정
    static_tf_cam_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_to_base_tf',
        arguments=[
            # x y z qx qy qz qw  parent_frame  child_frame
            LaunchConfiguration('cam_tf_x'),
            LaunchConfiguration('cam_tf_y'),
            LaunchConfiguration('cam_tf_z'),
            LaunchConfiguration('cam_tf_qx'),
            LaunchConfiguration('cam_tf_qy'),
            LaunchConfiguration('cam_tf_qz'),
            LaunchConfiguration('cam_tf_qw'),
            'base_link',
            'camera_color_optical_frame',
        ],
        output='screen',
    )

    # ── 4. 객체 검출 노드 ──────────────────────────────────────────────
    object_detector = Node(
        package='dsr_realsense_pick_place',
        executable='object_detector',
        name='object_detector',
        output='screen',
        parameters=[params_file],
    )

    # GUI만 ROS 디버그 영상(yaml 기본은 로컬 best.pt — 여기서 false 로 덮어씀).
    gui_node = Node(
        package='dsr_realsense_pick_place',
        executable='gui_node',
        name='pick_place_gui',
        output='screen',
        parameters=[
            params_file,
            {'use_local_yolo': False},
        ],
        condition=IfCondition(LaunchConfiguration('gui')),
    )


    # ── 5. Gripper 노드 (로봇 bringup 후 5초 지연 시작) ──────────────────
    gripper = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='dsr_realsense_pick_place',
                executable='gripper_node',
                name='rh_p12_rna_gripper',
                output='screen',
                parameters=[
                    params_file,
                    {'robot_ns':''}
                ]
            )
        ]
    )

    # ── 5. Pick & Place 노드 (로봇 bringup 후 5초 지연 시작) ──────────
    pick_place = TimerAction(
        period=10.0,
        # 로봇 서비스가 올라오기 전에 상태머신이 먼저 시작되지 않도록 잠시 대기한다.
        actions=[
            Node(
                package='dsr_realsense_pick_place',
                executable='pick_place_node',
                name='pick_place_node',
                output='screen',
                parameters=[params_file],
            )
        ]
    )

    # 최종 LaunchDescription 에 각 액션을 순서대로 담아 반환한다.
    return LaunchDescription(ARGUMENTS + [
        doosan_moveit,
        realsense_node,
        static_tf_cam_to_base,
        object_detector,
        gui_node,
        gripper,
        pick_place,
    ])
