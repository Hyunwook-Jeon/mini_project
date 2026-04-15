"""
pick_place.launch.py
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


ARGUMENTS = [
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
    DeclareLaunchArgument('use_realsense', default_value='true',
                          description='RealSense 카메라 노드 실행 여부'),
    DeclareLaunchArgument('camera_serial', default_value='',
                          description='RealSense 시리얼 번호'),
    DeclareLaunchArgument('cam_tf_x',  default_value='0.5'),
    DeclareLaunchArgument('cam_tf_y',  default_value='0.0'),
    DeclareLaunchArgument('cam_tf_z',  default_value='0.6'),
    DeclareLaunchArgument('cam_tf_qx', default_value='0.0'),
    DeclareLaunchArgument('cam_tf_qy', default_value='0.707'),
    DeclareLaunchArgument('cam_tf_qz', default_value='0.0'),
    DeclareLaunchArgument('cam_tf_qw', default_value='0.707'),
    DeclareLaunchArgument('gui', default_value='true',
                          description='PyQt GUI 실행 여부'),
]


def generate_launch_description():

    pkg_this    = get_package_share_directory('dsr_realsense_pick_place')
    params_file = os.path.join(pkg_this, 'config', 'pick_place_params.yaml')

    # ── 1. Doosan bringup (rviz 버전) ──────────────────────────────────
    # moveit 버전은 내부 컨트롤러가 달라 move_joint success=False 발생
    doosan_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('dsr_bringup2'),
            '/launch/dsr_bringup2_rviz.launch.py'   # ← moveit → rviz
        ]),
        launch_arguments={
            'mode':  LaunchConfiguration('mode'),
            'host':  LaunchConfiguration('host'),
            'port':  LaunchConfiguration('port'),
            'model': LaunchConfiguration('model'),
            'color': LaunchConfiguration('color'),
            'name':  'dsr01',
        }.items(),
    )

    # ── 2. robot_mode=1 설정 (bringup 6초 후) ──────────────────────────
    # pick_place_node 내부에서도 설정하지만, launch 단계에서도 보장
    set_robot_mode = TimerAction(
        period=10.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'service', 'call',
                    '/dsr01/system/set_robot_mode',
                    'dsr_msgs2/srv/SetRobotMode',
                    '{robot_mode: 1}',
                ],
                output='screen',
            )
        ]
    )

    # ── 3. RealSense 카메라 ─────────────────────────────────────────────
    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'),
            '/launch/rs_launch.py'
        ]),
        launch_arguments={
            'align_depth.enable': 'true',
            'pointcloud.enable':  'true',
            'serial_no':          LaunchConfiguration('camera_serial'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_realsense')),
    )

    # ── 4. 카메라 → 베이스 정적 TF ─────────────────────────────────────
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_to_base_tf',
        arguments=[
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

    # ── 5. 객체 검출 노드 ───────────────────────────────────────────────
    object_detector = Node(
        package='dsr_realsense_pick_place',
        executable='object_detector',
        name='object_detector',
        output='screen',
        parameters=[params_file],
    )

    # ── 6. GUI 노드 ─────────────────────────────────────────────────────
    gui_node = Node(
        package='dsr_realsense_pick_place',
        executable='gui_node',
        name='pick_place_gui',
        output='screen',
        parameters=[params_file],
        condition=IfCondition(LaunchConfiguration('gui')),
    )

    # ── 7. Gripper 노드 (5초 후) ────────────────────────────────────────
    gripper = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='dsr_realsense_pick_place',
                executable='gripper_node',
                name='rh_p12_rna_gripper',
                output='screen',
                parameters=[params_file, 
                            {'robot_ns': 'dsr01'}],
            )
        ]
    )

    # ── 8. Pick & Place 노드 (12초 후) ─────────────────────────────────
    # 타이밍: 0초 bringup → 6초 set_robot_mode → 12초 pick_place_node 시작
    # pick_place_node 내부 _wait_for_services()에서 추가로 3초 대기 후 재확인
    pick_place = TimerAction(
        period=12.0,
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

    return LaunchDescription(ARGUMENTS + [
        doosan_bringup,       # 0초: 로봇 드라이버 + rviz
        set_robot_mode,       # 6초: robot_mode=1
        realsense_node,       # 즉시: 카메라
        static_tf,            # 즉시: TF
        object_detector,      # 즉시: YOLO 검출
        gui_node,             # 즉시: GUI (조건부)
        gripper,              # 5초: 그리퍼 노드
        pick_place,           # 12초: 상태머신
    ])