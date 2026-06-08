# RealSense Pick & Place 전체 노드와 그리퍼 TCP 브릿지를 실행하는 launch 파일

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

    DeclareLaunchArgument('host',  default_value='110.120.1.50',

                          description='로봇 IP (real 모드)'),
    DeclareLaunchArgument('port',  default_value='12345',
                          description='로봇 포트'),
    DeclareLaunchArgument('model', default_value='e0509',
                          description='Doosan 모델명'),
    DeclareLaunchArgument('color', default_value='white',
                          description='로봇 색상'),
    DeclareLaunchArgument('robot_name', default_value='dsr01',
                          description='Doosan ROS namespace/name'),
    DeclareLaunchArgument('robot_base_frame', default_value='base_link',
                          description='로봇 기준 base frame'),
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
    DeclareLaunchArgument('use_launch_set_robot_mode', default_value='false',
                          description='Fallback: launch에서 set_robot_mode service call 실행 여부'),
    DeclareLaunchArgument('gripper_tcp_port', default_value='20002',
                          description='컨트롤러 DRL 그리퍼 TCP 서버 포트'),
]


def generate_launch_description():

    pkg_this = get_package_share_directory('dsr_realsense_pick_place')
    params_file = os.path.join(pkg_this, 'config', 'pick_place_params.yaml')

    doosan_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('dsr_bringup2'),
            '/launch/dsr_bringup2_rviz.launch.py'
        ]),
        launch_arguments={
            'mode':  LaunchConfiguration('mode'),
            'host':  LaunchConfiguration('host'),
            'port':  LaunchConfiguration('port'),
            'model': LaunchConfiguration('model'),
            'color': LaunchConfiguration('color'),
            'name':  LaunchConfiguration('robot_name'),
        }.items(),
    )

    set_robot_mode = TimerAction(
        period=10.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'service', 'call',
                    ["/", LaunchConfiguration('robot_name'), '/system/set_robot_mode'],
                    'dsr_msgs2/srv/SetRobotMode',
                    '{robot_mode: 1}',
                ],
                output='screen',
            )
        ],
        condition=IfCondition(LaunchConfiguration('use_launch_set_robot_mode')),
    )

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
            LaunchConfiguration('robot_base_frame'),
            'camera_color_optical_frame',
        ],
        output='screen',
    )

    object_detector = Node(
        package='dsr_realsense_pick_place',
        executable='object_detector',
        name='object_detector',
        output='screen',
        parameters=[params_file, {
            'robot_base_frame': LaunchConfiguration('robot_base_frame'),
        }],
    )

    gui_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='dsr_realsense_pick_place',
                executable='gui_node',
                name='pick_place_gui',
                output='screen',
                parameters=[params_file],
                condition=IfCondition(LaunchConfiguration('gui')),
            )
        ],
    )

    gripper = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='dsr_gripper_tcp',
                executable='gripper_service_node',
                name='gripper_service',
                output='screen',
                parameters=[{
                    'controller_host': LaunchConfiguration('host'),
                    'tcp_port': LaunchConfiguration('gripper_tcp_port'),
                    'namespace': LaunchConfiguration('robot_name'),
                    'goal_current': 400,
                    'profile_velocity': 1500,
                    'profile_acceleration': 1000,
                    'connect_timeout_sec': 60.0,
                    'post_drl_start_sleep_sec': 2.0,
                    'drl_idle_stable_sec': 2.0,
                    'tcp_server_open_retry_sec': 0.5,
                    # DRL이 시작 즉시 시리얼 포트를 강제 recycle하므로 첫 시도에 성공 가능성 높음.
                    # 그래도 cold-boot/motor stuck 대비 PC 재시도는 남겨둔다.
                    'init_attempts': 5,            # 첫 시도 깨끗하면 1회로 끝, 보험으로 5회
                    'init_timeout_sec': 20.0,      # DRL 내부 24×0.5s=12s + serial reset 여유 → 20s면 충분
                    'init_retry_delay_sec': 1.0,   # 빠른 재시도 (DRL 측 정리 시간 1초면 충분)
                }]
            ),
            Node(
                package='dsr_realsense_pick_place',
                executable='gripper_node',
                name='rh_p12_rna_gripper',
                output='screen',
                parameters=[params_file, {
                    'robot_ns': LaunchConfiguration('robot_name'),
                }],
            )
        ]
    )

    pick_place = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='dsr_realsense_pick_place',
                executable='pick_place_node',
                name='pick_place_node',
                output='screen',
                parameters=[params_file, {
                    'robot_namespace': LaunchConfiguration('robot_name'),
                    'robot_base_frame': LaunchConfiguration('robot_base_frame'),
                }],
            )
        ]
    )

    return LaunchDescription(ARGUMENTS + [
        doosan_bringup,
        set_robot_mode,
        realsense_node,
        static_tf,
        object_detector,
        gui_node,
        gripper,
        pick_place,
    ])
