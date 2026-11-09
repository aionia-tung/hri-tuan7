"""Bai 07: Gazebo (maze + vat the) + TurtleBot 4 + map_server/AMCL + Nav2 + RViz2 (khu vuc).

Ke thua launch bai 05 (tb4_maps/tb4_week5_sim.launch.py).

Tuong duong turtlebot4_ignition.launch.py localization:=true nav2:=true, nhung:
  - them thu muc worlds/ cua package vao IGN_GAZEBO_RESOURCE_PATH de nap world
    maze_objects.sdf (launch goc ghi de bien nay nen khong truyen world rieng duoc);
  - sua loi GPU-lidar: Sensors system cua Create 3 dung render engine Ogre1, tren GPU
    Intel/Mesa moi tia /scan = range_min (0.164 m). Launch tao overlay
    irobot_create_description doi sang ogre2 va dat no truoc trong AMENT_PREFIX_PATH;
  - AMCL tu dat initial pose = pose spawn (set_initial_pose), khong can 2D Pose Estimate;
  - Nav2 dung tham so goc cua TurtleBot 4 nhung cho phep chinh van toc (max_speed, max_turn);
  - gui:=false chay Gazebo khong cua so de mo phong nhanh hon;
  - mo RViz2 voi config hien khu vuc phan vung, nhan, diem dich va duong du kien.
"""

import os
import shutil
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CACHE = Path.home() / '.cache' / 'tb4_voice'


def make_ogre2_overlay():
    """Ban sao irobot_create_description voi <render_engine>ogre2</render_engine>."""
    source = Path(get_package_share_directory('irobot_create_description'))
    prefix = CACHE / 'ogre2_overlay'
    target = prefix / 'share' / 'irobot_create_description'
    if source == target:
        return  # overlay da co trong AMENT_PREFIX_PATH
    if not target.exists():
        shutil.copytree(source, target)
    xacro = target / 'urdf' / 'create3.urdf.xacro'
    text = (source / 'urdf' / 'create3.urdf.xacro').read_text()
    xacro.write_text(text.replace('<render_engine>ogre</render_engine>',
                                  '<render_engine>ogre2</render_engine>'))
    index = prefix / 'share' / 'ament_index' / 'resource_index' / 'packages'
    index.mkdir(parents=True, exist_ok=True)
    (index / 'irobot_create_description').touch()
    # Dat overlay len truoc de get_package_share_directory/xacro tim thay ban nay
    os.environ['AMENT_PREFIX_PATH'] = f'{prefix}:{os.environ.get("AMENT_PREFIX_PATH", "")}'


def launch_setup(context):
    value = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    if value('lidar_ogre2') == 'true':
        make_ogre2_overlay()

    pkg_tb4_voice = get_package_share_directory('tb4_voice')
    pkg_tb4_ign = get_package_share_directory('turtlebot4_ignition_bringup')
    pkg_tb4_gui_plugins = get_package_share_directory('turtlebot4_ignition_gui_plugins')
    pkg_tb4_description = get_package_share_directory('turtlebot4_description')
    pkg_tb4_navigation = get_package_share_directory('turtlebot4_navigation')
    pkg_create_description = get_package_share_directory('irobot_create_description')
    pkg_create_ign = get_package_share_directory('irobot_create_ignition_bringup')
    pkg_create_ign_plugins = get_package_share_directory('irobot_create_ignition_plugins')
    pkg_ros_ign_gazebo = get_package_share_directory('ros_ign_gazebo')

    # Tham so AMCL goc cua TurtleBot 4 + initial pose = pose spawn (map trung world)
    params = yaml.safe_load(Path(pkg_tb4_navigation, 'config', 'localization.yaml').read_text())
    amcl = params['amcl']['ros__parameters']
    amcl['set_initial_pose'] = True
    amcl['initial_pose'] = {'x': float(value('x')), 'y': float(value('y')), 'z': 0.0,
                            'yaw': float(value('yaw'))}
    CACHE.mkdir(parents=True, exist_ok=True)
    localization_params = CACHE / 'localization.yaml'
    localization_params.write_text(yaml.safe_dump(params))

    # Tham so Nav2 goc (0.26 m/s, 1.0 rad/s) + van toc tuy chinh. Create 3 gioi han ~0.306 m/s.
    speed, turn = float(value('max_speed')), float(value('max_turn'))
    nav2 = yaml.safe_load(Path(pkg_tb4_navigation, 'config', 'nav2.yaml').read_text())
    dwb = nav2['controller_server']['ros__parameters']['FollowPath']
    dwb.update(max_vel_x=speed, max_speed_xy=speed, max_vel_theta=turn)
    smoother = nav2['velocity_smoother']['ros__parameters']
    smoother['max_velocity'] = [speed, 0.0, turn]
    smoother['min_velocity'] = [-speed, 0.0, -turn]
    nav2_params = CACHE / 'nav2.yaml'
    nav2_params.write_text(yaml.safe_dump(nav2))

    ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=':'.join([
            os.path.join(pkg_tb4_voice, 'worlds'),
            os.path.join(pkg_tb4_ign, 'worlds'),
            os.path.join(pkg_create_ign, 'worlds'),
            str(Path(pkg_tb4_description).parent.resolve()),
            str(Path(pkg_create_description).parent.resolve()),
        ]))
    ign_gui_plugin_path = SetEnvironmentVariable(
        name='IGN_GUI_PLUGIN_PATH',
        value=':'.join([
            os.path.join(pkg_tb4_gui_plugins, 'lib'),
            os.path.join(pkg_create_ign_plugins, 'lib'),
        ]))

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_ign_gazebo, 'launch', 'ign_gazebo.launch.py')),
        launch_arguments=[
            ('ign_args', ' '.join([value('world') + '.sdf', '-r', '-v 4', '--gui-config',
                                   os.path.join(pkg_tb4_ign, 'gui', value('model'), 'gui.config')]
                                  + ([] if value('gui') == 'true' else ['-s'])))
        ])

    clock_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='clock_bridge',
        output='screen',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'])

    # Spawn robot + dock, bridge cam bien, map_server + AMCL (localization) va Nav2.
    # Cac LaunchConfiguration (world, map, model, x, y, yaw...) duoc truyen xuong launch con.
    robot_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb4_ign, 'launch', 'turtlebot4_spawn.launch.py')),
        launch_arguments=[
            ('rviz', 'false'),
            ('localization', 'true'),
            ('nav2', 'true'),
            ('params', str(localization_params)),
            ('params_file', str(nav2_params)),
        ])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        arguments=['-d', os.path.join(pkg_tb4_voice, 'config', 'voice_nav.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')))

    return [ign_resource_path, ign_gui_plugin_path, gazebo, clock_bridge, robot_spawn, rviz]


def generate_launch_description():
    pkg_tb4_voice = get_package_share_directory('tb4_voice')
    arguments = [
        DeclareLaunchArgument('world', default_value='maze_objects',
                              description='Ten world (maze_objects, maze, depot, warehouse)'),
        DeclareLaunchArgument('map', default_value=os.path.join(pkg_tb4_voice, 'maps', 'map.yaml'),
                              description='Ban do maze (tu bai thuc hanh 04)'),
        DeclareLaunchArgument('model', default_value='standard', choices=['standard', 'lite']),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true', choices=['true', 'false'],
                              description='Mo cua so Gazebo (false = chi chay server, nhanh hon)'),
        DeclareLaunchArgument('max_speed', default_value='0.3', description='Van toc thang toi da (m/s)'),
        DeclareLaunchArgument('max_turn', default_value='1.9', description='Van toc quay toi da (rad/s)'),
        DeclareLaunchArgument('lidar_ogre2', default_value='true', choices=['true', 'false'],
                              description='Render cam bien bang ogre2 (sua loi /scan = range_min)'),
    ]
    for pose_element in ['x', 'y', 'z', 'yaw']:
        arguments.append(DeclareLaunchArgument(pose_element, default_value='0.0'))
    return LaunchDescription(arguments + [OpaqueFunction(function=launch_setup)])
