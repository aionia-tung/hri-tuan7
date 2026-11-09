"""Bai 07: node nhan dang giong noi + node dieu huong theo khu vuc.

Chay sau khi voice_sim.launch.py da khoi dong xong (AMCL + Nav2 san sang):
    ros2 launch tb4_voice voice_nav.launch.py                 # micro
    ros2 launch tb4_voice voice_nav.launch.py source:=files   # phat test_audio/demo_*.mp3
    ros2 launch tb4_voice voice_nav.launch.py speech:=false   # chi go lenh vao /voice/text
"""

import os
from glob import glob

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def speech_node(context):
    value = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    pkg = get_package_share_directory('tb4_voice')
    files = sorted(glob(os.path.join(pkg, 'test_audio', value('demo_glob')))) or ['']
    return [Node(
        package='tb4_voice', executable='speech_node', name='speech_recognition', output='screen',
        parameters=[os.path.join(pkg, 'config', 'voice_nav.yaml'), {
            'use_sim_time': True,
            'source': value('source'),
            'device': value('device'),
            'use_grammar': value('grammar') == 'true',
            'wav_files': files,
        }],
        condition=IfCondition(value('speech')))]


def generate_launch_description():
    pkg = get_package_share_directory('tb4_voice')
    arguments = [
        DeclareLaunchArgument('speech', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('source', default_value='mic', choices=['mic', 'files']),
        DeclareLaunchArgument('device', default_value='default', description='thiet bi ALSA'),
        DeclareLaunchArgument('grammar', default_value='true', choices=['true', 'false'],
                              description='gioi han tu vung Vosk vao lenh + ten khu vuc'),
        DeclareLaunchArgument('demo_glob', default_value='demo_*.mp3',
                              description='file trong test_audio/ phat khi source:=files'),
        DeclareLaunchArgument('speak', default_value='false', choices=['true', 'false'],
                              description='doc phan hoi bang spd-say'),
    ]
    navigator = Node(
        package='tb4_voice', executable='voice_nav_node', name='voice_nav', output='screen',
        parameters=[os.path.join(pkg, 'config', 'voice_nav.yaml'), {
            'use_sim_time': True,
            'speak': LaunchConfiguration('speak'),
        }])
    return LaunchDescription(arguments + [navigator, OpaqueFunction(function=speech_node)])
