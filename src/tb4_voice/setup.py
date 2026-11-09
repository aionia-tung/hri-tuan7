from glob import glob

from setuptools import setup

package_name = 'tb4_voice'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/config', glob('config/*.yaml') + glob('config/*.rviz')),
        (f'share/{package_name}/maps', glob('maps/*')),
        (f'share/{package_name}/worlds', glob('worlds/*.sdf')),
        (f'share/{package_name}/test_audio', glob('test_audio/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aionia-tung',
    maintainer_email='tst159205@gmail.com',
    description='Map segmentation + Vietnamese voice navigation for TurtleBot 4 (Nav2).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segment_map = tb4_voice.segment_map:main',
            'speech_node = tb4_voice.speech_node:main',
            'voice_nav_node = tb4_voice.voice_nav_node:main',
        ],
    },
)
