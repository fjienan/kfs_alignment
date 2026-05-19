import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'kfs_alignment'

setup(
    name=package_name,
    version='0.0.0',
    # 使用 find_packages 自动查找 Python 源码目录，无需手动列出
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # 1. 打包 config 文件夹里的所有文件 (.yaml, .rviz 等)
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.*'))),
        
        # 2. 打包 launch 文件夹里的所有启动文件
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        
        # 3. 【核心修复】分别打包 obb 和 pose 子文件夹里的模型文件，并且只抓取文件，忽略文件夹！
        (os.path.join('share', package_name, 'models', 'obb'), glob(os.path.join('models', 'obb', '*.pt'))),
        (os.path.join('share', package_name, 'models', 'pose'), glob(os.path.join('models', 'pose', '*.pt'))),
        
        # 4. 如果你单独建了 rviz 文件夹放配置文件，也会自动打包
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fjienan',
    maintainer_email='fjienan@example.com',
    description='KFS Alignment: Cube face corner detection, pose estimation, and chassis control',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cube_pose_node = kfs_alignment.cube_pose_node:main',
            'pose_fusion_node = kfs_alignment.pose_fusion_node:main',
            'kfs_align_controller_node = kfs_alignment.kfs_align_controller_node:main',
        ],
    },
)