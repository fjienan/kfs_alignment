"""Launch kfs_alignment nodes with a params file.

This launch file starts the vision node, the chassis control node, and RViz.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # 1. 包名修改为新仓库名 kfs_alignment
    pkg_dir = get_package_share_directory("kfs_alignment")
    
    # 假设你的配置文件名没变，依然放在 config 目录下
    default_params = os.path.join(pkg_dir, "config", "cube_pose_estimator.yaml")
    default_rviz = os.path.join(pkg_dir, "config", "cube_pose_estimator.rviz")

    params_file = LaunchConfiguration("params_file")
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    use_grayscale = LaunchConfiguration("use_grayscale")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Path to the ROS2 parameters YAML file.",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Whether to start RViz2 with a pre-configured layout.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz,
                description="Path to an RViz2 config file.",
            ),
            DeclareLaunchArgument(
                "use_grayscale",
                default_value="false",
                description="Whether to capture/process images in grayscale mode.",
            ),
            
            # --- 1. 视觉感知节点 ---
            Node(
                package="kfs_alignment", # 改为新包名
                executable="cube_pose_node",
                name="cube_pose_estimator",
                output="screen",
                parameters=[
                    params_file,
                    {"use_grayscale": use_grayscale}
                ],
            ),
            
            # --- 2. 【新增】底盘对齐控制节点 ---
            Node(
                package="kfs_alignment", # 改为新包名
                executable="kfs_align_controller_node", # 对应 setup.py 里的名字
                name="kfs_align_controller",
                output="screen",
                # 同样加载这个 yaml 文件，方便你在 yaml 里统一管理 PID 和速度限制
                parameters=[
                    params_file 
                ],
            ),

            # --- 3. RViz 可视化节点 ---
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(rviz),
            ),
        ]
    )