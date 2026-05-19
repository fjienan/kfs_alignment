"""Launch cube_pose_estimator + pose_fusion_node.

This launches:
- cube_pose_node (vision corners -> PnP pose)
- pose_fusion_node (fuse PoseStamped with IMU)

Both nodes take their own params YAML.
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
    pkg_dir = get_package_share_directory("cube_pose_estimator")
    default_estimator_params = os.path.join(pkg_dir, "config", "cube_pose_estimator.yaml")
    default_fusion_params = os.path.join(pkg_dir, "config", "cube_pose_fusion.yaml")
    default_rviz = os.path.join(pkg_dir, "config", "cube_pose_estimator.rviz")

    estimator_params_file = LaunchConfiguration("estimator_params_file")
    fusion_params_file = LaunchConfiguration("fusion_params_file")
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "estimator_params_file",
                default_value=default_estimator_params,
                description="Path to cube_pose_node parameters YAML.",
            ),
            DeclareLaunchArgument(
                "fusion_params_file",
                default_value=default_fusion_params,
                description="Path to pose_fusion_node parameters YAML.",
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
            Node(
                package="cube_pose_estimator",
                executable="cube_pose_node",
                name="cube_pose_estimator",
                output="screen",
                parameters=[estimator_params_file],
            ),
            Node(
                package="cube_pose_estimator",
                executable="pose_fusion_node",
                name="cube_pose_fusion",
                output="screen",
                parameters=[fusion_params_file],
            ),
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

