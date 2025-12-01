import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('depth_estimation_ros2')
    params_file = os.path.join(pkg_dir, 'config', 'depth_omninxt_params.yaml')

    # 1. Declare the argument
    frame_id_arg = DeclareLaunchArgument(
        'pointcloud_frame_id',
        default_value='drone_centroid', # Default fallback
        description='The frame ID to use for the published point cloud'
    )

    depth_estimation_node = Node(
        package='depth_estimation_ros2',
        executable='depth_estimation_node',
        name='depth_estimation_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            params_file,
            # 2. Override the parameter with the launch configuration
            {'pointcloud_frame_id': LaunchConfiguration('pointcloud_frame_id')}
        ]
    )

    return LaunchDescription([
        frame_id_arg,
        depth_estimation_node
    ])
