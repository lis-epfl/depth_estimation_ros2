import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('depth_estimation_ros2')

    params_file = os.path.join(pkg_dir, 'config', 'depth_omninxt_params.yaml')

    depth_estimation_node = Node(
        package='depth_estimation_ros2',
        executable='depth_estimation_node',
        name='depth_estimation_node',
        output='screen',
        emulate_tty=True,
        parameters=[params_file]
    )

    # --- ADDED NODE ---
    # Publishes a static transform from 'world' to 'base_link' at 0,0,0
    # static_tf_publisher = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_world_broadcaster',
    #     arguments=['0', '0', '0', '0', '0', '0', 'world', 'drone_centroid']
    # )

    return LaunchDescription([
        # static_tf_publisher,  # Add the new node to the launch description
        depth_estimation_node
    ])
