from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare an argument for the DAE file
    dae_file_arg = DeclareLaunchArgument(
        'dae_file',
        default_value='/home/aggelosubuntu/ur_ws/src/robot_control/models/mass_object.dae',  # Specify a default DAE file
        description='Path to the DAE file to visualize'
    )

    # Declare an argument for the frame_id (which was missing)
    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='map',  # You can set a default frame_id
        description='The frame_id for the PointCloud2 message'
    )

    # Get the value of the DAE file and frame_id from the command line
    dae_file = LaunchConfiguration('dae_file')
    frame_id = LaunchConfiguration('frame_id')

    # Node to view the mesh (update this based on your setup)
    view_mesh_node = Node(
        package='robot_control',
        executable='publish_point_cloud',  
        name='point_cloud_publisher_node',
        output='log',
        parameters=[{
            'dae_file': dae_file  # Pass dae_file as a parameter
           
        }]
    )
 
    # RViz node (you had this, but it wasn't included in LaunchDescription)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
    )

    # Return a launch description with both the mesh viewer and RViz nodes
    return LaunchDescription([
        dae_file_arg,
        frame_id_arg,
        view_mesh_node,
        rviz_node
    ])
