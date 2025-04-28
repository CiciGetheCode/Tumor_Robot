import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from std_msgs.msg import Header
from xml.etree import ElementTree as ET

class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('point_cloud_publisher')

        # Declare default parameters
        self.declare_parameter('dae_file', '/home/aggelosubuntu/ur_ws/src/robot_control/models/mass_object.dae')
        self.declare_parameter('frame_id', 'map')
        self.declate_parameter('primal_axis' , axis = 0)

        # Get parameters
        self.dae_file= self.get_parameter('dae_file').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.primal_axis = self.get_parameter('primal_axis').get_parameter_value().int_value
	
        # Publisher for PointCloud2
        self.publisher = self.create_publisher(PointCloud2, 'mass_point_cloud', 10)
        self.timer = self.create_timer(1.0, self.publish_point_cloud)
	
	
	# Flag to ensure point cloud is published only once
        self.published_once = False
    def publish_point_cloud(self):
        if self.published_once:
            return  # Skip if already published

        try:
            tree = ET.parse(self.dae_file)
            root = tree.getroot()
            vertices = []

            # Look for the float_array element containing positions
            for source in root.findall(".//{http://www.collada.org/2005/11/COLLADASchema}float_array"):
                if "positions" in source.attrib['id']:
                    vertices = list(map(float, source.text.strip().split()))
                    break

            if not vertices:
                self.get_logger().error("No vertices found in the DAE file.")
                return

            # Reshape vertices into (x, y, z) format
            points = np.array(vertices).reshape(-1, 3)

            # Create and publish the PointCloud2 message
            header = Header()
            header.frame_id = self.frame_id
            header.stamp = self.get_clock().now().to_msg()
            cloud_msg = self.create_point_cloud2(points, header)
            self.publisher.publish(cloud_msg)

            #self.published_once = True  # Set the flag to prevent further publications
            self.get_logger().info(f"Published point cloud with {len(points)} points")

        except Exception as e:
            self.get_logger().error(f"Failed to parse DAE file: {e}")

    def create_point_cloud2(self, points, header):
        """Create a PointCloud2 message."""
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_cloud = np.zeros(len(points), dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32)
        ])
        point_cloud['x'], point_cloud['y'], point_cloud['z'] = points[:, 0], points[:, 1], points[:, 2]
        return PointCloud2(
            header=header,
            height=1,
            width=len(points),
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=12 * len(points),
            data=point_cloud.tobytes(),
            is_dense=True,
        )

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
