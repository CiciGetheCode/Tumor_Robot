import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time 
class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')
        # Publisher to the joint trajectory controller
        self.publisher = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        
        # Define joint names based on your UR5e robot's setup
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

    def send_trajectory(self):
        # Create a JointTrajectory message
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        # Define the first point in the trajectory
        point1 = JointTrajectoryPoint()
        point1.positions = [0.0, -1.57, 1.57, -1.57, 0.0, 0.0]  # Example positions in radians
        point1.time_from_start.sec = 2  # Move to this position in 2 seconds
        trajectory.points.append(point1)

        # Define the second point in the trajectory
        point2 = JointTrajectoryPoint()
        point2.positions = [0.5, -1.0, 1.0, -1.0, 0.5, 0.0]  # Another example position
        point2.time_from_start.sec = 5  # Move to this position in 5 seconds
        trajectory.points.append(point2)

        # Publish the trajectory
        self.publisher.publish(trajectory)
        self.get_logger().info("Trajectory sent!")

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisher()
    node.send_trajectory()  # Send the defined trajectory
    rclpy.spin(node)  # Keep the node alive to maintain the connection
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
