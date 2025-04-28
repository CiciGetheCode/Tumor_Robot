import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time

class JointPublisher(Node):
    def __init__(self):
        super().__init__('joint_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.joint_state = JointState()
        self.joint_state.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.joint_state.position = [0.0] * 6  # Initialize joint angles to zero

    def timer_callback(self):
        # Update joint positions here
        # Example: Set a desired position for each joint
        self.joint_state.position = [0.0, -1.0, 1.0, 0.0, 0.5, 0.0]
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(self.joint_state)
        self.get_logger().info("Publishing joint states...")

    

def main(args=None):
    rclpy.init(args=args)
    joint_publisher = JointPublisher()
    rclpy.spin(joint_publisher)
    joint_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
