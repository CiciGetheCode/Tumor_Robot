from compute_convex_hull import compute_convex_hull_trajectory
import numpy as np
import kinematics as kin

def send_trajectory_path(self,event = None):
        

        
        """
        Moves the robotic arm along a trajectory by:
        - Updating the end-effector position and orientation.
        - Computing inverse kinematics.
        - Updating the joint configuration in the UI.
        """


            
        # ✅ Load trajectory from external function
        path_to_dae = "c:/Users/dtsits/aggelos-python/11803_Airplane_v1_l1.dae"
        self.trajectory_points_list = compute_convex_hull_trajectory(
            path_to_dae, 
            max_radius=0.8, 
            fixed_orientation=(0, 0, 0),  # or dynamically computed
            max_point_spacing=0.01
        )

        #  # ✅ Define a test trajectory with waypoints (position, orientation)
        # self.trajectory_points_list = [
        #     ((-0.015, 0.697, 0.203), (-68.5,2.1,53.7)),   # Start position
        #     ((-0.096,0.335,0.477) , (109.6 , -23.9 , -124.3)),
        #     # ((-0.015, 0.697, 0.203), (-68.5,2.1,53.7))
        #    ]   

        if not hasattr(self, 'trajectory_points_list') or not self.trajectory_points_list:
            print("No trajectory points available.")
            return

        print(f"Executing trajectory with {len(self.trajectory_points_list)} points...")

        for position, orientation in self.trajectory_points_list:
            print(f"Moving to: {position}, Orientation: {orientation}")

            # Convert orientation from degrees to radians
            orientation_rad = [np.deg2rad(a) for a in orientation]

            # Save values for consistency (optional)
            self.chosen_invkine_3d_position = list(position)
            self.chosen_invkine_orientation = orientation_rad

            # ✅ Get transformation matrix
            T = self.get_transformation_matrix(self.chosen_invkine_3d_position, self.chosen_invkine_orientation)

            # ✅ Directly solve IK (the missing piece!)
            joint_angles, success = kin.compute_inverse_kinematics(
                self.built_robotic_manipulator,
                T,
                self.invkine_tolerance
            )
            # Save and log
            self.invkine_joints_configuration = joint_angles
            print("💡 Raw IK Joint Angles (rad):", joint_angles)
            if success:
                print("✅ IK Solver succeeded")
                # You can optionally re-use the formatting from your indicator here
            else:
                print("❌ IK Solver failed. No valid solution.")
            
            if success:
                self.forward_kinematics_variables = joint_angles
                self.copy_fkine_to_control_values()
                
            else:
                print("❌ Skipping control update due to failed IK.")

           
            self.send_command_to_all_motors() #go all motors    