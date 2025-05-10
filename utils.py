import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial import ConvexHull
import pyvista as pv
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from robots_kinematics import compute_inverse_kinematics


MAX_SHIFT_TRIES = 10
SHIFT_STEP = 0.01  # 1 cm per iteration


class RobotTrajectory:
    def __init__(self, dae_path, min_bounds, max_bounds):
        self.dae_path = dae_path
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.trajectory_points_list = []


    def is_pose_reachable(self, position, orientation):
        """
        Checks if a given pose (position + orientation) is reachable using inverse kinematics.
        
        Parameters:
            position (list or np.ndarray): Target 3D position [x, y, z]
            orientation (list or np.ndarray): Orientation [roll, pitch, yaw] in radians

        Returns:
            bool: True if IK solver finds a valid solution, False otherwise.
        """
        try:
            if not self.robotic_manipulator_is_built:
                return False

            # Get transformation matrix from position and orientation
            target_pose_T = self.get_transformation_matrix(position, orientation)

            # Call the existing IK solver function
            _, success = kin.compute_inverse_kinematics(self.built_robotic_manipulator, target_pose_T, self.invkine_tolerance)

            return bool(success)
        except:
            return False

    def adjust_pose_until_reachable(self, pos, orientation, max_tries=MAX_SHIFT_TRIES,shift_step=0.1):
        pos = np.array(pos)
        for _ in range(max_tries):
            if self.is_pose_reachable(pos, orientation):        
                return pos, orientation
            pos[2] += shift_step
            print("Shifted_step is: " ,shift_step)
        
        return None, orientation


    def center_points_to_origin(self,points):
        """
        Move (translate) a set of 3D points so that their centroid becomes (0, 0, 0).
        
        Args:
            points (np.ndarray): (N, 3) array of points.

        Returns:
            np.ndarray: Centered points.
        """
        points = np.asarray(points)

        # Compute centroid
        centroid = np.mean(points, axis=0)

        # Subtract centroid from all points (move to 0,0,0)
        centered_points = points - centroid

        return centered_points

    def extract_points_from_dae(self,dae_path):
        """
        Parses a .dae (COLLADA) file and extracts 3D points from <float_array> entries.
        Assumes the data represents positions in x, y, z triples.
        """
        try:
            tree = ET.parse(dae_path)
            root = tree.getroot()

            ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}  # default COLLADA namespace

            # You may need to adapt this depending on the structure of your file
            float_arrays = root.findall('.//c:float_array', ns)
            all_points = []

            for fa in float_arrays:
                floats = list(map(float, fa.text.strip().split()))
                if len(floats) % 3 != 0:
                    continue
                # for i in range(0, len(floats), 3):
                #     point = floats[i:i+3]
                #     all_points.append(point)
                for i in range(0, len(floats), 3): 
                    point = floats[i:i+3]
                    if len(point) == 3:
                        all_points.append(point)

            return np.array(all_points)

        except Exception as e:
            print(f"❌ Error reading .dae file: {e}")
            return None

    def compute_convex_hull_from_dae(self,dae_path):
        points = self.extract_points_from_dae(dae_path)

        if points is None or len(points) < 4:
            print("⚠️ Not enough points to compute a convex hull.")
            return None

        print(f"✅ Loaded {len(points)} points from file.")
        hull = ConvexHull(points)
        convex_hull_points = points[hull.vertices]
        return convex_hull_points

    def interpolate_line_segment(self,p1, p2, max_point_spacing=0.01):
        """
        Interpolate between p1 and p2 with dynamic resolution based on distance.
        max_point_spacing defines the max distance between two points.
        """
        distance = np.linalg.norm(p2 - p1)
        num_points = max(int(distance / max_point_spacing), 2)  # at least 2 points
        # print([tuple(p1 + (p2 - p1) * t) for t in np.linspace(0, 1, num_points)])

        return [tuple(p1 + (p2 - p1) * t) for t in np.linspace(0, 1, num_points)]

    def compute_convex_hull_trajectory(self,dae_path, max_radius=0.8, fixed_orientation=(0, 0, 0), max_point_spacing=0.5):
        points = self.extract_points_from_dae(dae_path)
        if points is None or len(points) < 4:
            print("⚠️ Not enough points to compute a convex hull.")
            return []

        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        projected_points = hull_points

        trajectory = []
        for i in range(len(projected_points) - 1):
            p1 = np.array(projected_points[i])
            p2 = np.array(projected_points[i + 1])
            segment = self.interpolate_line_segment(p1, p2, max_point_spacing)
            for pos in segment:
                trajectory.append((pos, fixed_orientation))

        # Optional: close the loop
        p1 = np.array(projected_points[-1])
        p2 = np.array(projected_points[0])
        segment = self.interpolate_line_segment(p1, p2, max_point_spacing)
        for pos in segment:
            trajectory.append((pos, fixed_orientation))
        
        return trajectory

    def compute_convex_hull_trajectory_3d(self,dae_path, max_radius=0.8, fixed_orientation=(0, 0, 0), max_point_spacing=0.5):
        points = self.extract_points_from_dae(dae_path)
        if points is None or len(points) < 4:
            print("⚠️ Not enough points to compute a convex hull.")
            return np.array([])

        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        projected_points = hull_points

        trajectory = []
        for i in range(len(projected_points) - 1):
            p1 = np.array(projected_points[i])
            p2 = np.array(projected_points[i + 1])
            segment = self.interpolate_line_segment(p1, p2, max_point_spacing)
            for pos in segment:
                trajectory.append((pos, fixed_orientation))

        # Optional: close the loop
        p1 = np.array(projected_points[-1])
        p2 = np.array(projected_points[0])
        segment = self.interpolate_line_segment(p1, p2, max_point_spacing)
        for pos in segment:
            trajectory.append((pos, fixed_orientation))

        # Convert trajectory to np.array with 3D points
        trajectory_np = np.array([pos for pos, _ in trajectory])

        # Reshape to have 3 dimensions (N, 1, 3)
        trajectory_3d = trajectory_np.reshape((-1, 1, 3))

        return trajectory_3d

    def calculate_initial_position(self,points, orientation=np.array([0, 0, 0]), z_offset=0.1):
        """
        Computes a safe starting pose: at the X,Y centroid, and Z set to 1 cm above the object's topmost point.
        Returns np.ndarray for both position and orientation.
        """
        if points is None or len(points) == 0:
            raise ValueError("❌ Point cloud is empty. Can't compute initial position.")

        points = np.reshape(points, (-1, 3))


        # Make sure we have at least 3 columns
        if points.shape[1] != 3:
            raise ValueError("❌ Expected 3D points (x, y, z), but got shape: {}".format(points.shape))

        # Centroid in X and Y (flatten in case result is multi-dimensional)
        centroid_xy = np.mean(points[:, :2], axis=0).flatten()

        # Max Z (highest point)
        max_z = np.max(points[:, 2])
        safe_z = max_z + z_offset

        # ✅ Ensure scalars and create position
        initial_pos = np.array([float(centroid_xy[0]), float(centroid_xy[1]), float(safe_z)])
        orientation = np.array(orientation, dtype=float)

        return initial_pos, orientation

    def calculate_initial_position_3d(self,points, orientation=np.array([0, 0, 0]), z_offset=0.01):
        """
        Computes a safe starting pose: at the X,Y centroid, and Z set to 1 cm above the object's topmost point.
        Returns np.ndarray for both position and orientation.

        Args:
            points (np.ndarray): 3D points cloud (N x 3)
            orientation (np.ndarray): The orientation as a 3D vector (default is (0, 0, 0))
            z_offset (float): Offset added to the Z-coordinate (default is 0.01)

        Returns:
            np.ndarray: The initial position (shape: (1, 3))
            np.ndarray: The orientation (shape: (1, 3))
        """
        if points is None or len(points) == 0:
            raise ValueError("❌ Point cloud is empty. Can't compute initial position.")

        points = np.reshape(points, (-1, 3))

        # Make sure we have at least 3 columns
        if points.shape[1] != 3:
            raise ValueError("❌ Expected 3D points (x, y, z), but got shape: {}".format(points.shape))

        # Centroid in X and Y (flatten in case result is multi-dimensional)
        centroid_xy = np.mean(points[:, 1:], axis=0).flatten()

        # Max Z (highest point)
        max_x = np.max(points[:, 0])
        safe_x = max_x + z_offset

        # Ensure scalars and create position
        initial_pos = np.array([float(safe_x), float(centroid_xy[0]), float(centroid_xy[1])])

        # Reshape to have 3 dimensions (1, 3)
        initial_pos_3d = initial_pos.reshape((1, 1, 3))

        # Reshape orientation to (1, 1, 3)
        orientation_3d = np.array(orientation, dtype=float).reshape((1, 1, 3))

        return initial_pos_3d, orientation_3d

    def compute_cross_trajectory(self,initial_position,points, line_spacing=0.1):
        """
        Generate a cross-shaped trajectory centered on the initial_position.
        The cross consists of two lines (X and Y directions), each extending
        1cm beyond the object's bounds on both sides.

        Args:
            initial_position (np.ndarray): 3D point (x, y, z)
            visualization_points (np.ndarray): 3D point cloud (N x 3)
            line_spacing (float): spacing between points on the lines (in meters)

        Returns:
            np.ndarray: List of 3D points forming the cross
        """
        if initial_position is None or len(initial_position) != 3:
            print("⚠️ Initial position NOT valid")
            return np.array([])

        # Make sure points are in the correct shape
        points = np.reshape(points, (-1, 3))

        # Determine object width and length (X and Y spans)
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

        # Compute full width and length
        width = max_x - min_x + 0.02  # +1cm on each side
        length = max_y - min_y + 0.02

        # Half extents
        half_width = width / 2
        half_length = length / 2

        # Z remains constant for the cross
        z = initial_position[2]

        # Points along X-axis (cross bar)
        x_start = initial_position[0] - half_width
        x_end   = initial_position[0] + half_width
        x_line = np.linspace(x_start, x_end, int(width / line_spacing))
        x_points = np.array([[x, initial_position[1], z] for x in x_line])

        # Points along Y-axis (cross bar)
        y_start = initial_position[1] - half_length
        y_end   = initial_position[1] + half_length
        y_line = np.linspace(y_start, y_end, int(length / line_spacing))
        y_points = np.array([[initial_position[0], y, z] for y in y_line])

        # Combine all into a single array
        cross_trajectory = np.vstack((x_points, y_points))

        return cross_trajectory

    def compute_cross_trajectory_3d(self,initial_position, points, line_spacing=10):
        """
        Generate a cross-shaped trajectory centered on the initial_position.
        The cross consists of two lines (X and Y directions), each extending
        1cm beyond the object's bounds on both sides.

        Args:
            initial_position (np.ndarray): 3D point (x, y, z)
            points (np.ndarray): 3D point cloud (N x 3)
            line_spacing (float): spacing between points on the lines (in meters)

        Returns:
            np.ndarray: List of 3D points forming the cross with shape (N, 1, 3)
        """
        if initial_position is None or len(initial_position) != 3:
            print("⚠️ Initial position NOT valid")
            return np.array([])

        # Make sure points are in the correct shape
        points = np.reshape(points, (-1, 3))

        # return cross_trajectory_3d
        # Compute bounds in Y and Z directions
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])

        # Extend by 1cm on each side
        height = max_z - min_z + 1
        length = max_y - min_y - 5

        half_height = height / 2
        half_length = length / 2

        x = initial_position[0]

        # Line along Y
        y_line = np.linspace(initial_position[1] - half_length, initial_position[1] + half_length, int(length / line_spacing))
        y_points = np.array([[x, y, initial_position[2]] for y in y_line])

        # Line along Z
        z_line = np.linspace(initial_position[2] - half_height, initial_position[2] + half_height, int(height / line_spacing))
        z_points = np.array([[x, initial_position[1], z] for z in z_line])

        # Combine
        cross_trajectory = np.vstack((y_points, z_points))
        return cross_trajectory.reshape((-1, 1, 3))

    def shift_points_to_fit_workspace(self,points, min_bounds, max_bounds):
        """
        Shifts raw 3D points (Nx3) so their bounding box fits inside the workspace box.
        """
        object_min = points.min(axis=0)
        object_max = points.max(axis=0)

        shift = np.zeros(3)
        for i in range(3):
            if object_min[i] < min_bounds[i]:
                shift[i] = min_bounds[i] - object_min[i]
            elif object_max[i] > max_bounds[i]:
                shift[i] = max_bounds[i] - object_max[i]
        
         # Apply the shift to the points
        shifted_points = (points + shift)
        # Print the shifted points' min and max values for debugging
        print(f"Shifted points min: {shifted_points.min(axis=0)}")
        print(f"Shifted points max: {shifted_points.max(axis=0)}")

        return shifted_points

    def project_object_to_workspace(self,object_center, object_size, min_bounds, max_bounds):
        """
        Shift the object center if needed so that the entire object fits in the workspace.
        object_size: np.array([dx, dy, dz])
        """
        half_size = object_size / 2.0
        object_min = object_center - half_size
        object_max = object_center + half_size

        shift = np.zeros(3)
        for i in range(3):  # for x, y, z
            if object_min[i] < min_bounds[i]:
                shift[i] = min_bounds[i] - object_min[i]
            elif object_max[i] > max_bounds[i]:
                shift[i] = max_bounds[i] - object_max[i]

        adjusted_center = object_center + shift
        return adjusted_center

    def project_points_to_workspace(self,points, max_radius=0.8, offset=np.array([0, 0, 0])):
        """
        Normalize and scale points to fit within a sphere of max_radius from offset.
        """
        projected = []
        for pt in points:
            pt = np.array(pt)
            direction = pt - offset
            distance = np.linalg.norm(direction)

            if distance == 0:
                continue  # avoid division by zero

            # Scale down if point is outside workspace
            if distance > max_radius:
                direction = direction / distance * max_radius
            projected_point = offset + direction
            projected.append(tuple(projected_point))

        return projected

    def compute_total_trajectory_for_robot(self,convex_hull_points, cross_points, initial_position):
        """
        Combines the initial position, cross trajectory, and convex hull trajectory
        into a single 3D trajectory array suitable for the robot.

        Args:
            convex_hull_points (np.ndarray): Trajectory points along the convex hull (N, 1, 3)
            cross_points (np.ndarray): Cross trajectory points (M, 1, 3)
            initial_position (np.ndarray): Initial position (1, 1, 3) or (1, 3)

        Returns:
            np.ndarray: Combined trajectory of shape (N+M+1, 1, 3)
        """
        # Ensure all are NumPy arrays
        convex_hull_points = np.asarray(convex_hull_points, dtype=float)
        cross_points = np.asarray(cross_points, dtype=float)
        initial_position = np.asarray(initial_position, dtype=float)

        # Reshape to consistent (N, 1, 3) format
        if initial_position.shape == (1, 3):
            initial_position = initial_position.reshape((1, 1, 3))
        elif initial_position.shape == (3,):
            initial_position = initial_position.reshape((1, 1, 3))

        if cross_points.ndim == 2:
            cross_points = cross_points.reshape((-1, 1, 3))

        if convex_hull_points.ndim == 2:
            convex_hull_points = convex_hull_points.reshape((-1, 1, 3))

        print("initial_position shape:", initial_position.shape)
        print("cross_points shape:", cross_points.shape)
        print("convex_hull_points shape:", convex_hull_points.shape)


        # Combine all trajectories vertically
        final_trajectory_positions = np.vstack((initial_position, cross_points, convex_hull_points))[:, 0, :]

        # Define fixed orientation
        fixed_orientation = np.array([-1 ,0, 0], dtype=float)
        # Define the rotation: 45 degrees around the Y-axis
        rotation = R.from_euler('y', 45, degrees=True)
        fixed_orientation = rotation.apply([0, 0, 1])


        # Repeat the fixed orientation for each position
        repeated_orientations = np.tile(fixed_orientation, (final_trajectory_positions.shape[0], 1))

        # Combine positions and orientations into one array
        final_trajectory_for_robot = np.stack((final_trajectory_positions, repeated_orientations), axis=1)

        return final_trajectory_for_robot

    def select_widest_boundary_points_along_z(self,points):
        """
        Select points that lie only on the minimum and maximum Z boundaries.
        
        Args:
            points (np.ndarray): Array of 3D points (N, 3)

        Returns:
            np.ndarray: Boundary points only (top and bottom surfaces)
        """
        points = np.asarray(points)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be of shape (N, 3)")

        # Find min and max Z
        min_z = np.min(points[:, 2])
        max_z = np.max(points[:, 2])

        # Select only points that are at min_z or max_z
        boundary_points = points[
            (np.isclose(points[:, 2], min_z)) |
            (np.isclose(points[:, 2], max_z))
        ]

        return boundary_points

    def extract_outermost_ring(self,points, projection='xy'):
        """
        Collapses the orthogonal axis and computes the convex hull in the selected 2D projection.
        Returns 3D ring points with collapsed axis set to mean value.

        Args:
            points (np.ndarray): Full 3D point cloud (N, 3)
            projection (str): One of 'xy', 'yz', or 'xz'

        Returns:
            np.ndarray: 3D points forming the outer ring in the selected plane
        """
        points = np.asarray(points)
        
        if projection == 'xy':
            avg_z = np.min(points[:, 2])
            plane_2d = points[:, :2]
            collapsed = lambda x, y: [x, y, avg_z]

        elif projection == 'yz':
            avg_x = np.min(points[:, 0])
            plane_2d = points[:, 1:3]
            collapsed = lambda y, z: [avg_x, y, z]

        elif projection == 'xz':
            avg_y = np.min(points[:, 1])
            plane_2d = points[:, [0, 2]]
            collapsed = lambda x, z: [x, avg_y, z]

        else:
            raise ValueError("Projection must be 'xy', 'yz', or 'xz'")

        # Compute convex hull in the projected 2D plane
        hull = ConvexHull(plane_2d)
        hull_indices = hull.vertices
        ring_points = np.array([collapsed(*plane_2d[i]) for i in hull_indices])

        return ring_points

   
    def compute_optimal_transform(self, points, rotation_deg=45):
        """
        Automatically compute base_plane_origin and manual_offset to fit rotated object inside workspace.
        """
        points = np.asarray(points)
        centroid = points.mean(axis=0)

        # Step 1: Center points
        centered_points = points - centroid

        # Step 2: Rotate
        rotation = R.from_euler('y', rotation_deg, degrees=True)
        rotated_points = rotation.apply(centered_points)

        # Step 3: Check if object fits
        obj_min = rotated_points.min(axis=0)
        obj_max = rotated_points.max(axis=0)
        obj_size = obj_max - obj_min
        workspace_size = self.max_bounds - self.min_bounds

        if np.any(obj_size > workspace_size):
            raise ValueError("❌ Object too large to fit inside workspace.")

        # Step 4: Compute offset to shift into workspace
        desired_min = self.min_bounds + 0.01  # add margin
        shift = desired_min - obj_min

        # Step 5: Return origin + shift
        return centroid, shift

    def shift_dae_file_vertices(self, input_dae_path, output_dae_path,  base_plane_origin=np.array([0.0, 0.0, 0.0]), plane_rotation_deg=45, manual_offset=np.array([0.0, 0.0, 0.0])):
            
        tree = ET.parse(input_dae_path)
        root = tree.getroot()
        ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}

        float_arrays = root.findall(".//c:float_array", ns)
        if not float_arrays:
            raise ValueError("❌ Could not find <float_array> elements in DAE file.")

        # Rotation to match the plane's inclination (around Y-axis)
        rotation = R.from_euler('y', plane_rotation_deg, degrees=True)

        for float_array in float_arrays:
            float_values = list(map(float, float_array.text.strip().split()))
            points = np.array(float_values).reshape(-1, 3)

            # Center object at origin
            centroid = points.mean(axis=0)
            centered_points = points - centroid

            # Rotate around Y-axis
            rotated_points = rotation.apply(centered_points)

            # Translate to plane center
            transformed_points = rotated_points + base_plane_origin + manual_offset 
            
            # Debug output
            print(f"Transformed points min: {transformed_points.min(axis=0)}")
            print(f"Transformed points max: {transformed_points.max(axis=0)}")

            # Update <float_array> text
            updated_flat = ' '.join(f'{coord:.6f}' for coord in transformed_points.flatten())
            float_array.text = updated_flat

        tree.write(output_dae_path)
        self.last_plane_normal = rotation.apply([0, 0, 1])  # Normal vector to rotated plane
        print(f"✅ Transformed DAE saved to: {output_dae_path}")
    
    def do_everything(self,input_dae_path,output_dae_path):
        #  Define your reachable workspace bounds
        min_bounds = np.array([-0.32500893, -0.14676859,  0.009])
        max_bounds = np.array([ 0.66500893,  0.74676859,  0.68019031])

        dae_path = os.path.expanduser("~/experiment/robotic_manipulators_playground/mesh_mask.dae")
        # points = extract_points_from_dae(dae_path)
        # shifted_points = self.shift_points_to_fit_workspace(points, min_bounds, max_bounds)
       
        # points = self.extract_points_from_dae(output_dae_path)
        # # print("THis is the first",points[0])
        # points /= 100

        # base_origin, offset = self.compute_optimal_transform(points, rotation_deg=45)

        self.shift_dae_file_vertices(
            input_dae_path=input_dae_path,
            output_dae_path=output_dae_path,
            base_plane_origin=np.array([34.0, 23.0, 25.0]),
            plane_rotation_deg=45,
            manual_offset=np.array([0.0, 10.0, 20.0])
        )
        path_to_shifted_dae = os.path.expanduser(output_dae_path)
        convex_hull_points = self.compute_convex_hull_from_dae(path_to_shifted_dae) 
        
        # Define the fixed orientation vector (e.g., -X axis)
        fixed_orientation_vector = np.array([-1, 0, 0], dtype=float)
        # Rotate it 45 degrees around Y-axis to align with the inclined plane
        rotation = R.from_euler('y', 45, degrees=True)
        rotated_orientation_vector = rotation.apply(fixed_orientation_vector)



        if convex_hull_points is not None:
            print("🔺 Original Convex Hull Points:")
            # transformed_points = project_points_to_workspace(convex_hull_points, max_radius=0.8)
            transformed_points = convex_hull_points
            points_array = np.array(transformed_points)
            # print(len(transformed_points))

            initial_position, orientation = self.calculate_initial_position_3d(points_array, orientation= rotated_orientation_vector)
            # Compute cross trajectory
            cross_points = self.compute_cross_trajectory_3d(initial_position.flatten(), points_array)
            print("Cross points are : " ,len(cross_points))
            # points = self.compute_convex_hull_trajectory_3d(path_to_shifted_dae, max_radius=0.8, fixed_orientation=fixed_orientation, max_point_spacing=0.01)

        # # 🔧 FIX: Extract just the positions
        # trajectory_positions = [points[i][0] for i in range(len(points)) if i % 10 == 0]
    
        # # # Step 3: Get pre-position movement
        # # initial_position,orientation= calculate_initial_position_3d(points, orientation=[0, 0, 0])
        # initial_point = initial_position.reshape(1, 3)
    
        
            convex_hull_points_trajectory = self.extract_outermost_ring(points_array,projection='yz')
            trajectory = self.compute_total_trajectory_for_robot(convex_hull_points=convex_hull_points_trajectory,
                                                        cross_points=cross_points,
                                                        initial_position=initial_position
                                                        )
            
            # Set numpy to print floats in the usual format, without scientific notation
            np.set_printoptions(precision=8, suppress=True)

            trajectory_points_list = np.array(trajectory)

            # print(trajectory_points_list[0:10],'pre devision')

            # Divide only the first column (x values) by 100
            # trajectory_points_list[:, 0,:] = trajectory_points_list[:, 0,:] / 100
            trajectory_points_list[:, 0:1, :] = trajectory_points_list[:, 0:1, :] / 100  # scale all positions
            # 🔧 Shift the trajectory to be in a reachable region
            trajectory_points_list[:, 0:1, 0] += 0.25   # X → front
            trajectory_points_list[:, 0:1, 1] += 0.25   # Y → right
            trajectory_points_list[:, 0:1, 2] += 0.15   # Z → higher
                        # print(trajectory_points_list[0:10],'post devision')
            min_values = trajectory_points_list[:,0,:].min(axis=0)
            print(min_values)
            max_values = trajectory_points_list[:,0,:].max(axis=0) 
            print(max_values)

            # Print trajectory details
            print(f"Generated trajectory with {len(trajectory)} points.")
            print(f"Trajectory dtype: {trajectory.dtype}")
            trajectory_output = []
            # for pos, orientation_vec in trajectory:
            #                 # Prevent warning from align_vectors when vectors are already aligned
            #     if np.allclose(orientation_vec, [0, 0, 1], atol=1e-6):
            #         euler_deg = (0.0, 0.0, 0.0)
            #     else:
            #         euler_deg = tuple(R.from_rotvec(rotation.as_rotvec()).as_euler('xyz', degrees=True))
            for pos, orientation_vec in trajectory:
                adjusted_pos, adjusted_ori = self.adjust_pose_until_reachable(pos, orientation_vec)
                if adjusted_pos is not None:
                    pos_tuple = tuple(adjusted_pos)
                    euler_deg = tuple(R.from_rotvec(R.from_euler('y', 45, degrees=True).as_rotvec()).as_euler('xyz', degrees=True))
                    trajectory_output.append((pos_tuple, euler_deg))
                else:
                    print(f"❌ Could not adjust pose {pos} to be reachable. Skipping.")


                pos_tuple = tuple(pos / 100)
                trajectory_output.append((pos_tuple, euler_deg))

            self.trajectory_points_list = trajectory_output
            return self.trajectory_points_list 
        else:
            print("❌ No convex hull points generated.")
            self.trajectory_points_list = None
            return None
        

    def visualize_points(self, points):
        
        # Trajectory points from the user
        self.points = self.trajectory_points_list
        positions = [pos for pos, _ in points]
        x_vals, y_vals, z_vals = zip(*positions)

        # Create the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_vals, y_vals, z_vals, marker='o')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("3D Trajectory Path")

        # Draw boundary box lines
        x_range = [-0.325, 0.665]
        y_range = [-0.147, 0.747]
        z_range = [0.009, 0.680]

        # All 8 corners of the bounding box
        corners = [
            (x_range[0], y_range[0], z_range[0]),
            (x_range[0], y_range[0], z_range[1]),
            (x_range[0], y_range[1], z_range[0]),
            (x_range[0], y_range[1], z_range[1]),
            (x_range[1], y_range[0], z_range[0]),
            (x_range[1], y_range[0], z_range[1]),
            (x_range[1], y_range[1], z_range[0]),
            (x_range[1], y_range[1], z_range[1]),
        ]

        # Draw lines between corner pairs to form the box
        edges = [
            (0, 1), (0, 2), (0, 4),
            (3, 1), (3, 2), (3, 7),
            (5, 1), (5, 4), (5, 7),
            (6, 2), (6, 4), (6, 7)
        ]

        for start, end in edges:
            x = [corners[start][0], corners[end][0]]
            y = [corners[start][1], corners[end][1]]
            z = [corners[start][2], corners[end][2]]
            ax.plot(x, y, z, color='gray', linestyle='--')

        plt.tight_layout()
        plt.show()

       
if __name__ == "__main__":
    # Define workspace bounds and dae_path
    min_bounds = np.array([-0.32500893, -0.14676859, 0.009])
    max_bounds = np.array([0.66500893, 0.74676859, 0.68019031])
    input_dae_path = os.path.expanduser("~/experiment/robotic_manipulators_playground/mesh_mask.dae")
    output_dae_path = os.path.expanduser("~/experiment/robotic_manipulators_playground/shifted_object.dae")
    # Instantiate the RobotTrajectory class
    robot_trajectory = RobotTrajectory(input_dae_path, min_bounds, max_bounds)

    # Call the 'do_everything' method to compute the trajectory
    robot_trajectory.do_everything(input_dae_path,output_dae_path)
    # Print the first 5 and last 5 characters of the input DAE path
    # print(f"Input DAE path (first 5 characters): {input_dae_path[:5]}")
    # print(f"Input DAE path (last 5 characters): {input_dae_path[-5:]}")

    # # Print the first 5 and last 5 characters of the output DAE path
    # print(f"Output DAE path (first 5 characters): {output_dae_path[:5]}")
    # print(f"Output DAE path (last 5 characters): {output_dae_path[-5:]}")


    # Retrieve the computed trajectory points list
    trajectory_points = robot_trajectory.trajectory_points_list
    if len(trajectory_points)>0:
        print(f"Computed trajectory with {len(trajectory_points)} points.")
        # print(trajectory_points)
    else:
        print("No trajectory points available.convex")

    robot_trajectory.visualize_points(trajectory_points)

