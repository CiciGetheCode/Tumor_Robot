import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial import ConvexHull
import pyvista as pv
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
# from robots_kinematics import compute_inverse_kinematics 
# from robotic_manipulators_playground import robotic_manipulators_playground_window 
# import kinematics as kin
import pandas as pd
MAX_SHIFT_TRIES = 10
SHIFT_STEP = 0.01  # 1 cm per iteration

# from utils import RobotTrajectory
# import os 
# import numpy as np

class RobotTrajectory:
    def __init__(self, dae_path, min_bounds, max_bounds):
        self.dae_path = dae_path
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.trajectory_points_list = []
        self.cross_trajectory = []
        self.convex_trajectory = []
        self.cross_along_y  = []
        self.cross_along_z = []
        self.robottrajectory_instance = None 

    def split_cross_only_by_y_and_z(self, cross_only):
        """
        Splits the cross trajectory into:
        - `cross_along_y`: one representative point per unique Y-value (based on Z proximity to center)
        - `cross_along_z`: all remaining points not in `cross_along_y`
        """
        cross_only_array = np.array([np.array(p[0]) for p in cross_only])  # shape (N, 3)

        self.cross_along_y = []
        self.cross_along_z = []

        y_vals = cross_only_array[:, 1]
        z_vals = cross_only_array[:, 2]
        mid_z = (np.max(z_vals) + np.min(z_vals)) / 2

        grouped_by_y = defaultdict(list)
        for point in cross_only_array:
            y_rounded = round(point[1], 6)
            grouped_by_y[y_rounded].append(point)

        cross_along_y_set = set()

        for y_key, points in grouped_by_y.items():
            # Pick the point with Z closest to mid_z as the Y representative
            best_y_point = min(points, key=lambda p: abs(p[2] - mid_z))
            self.cross_along_y.append(best_y_point)
            cross_along_y_set.add(tuple(best_y_point))  # For fast comparison

        for point in cross_only_array:
            if tuple(point) not in cross_along_y_set:
                self.cross_along_z.append(point)

        self.cross_along_y = np.array(self.cross_along_y)
        self.cross_along_z = np.array(self.cross_along_z)

        print("✅ cross_along_y count:", len(self.cross_along_y))
        print("✅ cross_along_z count:", len(self.cross_along_z))

    # def split_cross_only_by_y_and_z(self, cross_only):
    #     """
    #     Organizes the cross trajectory points by unique y-values.
    #     Keeps one representative per y (closest to mid-z), and places the rest in cross_along_z.
    #     """
    #     cross_only_array = np.array([np.array(p[0]) for p in cross_only])  # Extract positions

    #     self.cross_along_y = []
    #     self.cross_along_z = []

    #     y_vals = cross_only_array[:, 1]
    #     z_vals = cross_only_array[:, 2]
    #     mid_y = (np.max(y_vals) + np.min(y_vals)) / 2
    #     mid_z = (np.max(z_vals) + np.min(z_vals)) / 2

    #     grouped_by_y = defaultdict(list)
    #     for i, point in enumerate(cross_only_array):
    #         y_rounded = round(point[1], 6)
    #         grouped_by_y[y_rounded].append(point)

    #     for y_key, points in grouped_by_y.items():
    #         if len(points) == 1:
    #             self.cross_along_y.append(points[0])
    #         else:
    #             sorted_points = sorted(points, key=lambda p: p[2])
    #             mid_candidates = [p for p in sorted_points if abs(p[1] - mid_y) < 1e-6]

    #             if mid_candidates:
    #                 mid_z_point = min(mid_candidates, key=lambda p: abs(p[2] - mid_z))
    #                 self.cross_along_y.append(mid_z_point)
    #                 for p in sorted_points:
    #                     if not np.allclose(p, mid_z_point, atol=1e-8):
    #                         self.cross_along_z.append(p)
    #             else:
    #                 self.cross_along_y.append(sorted_points[0])
    #                 self.cross_along_z.extend(sorted_points[1:])

    #     self.cross_along_y = np.array(self.cross_along_y)
    #     self.cross_along_z = np.array(self.cross_along_z)

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
        # Rotate trajectory positions around Y axis to match the inclined object
        rotation = R.from_euler('y', -45, degrees=True)
        rotation2 = R.from_euler('z', 180, degrees=True)
        final_trajectory_positions = rotation.apply(final_trajectory_positions)
        final_trajectory_positions = rotation2.apply(final_trajectory_positions)
        
        
        # Define fixed orientation
        fixed_orientation = np.array([-1 ,0, 0], dtype=float)        
        # Define the rotation: 45 degrees around the Y-axis
        rotation = R.from_euler('y', 45, degrees=True)
        # rotation = R.from_euler('z', 180, degrees=True)
        fixed_orientation = rotation.apply([0, 0, 1])
        # Repeat the fixed orientation for each position
        repeated_orientations = np.tile(fixed_orientation, (final_trajectory_positions.shape[0], 1))
        # Combine positions and orientations into one array
        final_trajectory_for_robot = np.stack((final_trajectory_positions, repeated_orientations), axis=1)

        
        
        
        
        return final_trajectory_for_robot

    def compute_total_trajectory_for_robot_SCARA(self,convex_hull_points, cross_points, initial_position):
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
                # Rotate trajectory positions around Y axis to match the inclined object
        rotation = R.from_euler('y', -90, degrees=True)
        final_trajectory_positions = rotation.apply(final_trajectory_positions)
        # Define fixed orientation
        fixed_orientation = np.array([-1 ,0, 0], dtype=float)
        # Define the rotation: 45 degrees around the Y-axis
        rotation = R.from_euler('y', -90, degrees=True)
        fixed_orientation = rotation.apply([0, 0, 1])


        # Repeat the fixed orientation for each position
        repeated_orientations = np.tile(fixed_orientation, (final_trajectory_positions.shape[0], 1))

        # Combine positions and orientations into one array
        final_trajectory_for_robot = np.stack((final_trajectory_positions, repeated_orientations), axis=1)

        return final_trajectory_for_robot


    def compute_total_trajectory_for_robot_cross_algorithm(self,convex_hull_points, cross_points, initial_position):
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
        # Rotate trajectory positions around Y axis to match the inclined object
        rotation = R.from_euler('y', 0, degrees=True)
        rotation2 = R.from_euler('z', 0, degrees=True)
        final_trajectory_positions = rotation.apply(final_trajectory_positions)
        final_trajectory_positions = rotation2.apply(final_trajectory_positions)
        
        
        # Define fixed orientation
        fixed_orientation = np.array([-1 ,0, 0], dtype=float)        
        # Define the rotation: 45 degrees around the Y-axis
        rotation = R.from_euler('y', 45, degrees=True)
        # rotation = R.from_euler('z', 180, degrees=True)
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

    def extract_outermost_ring(self, points, projection='xy', max_point_spacing=0.01):
        """
        Compute a smooth ring on the outer boundary of a point cloud, in 3D space,
        based on a 2D convex hull in a selected projection plane.
        """
        points = np.asarray(points)

        if projection == 'xy':
            collapsed_axis = 2
            plane_2d = points[:, :2]
            collapsed = lambda x, y: [x, y, np.min(points[:, 2])]

        elif projection == 'yz':
            collapsed_axis = 0
            plane_2d = points[:, 1:3]
            collapsed = lambda y, z: [np.min(points[:, 0]), y, z]

        elif projection == 'xz':
            collapsed_axis = 1
            plane_2d = points[:, [0, 2]]
            collapsed = lambda x, z: [x, np.min(points[:, 1]), z]

        else:
            raise ValueError("Projection must be 'xy', 'yz', or 'xz'")

        # Compute convex hull
        hull = ConvexHull(plane_2d)
        hull_indices = hull.vertices

        interpolated_ring_3d = []

        # Interpolate between consecutive points on the hull
        for i in range(len(hull_indices)):
            i1 = hull_indices[i]
            i2 = hull_indices[(i + 1) % len(hull_indices)]  # wraps to close the loop
            p1_2d = plane_2d[i1]
            p2_2d = plane_2d[i2]

            segment = self.interpolate_line_segment(p1_2d, p2_2d, max_point_spacing)
            for pt in segment:
                interpolated_ring_3d.append(collapsed(*pt))

        return np.array(interpolated_ring_3d).tolist() 
 
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
            centered_points = centered_points - manual_offset

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
    
    def do_everything(self,input_dae_path,output_dae_path,manual_offset = np.array([0.0,0.0,0.0]),fixed_plane_orientation_rpy=np.array([0.0, 0.0, 0.0])):
        #  Define your reachable workspace bounds
        min_bounds = np.array([-0.32500893, -0.14676859,  0.009])
        max_bounds = np.array([ 0.66500893,  0.74676859,  0.68019031])

        dae_path = os.path.expanduser("~/experiment/robotic_manipulators_playground/mesh_mask_scaled_3x.dae")
        # points = extract_points_from_dae(dae_path)
        # shifted_points = self.shift_points_to_fit_workspace(points, min_bounds, max_bounds)
       
        # points = self.extract_points_from_dae(output_dae_path)
        # # print("THis is the first",points[0])
        # points /= 100

        # base_origin, offset = self.compute_optimal_transform(points, rotation_deg=45)
        # points = self.extract_points_from_dae(input_dae_path)
        # base_origin, offset = self.compute_optimal_transform(points, rotation_deg=45)

        self.shift_dae_file_vertices(
            input_dae_path=input_dae_path,
            output_dae_path=output_dae_path,
            base_plane_origin=np.array([32.0, 0.0, 51.0]),
            plane_rotation_deg=45,
            manual_offset=manual_offset)
        
        path_to_shifted_dae = os.path.expanduser(output_dae_path)
        convex_hull_points = self.compute_convex_hull_from_dae(path_to_shifted_dae) 
        
        # Define the fixed orientation vector (e.g., -X axis)
        # fixed_orientation_vector = np.array([-1, 0, 0], dtype=float)
        # # Rotate it 45 degrees around Y-axis to align with the inclined plane
        # rotation = R.from_euler('y', 45, degrees=True)
        # rotated_orientation_vector = rotation.apply(fixed_orientation_vector)
          # rotated_orientation_vector = rotation.apply(fixed_orientation_vector)
        rotated_orientation_vector = fixed_plane_orientation_rpy




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
    
        
            convex_hull_points_trajectory = self.extract_outermost_ring(points_array,projection='yz',max_point_spacing=5)
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
            trajectory_points_list[:, 0:1, :] = trajectory_points_list[:, 0:1, :] / 1000 # scale all positions
            # 🔧 Shift the trajectory to be in a reachable region
            # trajectory_points_list[:, 0:1, 0] += 0.25   # X → front
            trajectory_points_list[:, 0:1, 1] -= 0.25   # Y → right
            # trajectory_points_list[:, 0:1, 2] += 0.15   # Z → higher
            #             # 🔁 Rotate trajectory positions to match tool orientation
            # rotation = R.from_euler('y', 45, degrees=True)
            # trajectory_points_list[:, 0, :] = rotation.apply(trajectory_points_list[:, 0, :])

            # print(trajectory_points_list[0:10],'post devision')
            min_values = trajectory_points_list[:,0,:].min(axis=0)
            print(min_values)
            max_values = trajectory_points_list[:,0,:].max(axis=0) 
            print(max_values)

            # Print trajectory details
            print(f"Generated trajectory with {len(trajectory)} points.")
            print(f"Trajectory dtype: {trajectory.dtype}")
            # trajectory_output = []
            # for pos, orientation_vec in trajectory:
            #                 # Prevent warning from align_vectors when vectors are already aligned
            #     if np.allclose(orientation_vec, [0, 0, 1], atol=1e-6):
            #         euler_deg = (0.0, 0.0, 0.0)
            #     else:
            #         euler_deg = tuple(R.from_rotvec(rotation.as_rotvec()).as_euler('xyz', degrees=True))

            #     pos_tuple = tuple(pos / 1000)
            #     trajectory_output.append((pos_tuple, euler_deg))
            # self.last_cross_point_count = len(cross_points)
            # self.trajectory_points_list = trajectory_output


            trajectory_output = []
            cross_output = []
            convex_output = []

            for idx, (pos, orientation_vec) in enumerate(trajectory):
                if np.allclose(orientation_vec, [0, 0, 1], atol=1e-6):
                    euler_rad = (0.0, 0.0, 0.0)
                else:
                    euler_rad = tuple(fixed_plane_orientation_rpy)
                pos_tuple = tuple(pos / 1000)

                trajectory_output.append((pos_tuple, euler_rad))

                if idx < len(cross_points):  # Assuming cross is first
                    cross_output.append((pos_tuple, euler_rad))
                else:
                    convex_output.append((pos_tuple, euler_rad))

            self.trajectory_points_list = trajectory_output
            self.cross_trajectory = cross_output
            self.convex_trajectory = convex_output

            # print("Last cross point:", cross_output[-1][0])
            # print("First convex point:", convex_output[0][0])

            
        
            # print(f"Debug ", cross_output[0],"/n" , self.cross_trajectory[0])
            # print(f"Debug ",convex_output[0],"/n",self.convex_trajectory[0])
            # print(f"Debug ",trajectory_output[0],"/n",self.trajectory_points_list[0])
            # self.cross_trajectory = cross_points # 1 initial + N cross
            # self.convex_trajectory = convex_hull_points # Remaining
            # self.cross_trajectory.tolist()
            # self.convex_trajectory.tolist()
            # Save trajectory as numpy for debugging
            np.save("/mnt/c/Users/aggel/Desktop/planned_trajectory_positions.npy", [p for p, _ in self.trajectory_points_list])
            np.save("/mnt/c/Users/aggel/Desktop/planned_CROSS_positions.npy", [p for p, _ in self.cross_trajectory])
            np.save("/mnt/c/Users/aggel/Desktop/planned_CONVEX_positions.npy", [p for p, _ in self.convex_trajectory])


            return self.trajectory_points_list 
        else:
            print("❌ No convex hull points generated.")
            self.trajectory_points_list = None
            return None
    
    def do_everything_scara(self,input_dae_path,output_dae_path,manual_offset = np.array([0.0,0.0,0.0]), fixed_plane_orientation_rpy=np.array([0.0, 0.0, 0.0])):
        
        self.shift_dae_file_vertices(
            input_dae_path=input_dae_path,
            output_dae_path=output_dae_path,
            base_plane_origin=np.array([0.0, 0.0, 0.0]),
            plane_rotation_deg=0  ,
            manual_offset=manual_offset)
        
        path_to_shifted_dae = os.path.expanduser(output_dae_path)
        convex_hull_points = self.compute_convex_hull_from_dae(path_to_shifted_dae) 
        
        # # Define the fixed orientation vector (e.g., -X axis)
        # fixed_orientation_vector = np.array([-1, 0, 0], dtype=float)
        # # Rotate it 45 degrees around Y-axis to align with the inclined plane
        # rotation = R.from_euler('y', 0, degrees=True)
        # rotated_orientation_vector = rotation.apply(fixed_orientation_vector)
        rotated_orientation_vector = fixed_plane_orientation_rpy



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

        
            convex_hull_points_trajectory = self.extract_outermost_ring(points_array,projection='yz',max_point_spacing=5)
            trajectory = self.compute_total_trajectory_for_robot_SCARA(convex_hull_points=convex_hull_points_trajectory,
                                                        cross_points=cross_points,
                                                        initial_position=initial_position
                                                        )
            
            # Set numpy to print floats in the usual format, without scientific notation
            np.set_printoptions(precision=8, suppress=True)

            # Scale only the position part (index 0 of each tuple) by 1/1000
            trajectory_scaled = [(pos , orientation) for (pos, orientation) in trajectory]


            trajectory_points_list = np.array(trajectory_scaled)

            # print(trajectory_points_list[0:10],'pre devision')

            # Divide only the first column (x values) by 100
            # trajectory_points_list[:, 0,:] = trajectory_points_list[:, 0,:] / 100
            trajectory_points_list[:, 0:1, :] = trajectory_points_list[:, 0:1, :] /1000  # scale all positions
            # 🔧 Shift the trajectory to be in a reachable region
            # trajectory_points_list[:, 0:1, 0] += 0.25   # X → front
            # trajectory_points_list[:, 0:1, 1] -= 0.25   # Y → right
            # trajectory_points_list[:, 0:1, 2] += 0.15   # Z → higher
            #             # 🔁 Rotate trajectory positions to match tool orientation
            # rotation = R.from_euler('y', 45, degrees=True)
            # trajectory_points_list[:, 0, :] = rotation.apply(trajectory_points_list[:, 0, :])

            # print(trajectory_points_list[0:10],'post devision')
            # min_values = trajectory_points_list[:,0,:].min(axis=0)
            # print(min_values)
            # max_values = trajectory_points_list[:,0,:].max(axis=0) 
            # print(max_values)

            # # Print trajectory details
            # print(f"Generated trajectory with {len(trajectory)} points.")
            # print(f"Trajectory dtype: {trajectory.dtype}")
            # trajectory_output = []
            # for pos, orientation_vec in trajectory:
            #                 # Prevent warning from align_vectors when vectors are already aligned
            #     if np.allclose(orientation_vec, [0, 0, 1], atol=1e-6):
            #         euler_deg = (0.0, 0.0, 0.0)
            #     else:
            #         euler_deg = tuple(R.from_rotvec(rotation.as_rotvec()).as_euler('xyz', degrees=True))

            #     pos_tuple = tuple(pos / 1000)
            #     trajectory_output.append((pos_tuple, euler_deg))
            # self.last_cross_point_count = len(cross_points)
            # self.trajectory_points_list = trajectory_output


            trajectory_output = []
            cross_output = []
            convex_output = []

            for idx, (pos, orientation_vec) in enumerate(trajectory_scaled):
                if np.allclose(orientation_vec, [0, 0, 1], atol=1e-6):
                    euler_rad = (0.0, 0.0, 0.0)
                else:
                    euler_rad = tuple(fixed_plane_orientation_rpy)

                pos_tuple = tuple(pos / 1000)

                trajectory_output.append((pos_tuple, euler_rad))

                if idx < len(cross_points):  # Assuming cross is first
                    cross_output.append((pos_tuple, euler_rad))
                else:
                    convex_output.append((pos_tuple, euler_rad))

            self.trajectory_points_list = trajectory_output
            self.cross_trajectory = cross_output
            self.convex_trajectory = convex_output
            # print("Last cross point:", cross_output[-1][0])
            # print("First convex point:", convex_output[0][0])

            
        
            # print(f"Debug ", cross_output[0],"/n" , self.cross_trajectory[0])
            # print(f"Debug ",convex_output[0],"/n",self.convex_trajectory[0])
            # print(f"Debug ",trajectory_output[0],"/n",self.trajectory_points_list[0])
            # self.cross_trajectory = cross_points # 1 initial + N cross
            # self.convex_trajectory = convex_hull_points # Remaining
            # self.cross_trajectory.tolist()
            # self.convex_trajectory.tolist()
            # Save trajectory as numpy for debugging
            np.save("/mnt/c/Users/aggel/Desktop/planned_trajectory_positions.npy", [p for p, _ in self.trajectory_points_list])
            np.save("/mnt/c/Users/aggel/Desktop/planned_CROSS_positions.npy", [p for p, _ in self.cross_trajectory])
            np.save("/mnt/c/Users/aggel/Desktop/planned_CONVEX_positions.npy", [p for p, _ in self.convex_trajectory])


            return self.trajectory_points_list 
        else:
            print("❌ No convex hull points generated.")
            self.trajectory_points_list = None
            return None
    
    def do_everything_cross_algorithm(self,input_dae_path,output_dae_path,manual_offset = np.array([0.0,0.0,0.0]),fixed_plane_orientation_rpy=np.array([0.0, 0.0, 0.0])):
        #  Define your reachable workspace bounds
        min_bounds = np.array([-0.32500893, -0.14676859,  0.009])
        max_bounds = np.array([ 0.66500893,  0.74676859,  0.68019031])

        dae_path = os.path.expanduser("~/experiment/robotic_manipulators_playground/mesh_mask_scaled_3x.dae")
        # points = extract_points_from_dae(dae_path)
        # shifted_points = self.shift_points_to_fit_workspace(points, min_bounds, max_bounds)
       
        # points = self.extract_points_from_dae(output_dae_path)
        # # print("THis is the first",points[0])
        # points /= 100

        # base_origin, offset = self.compute_optimal_transform(points, rotation_deg=45)
        # points = self.extract_points_from_dae(input_dae_path)
        # base_origin, offset = self.compute_optimal_transform(points, rotation_deg=45)

        # self.shift_dae_file_vertices(
        #     input_dae_path=input_dae_path,
        #     output_dae_path=output_dae_path,
        #     base_plane_origin=np.array([32.0, 0.0, 51.0]),
        #     plane_rotation_deg=45,
        #     manual_offset=manual_offset)
        self.shift_dae_file_vertices(
            input_dae_path=input_dae_path,
            output_dae_path=output_dae_path,
            base_plane_origin=np.array([0.0, 0.0, 0.0]),
            plane_rotation_deg=0)
        
        path_to_shifted_dae = os.path.expanduser(output_dae_path)
        convex_hull_points = self.compute_convex_hull_from_dae(path_to_shifted_dae) 
        
        # Define the fixed orientation vector (e.g., -X axis)
        # fixed_orientation_vector = np.array([-1, 0, 0], dtype=float)
        # # Rotate it 45 degrees around Y-axis to align with the inclined plane
        # rotation = R.from_euler('y', 45, degrees=True)
        # rotated_orientation_vector = rotation.apply(fixed_orientation_vector)
          # rotated_orientation_vector = rotation.apply(fixed_orientation_vector)
        rotated_orientation_vector = fixed_plane_orientation_rpy




        if convex_hull_points is not None:
            print("🔺 Original Convex Hull Points:")
            # transformed_points = project_points_to_workspace(convex_hull_points, max_radius=0.8)
            transformed_points = convex_hull_points
            points_array = np.array(transformed_points)
            # print(len(transformed_points))

            initial_position, orientation = self.calculate_initial_position_3d(points_array, orientation= rotated_orientation_vector)
            # Compute cross trajectory
            cross_points = self.compute_cross_trajectory_3d(initial_position.flatten(), points_array)
            # print("Cross points are : " ,len(cross_points))
            # print("Cross points are : " ,cross_points)
            
            # points = self.compute_convex_hull_trajectory_3d(path_to_shifted_dae, max_radius=0.8, fixed_orientation=fixed_orientation, max_point_spacing=0.01)

        # # 🔧 FIX: Extract just the positions
        # trajectory_positions = [points[i][0] for i in range(len(points)) if i % 10 == 0]
    
        # # # Step 3: Get pre-position movement
        # # initial_position,orientation= calculate_initial_position_3d(points, orientation=[0, 0, 0])
        # initial_point = initial_position.reshape(1, 3)
    
        
            convex_hull_points_trajectory = self.extract_outermost_ring(points_array,projection='yz',max_point_spacing=5)
            trajectory = self.compute_total_trajectory_for_robot_cross_algorithm(convex_hull_points=convex_hull_points_trajectory,
                                                        cross_points=cross_points,
                                                        initial_position=initial_position
                                                        )
            
            # Set numpy to print floats in the usual format, without scientific notation
            np.set_printoptions(precision=8, suppress=True)

            trajectory_points_list = np.array(trajectory)

            # print(trajectory_points_list[0:10],'pre devision')

            # Divide only the first column (x values) by 100
            # trajectory_points_list[:, 0,:] = trajectory_points_list[:, 0,:] / 100
            trajectory_points_list[:, 0:1, :] = trajectory_points_list[:, 0:1, :] / 1000 # scale all positions
            # 🔧 Shift the trajectory to be in a reachable region
            # trajectory_points_list[:, 0:1, 0] += 0.25   # X → front
            trajectory_points_list[:, 0:1, 1] -= 0.25   # Y → right
            # trajectory_points_list[:, 0:1, 2] += 0.15   # Z → higher
            #             # 🔁 Rotate trajectory positions to match tool orientation
            # rotation = R.from_euler('y', 45, degrees=True)
            # trajectory_points_list[:, 0, :] = rotation.apply(trajectory_points_list[:, 0, :])

            # print(trajectory_points_list[0:10],'post devision')
            min_values = trajectory_points_list[:,0,:].min(axis=0)
            print(min_values)
            max_values = trajectory_points_list[:,0,:].max(axis=0) 
            print(max_values)

            # Print trajectory details
            print(f"Generated trajectory with {len(trajectory)} points.")
            print(f"Trajectory dtype: {trajectory.dtype}")
            # trajectory_output = []
            # for pos, orientation_vec in trajectory:
            #                 # Prevent warning from align_vectors when vectors are already aligned
            #     if np.allclose(orientation_vec, [0, 0, 1], atol=1e-6):
            #         euler_deg = (0.0, 0.0, 0.0)
            #     else:
            #         euler_deg = tuple(R.from_rotvec(rotation.as_rotvec()).as_euler('xyz', degrees=True))

            #     pos_tuple = tuple(pos / 1000)
            #     trajectory_output.append((pos_tuple, euler_deg))
            # self.last_cross_point_count = len(cross_points)
            # self.trajectory_points_list = trajectory_output


            trajectory_output = []
            cross_output = []
            convex_output = []

            for idx, (pos, orientation_vec) in enumerate(trajectory):
                if np.allclose(orientation_vec, [0, 0, 1], atol=1e-6):
                    euler_rad = (0.0, 0.0, 0.0)
                else:
                    euler_rad = tuple(fixed_plane_orientation_rpy)
                pos_tuple = tuple(pos / 1000)

                trajectory_output.append((pos_tuple, euler_rad))

                if idx < len(cross_points):  # Assuming cross is first
                    cross_output.append((pos_tuple, euler_rad))
                else:
                    convex_output.append((pos_tuple, euler_rad))

            self.trajectory_points_list = trajectory_output
            self.cross_trajectory = cross_output
            self.convex_trajectory = convex_output

            # print("Last cross point:", cross_output[-1][0])
            # print("First convex point:", convex_output[0][0])

            
        
            # print(f"Debug ", cross_output[0],"/n" , self.cross_trajectory[0])
            # print(f"Debug ",convex_output[0],"/n",self.convex_trajectory[0])
            # print(f"Debug ",trajectory_output[0],"/n",self.trajectory_points_list[0])
            # self.cross_trajectory = cross_points # 1 initial + N cross
            # self.convex_trajectory = convex_hull_points # Remaining
            # self.cross_trajectory.tolist()
            # self.convex_trajectory.tolist()
            # Save trajectory as numpy for debugging
            np.save("/mnt/c/Users/aggel/Desktop/planned_trajectory_positions.npy", [p for p, _ in self.trajectory_points_list])
            np.save("/mnt/c/Users/aggel/Desktop/planned_CROSS_positions.npy", [p for p, _ in self.cross_trajectory])
            np.save("/mnt/c/Users/aggel/Desktop/planned_CONVEX_positions.npy", [p for p, _ in self.convex_trajectory])


            return self.trajectory_points_list 
        else:
            print("❌ No convex hull points generated.")
            self.trajectory_points_list = None
            return None
    
    def visualize_points(self, points):
        # Load workspace points from txt
        points_txt = pd.read_csv("reachable_workspace.txt").values

        # Extract trajectory positions
        trajectory_positions = np.array([pos for pos, _ in points])
        

        # # Align workspace points by centroids
        # centroid_txt = points_txt.mean(axis=0)
        # centroid_traj = trajectory_positions.mean(axis=0)
        # translation = centroid_traj - centroid_txt
        # aligned_points = points_txt + translation
        # Align using Z-min instead of centroid
        z_offset = trajectory_positions[:, 2].min() - points_txt[:, 2].min()
        translation = np.array([0.0, 0.0, z_offset])
        aligned_points = points_txt #
        x_vals, y_vals, z_vals = zip(*trajectory_positions)
        # ===================== 🔷 PLOT 1: Comparison =====================
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')
        
        ax1.plot(trajectory_positions[:, 0], trajectory_positions[:, 1], trajectory_positions[:, 2],
                'o-', label='Trajectory', color='blue')

        # ax1.plot(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2],
        #         'x-', label='Aligned Workspace', color='orange')
        ax1.scatter(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2],
            label='Aligned Workspace', color='orange', marker='x')

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_zlim(0.0, 0.75)  # Match the bounding box height

        ax1.set_title("Trajectory vs. Aligned Workspace")
        ax1.legend()

      
        # ===================== 🔷 PLOT 2: Final Workspace with markers =====================
        # self.trajectory_points_list
        final_positions = [pos for pos, _ in points]
        x_vals, y_vals, z_vals = zip(*final_positions)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        # ax2.plot(x_vals, y_vals, z_vals, marker='o')
        ax2.scatter(x_vals, y_vals, z_vals, marker='o', color='blue', label='Trajectory Points')


        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title("Final Trajectory + Workspace Bounds")

        # Draw bounding box
        x_range = [-0.495, 0.495]
        y_range = [-0.447, 0.447]
        z_range = [0.009, 0.680]
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
        edges = [
            (0, 1), (0, 2), (0, 4),
            (3, 1), (3, 2), (3, 7),
            (5, 1), (5, 4), (5, 7),
            (6, 2), (6, 4), (6, 7)
        ]
        for start, end in edges:
            xs = [corners[start][0], corners[end][0]]
            ys = [corners[start][1], corners[end][1]]
            zs = [corners[start][2], corners[end][2]]
            ax2.plot(xs, ys, zs, color='gray', linestyle='--')

        # Red marker reference points
        highlight_points = [
            (0.0, 0.0, 0.000),
            (0.0, 0.0, 0.724),
        ]
        for point in highlight_points:
            ax2.scatter(*point, color='red', s=50, label=f'{point}')

        ax2.legend()
        plt.tight_layout()
        plt.show()

    def visualize_points_scara(self, points):
        # Load workspace points from txt
        points_txt = pd.read_csv("reachable_workspace.txt").values

        # Extract trajectory positions
        trajectory_positions = np.array([pos for pos, _ in points])
        

        # # Align workspace points by centroids
        # centroid_txt = points_txt.mean(axis=0)
        # centroid_traj = trajectory_positions.mean(axis=0)
        # translation = centroid_traj - centroid_txt
        # aligned_points = points_txt + translation
        # Align using Z-min instead of centroid
        z_offset = trajectory_positions[:, 2].min() - points_txt[:, 2].min()
        translation = np.array([0.0, 0.0, z_offset])
        aligned_points = points_txt #
        x_vals, y_vals, z_vals = zip(*trajectory_positions)
        # ===================== 🔷 PLOT 1: Comparison =====================
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')
        
        ax1.plot(trajectory_positions[:, 0], trajectory_positions[:, 1], trajectory_positions[:, 2],
                'o-', label='Trajectory', color='blue')

        # ax1.plot(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2],
        #         'x-', label='Aligned Workspace', color='orange')
        ax1.scatter(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2],
            label='Aligned Workspace', color='orange', marker='x')

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_zlim(0.0, 0.75)  # Match the bounding box height

        ax1.set_title("Trajectory vs. Aligned Workspace")
        ax1.legend()

      
        # ===================== 🔷 PLOT 2: Final Workspace with markers =====================
        # self.trajectory_points_list
        final_positions = [pos for pos, _ in points]
        x_vals, y_vals, z_vals = zip(*final_positions)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        # ax2.plot(x_vals, y_vals, z_vals, marker='o')
        ax2.scatter(x_vals, y_vals, z_vals, marker='o', color='blue', label='Trajectory Points')


        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title("Final Trajectory + Workspace Bounds")

        # Draw bounding box
        x_range = [-0.200, 0.400]
        y_range = [-0.146, 0.546]
        z_range = [0.000, 0.300]
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
        edges = [
            (0, 1), (0, 2), (0, 4),
            (3, 1), (3, 2), (3, 7),
            (5, 1), (5, 4), (5, 7),
            (6, 2), (6, 4), (6, 7)
        ]
        for start, end in edges:
            xs = [corners[start][0], corners[end][0]]
            ys = [corners[start][1], corners[end][1]]
            zs = [corners[start][2], corners[end][2]]
            ax2.plot(xs, ys, zs, color='gray', linestyle='--')

        # Red marker reference points
        highlight_points = [
            (0.0, 0.0, 0.000),
            (0.0, 0.0, 0.724),
        ]
        for point in highlight_points:
            ax2.scatter(*point, color='red', s=50, label=f'{point}')

        ax2.legend()
        plt.tight_layout()
        plt.show()

            
       
if __name__ == "__main__":
    # Define workspace bounds and dae_path
    min_bounds = np.array([-0.495, -0.447, 0.009])
    max_bounds = np.array([0.495, 0.447, 0.680])
    input_dae_path = os.path.expanduser("~/experiment/robotic_manipulators_playground/mesh_mask_scaled_3x.dae")
    output_dae_path = os.path.expanduser("~/experiment/robotic_manipulators_playground/shifted_object_scaled_3x.dae")
    # Instantiate the RobotTrajectory class
    robot_trajectory = RobotTrajectory(input_dae_path, min_bounds, max_bounds)

    # Call the 'do_everything' method to compute the trajectory
    offset = [np.array([300.0, 295.0, 300.0])]
        
    # robot_trajectory.do_everything(input_dae_path,output_dae_path,manual_offset=offset)
    robot_trajectory.do_everything_cross_algorithm(input_dae_path,output_dae_path)
    # Print the first 5 and last 5 characters of the input DAE path
    # print(f"Input DAE path (first 5 characters): {input_dae_path[:5]}")
    # print(f"Input DAE path (last 5 characters): {input_dae_path[-5:]}")

    # # Print the first 5 and last 5 characters of the output DAE path
    # print(f"Output DAE path (first 5 characters): {output_dae_path[:5]}")
    # print(f"Output DAE path (last 5 characters): {output_dae_path[-5:]}")

    # robot_trajectory.do_everything_scara(input_dae_path,output_dae_path,manual_offset =np.array([200.0, 300.0, -200.0]) )
    # Retrieve the computed trajectory points list
    trajectory_points = robot_trajectory.trajectory_points_list
    if len(trajectory_points)>0:
        print(f"Computed trajectory with {len(trajectory_points)} points.")
        print(trajectory_points[8])
    else:
        print("No trajectory points available.convex")
    convex_points = robot_trajectory.convex_trajectory
    cross_points = robot_trajectory.cross_trajectory
    # print(trajectory_points)
    
    # robot_trajectory.visualize_points_scara(trajectory_points)
    robot_trajectory.visualize_points(trajectory_points)
    # robot_trajectory.visualize_points(convex_points)
    # robot_trajectory.visualize_points(cross_points)



# if __name__ == "__main__":
#     min_bounds = np.array([-0.32500893, -0.14676859, 0.009])
#     max_bounds = np.array([0.66500893, 0.74676859, 0.68019031])
#     input_dae_path = os.path.expanduser("~/experiment/robotic_manipulators_playground/mesh_mask.dae")
#     output_dae_path = os.path.expanduser("~/experiment/robotic_manipulators_playground/shifted_object.dae")

#     robot_trajectory = RobotTrajectory(input_dae_path, min_bounds, max_bounds)
#     robot_trajectory.do_everything(input_dae_path, output_dae_path)
    
#     trajectory_points = robot_trajectory.trajectory_points_list
#     print(trajectory_points)
#     if trajectory_points is not None and len(trajectory_points) > 0:
#         print(f"Computed trajectory with {len(trajectory_points)} points.")
        
#         robot_trajectory.visualize_points(trajectory_points)
#     else:
#         print("No trajectory points available.")
