import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial import ConvexHull
import pyvista as pv
from scipy.spatial.transform import Rotation as R

def extract_points_from_dae(dae_path):
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
            for i in range(0, len(floats), 3000): 
                point = floats[i:i+3]
                if len(point) == 3:
                    all_points.append(point)

        return np.array(all_points)

    except Exception as e:
        print(f"❌ Error reading .dae file: {e}")
        return None

def compute_convex_hull_from_dae(dae_path):
    points = extract_points_from_dae(dae_path)

    if points is None or len(points) < 4:
        print("⚠️ Not enough points to compute a convex hull.")
        return None

    print(f"✅ Loaded {len(points)} points from file.")
    hull = ConvexHull(points)
    convex_hull_points = points[hull.vertices]
    return convex_hull_points

def interpolate_line_segment(p1, p2, max_point_spacing=0.01):
    """
    Interpolate between p1 and p2 with dynamic resolution based on distance.
    max_point_spacing defines the max distance between two points.
    """
    distance = np.linalg.norm(p2 - p1)
    num_points = max(int(distance / max_point_spacing), 2)  # at least 2 points
    # print([tuple(p1 + (p2 - p1) * t) for t in np.linspace(0, 1, num_points)])

    return [tuple(p1 + (p2 - p1) * t) for t in np.linspace(0, 1, num_points)]

def compute_convex_hull_trajectory(dae_path, max_radius=0.8, fixed_orientation=(0, 0, 0), max_point_spacing=0.01):
    points = extract_points_from_dae(dae_path)
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
        segment = interpolate_line_segment(p1, p2, max_point_spacing)
        for pos in segment:
            trajectory.append((pos, fixed_orientation))

    # Optional: close the loop
    p1 = np.array(projected_points[-1])
    p2 = np.array(projected_points[0])
    segment = interpolate_line_segment(p1, p2, max_point_spacing)
    for pos in segment:
        trajectory.append((pos, fixed_orientation))
    
    return trajectory

def compute_convex_hull_trajectory_3d(dae_path, max_radius=0.8, fixed_orientation=(0, 0, 0), max_point_spacing=0.01):
    points = extract_points_from_dae(dae_path)
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
        segment = interpolate_line_segment(p1, p2, max_point_spacing)
        for pos in segment:
            trajectory.append((pos, fixed_orientation))

    # Optional: close the loop
    p1 = np.array(projected_points[-1])
    p2 = np.array(projected_points[0])
    segment = interpolate_line_segment(p1, p2, max_point_spacing)
    for pos in segment:
        trajectory.append((pos, fixed_orientation))

    # Convert trajectory to np.array with 3D points
    trajectory_np = np.array([pos for pos, _ in trajectory])

    # Reshape to have 3 dimensions (N, 1, 3)
    trajectory_3d = trajectory_np.reshape((-1, 1, 3))

    return trajectory_3d

def calculate_initial_position(points, orientation=np.array([0, 0, 0]), z_offset=0.01):
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

def calculate_initial_position_3d(points, orientation=np.array([0, 0, 0]), z_offset=0.01):
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
    centroid_xy = np.mean(points[:, :2], axis=0).flatten()

    # Max Z (highest point)
    max_z = np.max(points[:, 2])
    safe_z = max_z + z_offset

    # Ensure scalars and create position
    initial_pos = np.array([float(centroid_xy[0]), float(centroid_xy[1]), float(safe_z)])

    # Reshape to have 3 dimensions (1, 3)
    initial_pos_3d = initial_pos.reshape((1, 1, 3))

    # Reshape orientation to (1, 1, 3)
    orientation_3d = np.array(orientation, dtype=float).reshape((1, 1, 3))

    return initial_pos_3d, orientation_3d

def compute_cross_trajectory(initial_position,points, line_spacing=0.01):
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

def compute_cross_trajectory_3d(initial_position, points, line_spacing=0.01):
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

    # Reshape to have 3 dimensions (N, 1, 3)
    cross_trajectory_3d = cross_trajectory.reshape((-1, 1, 3))

    return cross_trajectory_3d


def project_points_to_workspace(points, max_radius=0.8, offset=np.array([0, 0, 0])):
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

if __name__ == "__main__":
    path_to_dae = "c:/Users/dtsits/aggelos-python/11803_Airplane_v1_l1.dae"
    convex_hull_points = compute_convex_hull_from_dae(path_to_dae)

    if convex_hull_points is not None:
        print("🔺 Original Convex Hull Points:")
        # for point in convex_hull_points:
        #     print(tuple(point))

        # ✅ Project into robot workspace
        transformed_points = project_points_to_workspace(convex_hull_points, max_radius=0.8)
        points = compute_convex_hull_trajectory_3d(path_to_dae, max_radius=0.8, fixed_orientation=(0, 0, 0), max_point_spacing=0.01)

    # 🔧 FIX: Extract just the positions
    # trajectory_positions = [pos for i, (pos, _) in enumerate(points) if i % 10 == 0]
          
    # # Step 3: Get pre-position movement
    initial_position,orientation= calculate_initial_position_3d(points, orientation=[0, 0, 0])
    initial_point = initial_position.reshape(1, 3)
    cross_points = compute_cross_trajectory_3d(initial_position,points)
    # # Step 4: Combine into full trajectory
    points_array = np.array(points)
    # full_trajectory = [initial_position] + cross_points + points_array
    
    # full_trajectory_list = full_trajectory.tolist()

    print("Debugging Point Print")
    print(len(initial_position),type(initial_position))
    print(len(points),type(points))
    print(len(cross_points),type(cross_points),np.array(cross_points).shape)
    print(len(points_array),type(points_array),np.array(points_array).shape)
    # print(len(full_trajectory_list),type(full_trajectory_list))
    # print(len(full_trajectory),type(full_trajectory))
    
    

    # print(len(full_trajectory))
    # print("🔍 initial_position =", initial_position)
    # print("🔍 type(initial_position[0]) =", type(initial_position[0]))
    # print("🔍 initial_position[0] =", initial_position[0])

   

    # # Visualize with PyVista
    # trajectory_np = np.array(trajectory_positions)
   
    # print("🧮 Trajectory shape:", trajectory_np.shape)
    
    # plotter2 = pv.Plotter()

    # visualization_points  = extract_points_from_dae("c:/Users/dtsits/aggelos-python/11803_Airplane_v1_l1.dae")
    # if visualization_points.shape[0] > 1:
    #     polyline = pv.lines_from_points(visualization_points)
    #     plotter2.add_mesh(polyline, color='blue', line_width=3, label="Trajectory Path")
    #     plotter2.add_points(visualization_points, color='red', point_size=8, label="Trajectory Points")

  
    # # Your fixed orientation (roll, pitch, yaw) in radians
    # euler_orientation = (0, 0, 0)
    # rotation = R.from_euler('xyz', euler_orientation)
    # direction_vector = rotation.apply([0, 0, -1])  # Z-axis as end-effector forward direction

    # # Scale the arrow for visual clarity
    # arrow_length = 100

    # subsampled_positions = np.array(trajectory_positions[::500])
    # arrow_points = pv.PolyData(subsampled_positions)
    # arrow_template = pv.Arrow(direction=direction_vector, scale=arrow_length)
    # glyphs = arrow_points.glyph(orient=False, scale=False, geom=arrow_template)
    # plotter2.add_mesh(glyphs, color='green', label="EE Orientation")   
    # plotter2.add_points(initial_point, color='black', point_size=12, label="Initial Pose")
    # plotter2.add_points(cross_points, color='orange', point_size=5, label="Cross Trajectory")


    # # Add coordinate axes labels
    # plotter2.add_axes(line_width=2)
    # plotter2.add_legend()

    # # Optional: set view and scaling
    # plotter2.view_vector((1, 1, 1))      # View from diagonal
    # plotter2.set_scale(zscale=1.0, yscale=1.0, xscale=1.0)  # Optional: Adjust scales if needed

    # # Show the final plot
    # plotter2.show(title="Convex Hull Trajectory in 3D")

    # # Compute convex hull
    # hull = ConvexHull(points)
    # faces = hull.simplices  # Indices of vertices forming the convex hull triangles

    # # Create PyVista mesh from points and faces  
    # mesh_faces = []
    # for tri in faces:
    #     mesh_faces.append(3)         # 3 vertices per face
    #     mesh_faces.extend(tri)       # Add triangle vertex indices

    # mesh = pv.PolyData(points, mesh_faces)

    #     # 🔵 Build line segments for convex hull edges
    # lines = set()
    # for simplex in hull.simplices:
    #     for i in range(3):
    #         edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
    #         lines.add(edge)

    # edge_lines = []
    # for i, j in lines:
    #     edge_lines.append(points[i])
    #     edge_lines.append(points[j])

    # edge_lines_np = np.array(edge_lines)
    # hull_edges = pv.lines_from_points(edge_lines_np, close=False)

    # # # # Plot
    # # plotter3 = pv.Plotter()
    # # plotter3.add_mesh(mesh, color='lightblue', opacity=0.5, show_edges=False, label="Convex Hull Surface")
    # # plotter3.add_mesh(hull_edges, color='blue', line_width=2, label="Convex Hull Edges")
    # # plotter3.add_points(points, color='red', point_size=5, label="Original Points")
    # # plotter3.add_axes()
    # # plotter3.add_legend()
    # # plotter3.view_vector((1, 1, 1))
    # # plotter3.show(title="3D Convex Hull Mesh with Edges")
    

    # # # Plot
    # # plotter3 = pv.Plotter()
    # # plotter3.add_mesh(mesh, color='lightblue', opacity=0.5, show_edges=True, label="Convex Hull")
    # # plotter3.add_points(points, color='red', point_size=5, label="Original Points")
    # # plotter3.add_axes()
    # # plotter3.add_legend()
    # # plotter3.view_vector((1, 1, 1))
    # # plotter3.show(title="3D Convex Hull Mesh from DAE") 