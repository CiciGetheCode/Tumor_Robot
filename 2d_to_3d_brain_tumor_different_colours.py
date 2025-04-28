import cv2
import math
import numpy as np
import os
from tqdm import tqdm
from skimage import measure
import pyvista as pv
from PIL import Image
from datasets import load_dataset
import natsort
from natsort import natsorted
import matplotlib.pyplot as plt



'''

# def load_tiff_images_as_slices(folder):
#     # List all files in the folder
#     files = sorted([f for f in os.listdir(folder) if f.endswith('.tif')])

#     # Load each image slice into a list
#     slices = []
#     mask_slices=[]
#     for file in tqdm(files, desc="Loading slices", unit="file"):
#         image_path = os.path.join(folder, file)
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         if image is not None:
#             slices.append(image)
#             if image_path.__contains__("mask"):
#                 image_path_mask = image_path 
#                 image_mask = cv2.imread(image_path_mask,cv2.IMREAD_GRAYSCALE)
#                 mask_slices.append(image_mask)
                
#         else:
#             tqdm.write(f"Error loading {file}")  # Use tqdm.write() to print error messages without disrupting the progress bar
    
#     # print(np.array(mask_slices[0]))
#     # print(np.array(mask_slices[5]))
    
#     return np.array(slices) , np.array(mask_slices)


# def visualize_3d_mesh_with_gap(verts, faces, gap=0):
#     """
#     Visualizes the 3D mesh using PyVista with a small gap between slices.

#     Parameters:
#     - verts: Vertices of the 3D mesh.
#     - faces: Faces of the 3D mesh.
#     - gap: Gap size between slices (default is 0).
#     """
#     # Convert to numpy array if not already
#     verts = np.array(verts)
#     faces = np.array(faces)

#     # Ensure faces are reshaped correctly
#     if faces.ndim == 1:
#         faces = faces.reshape((-1, 4))  # Reshape assuming triangular faces

#     # Determine slice thickness (example uses number of vertices to divide slices)
#     slice_thickness = len(verts) // 10  # Change this value based on your data

#     # List to hold all the slices separately
#     all_meshes = []

#     # Iterate through slices and apply gap
#     for i in range(0, len(verts), slice_thickness):
#         # Select the vertices for the current slice
#         slice_verts = verts[i:i + slice_thickness].copy()

#         # Apply the gap to shift this slice along the x-axis
#         slice_verts[:, 0] += (i // slice_thickness) * gap

#         # Filter the faces that belong to the current slice
#         slice_faces = []
#         for face in faces:
#             # Ensure the face references the correct vertices
#             if all(i <= idx < i + slice_thickness for idx in face[1:]):
#                 adjusted_face = [face[0]] + [(idx - i) for idx in face[1:]]
#                 slice_faces.extend(adjusted_face)

#         # Convert the slice_faces list to a numpy array with the correct shape
#         slice_faces = np.array(slice_faces).reshape((-1, 4))  # Reshape based on triangular faces

#         # Create a mesh only if the slice has valid faces
#         if len(slice_faces) > 0:
#             mesh = pv.PolyData(slice_verts, slice_faces)
#             all_meshes.append(mesh)

#     # Visualize all the separated slices with gaps
#     plotter = pv.Plotter()
#     for mesh in tqdm(all_meshes, desc="Adding slices to plot"):
#         plotter.add_mesh(mesh, color='lightblue', show_edges=True, smooth_shading=True)

#     plotter.show()

# def visualize_slice(data, slice_index):
#     """
#     Visualizes a single 2D slice from a 3D numpy array.

#     Parameters:
#     - data: 3D NumPy array containing the image slices.
#     - slice_index: The index of the slice to visualize (along the z-axis).
#     """
#     # Get the dimensions of the 3D array
#     depth = data.shape[0]
    
#     # Ensure the slice index is within the valid range
#     if slice_index < 0 or slice_index >= depth:
#         print(f"Error: slice_index {slice_index} is out of range. Must be between 0 and {depth - 1}.")
#         return
    
#     # Extract the 2D slice at the specified index
#     slice_data = data[slice_index]
    
#     # Plot the slice
#     plt.figure(figsize=(6, 6))
#     plt.imshow(slice_data, cmap='gray')  # Use 'gray' colormap for grayscale images
#     plt.title(f"Slice {slice_index}")
#     plt.axis('off')  # Hide the axis for a cleaner look
#     plt.show()



def center_point_cloud(arr):
    """
    Centers the point cloud around the origin.
    """
    centroid = np.mean(arr, axis=0)
    return arr - centroid


# def load_tiff_images_as_slices(folder):
#     # List all files in the folder
#     files = sorted([f for f in os.listdir(folder) if f.endswith('.tif')])

#     # Load each image slice into lists
#     slices = []
#     mask_slices = []

#     for file in tqdm(files, desc="Loading slices", unit="file"):
#         image_path = os.path.join(folder, file)
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#         if image is not None:
#             # Add the image to slices or mask_slices based on filename
#             if "mask" in file.lower():  # Better to check in lowercase for consistent matching
#                 mask_slices.append(image)
#             else:
#                 slices.append(image)
#         else:
#             tqdm.write(f"Error loading {file}")  # Use tqdm.write to avoid progress bar disruption

#     if slices:
#         print("First slice sample:\n", np.array(slices[0]))  # Debug: print first slice data

#     return np.array(slices), np.array(mask_slices)


# def convert_2d_slice_to_nparray(folder):
#     """
#     Converts a folder of 2D image slices into a 3D numpy array.

#     Parameters:
#     - folder: The folder containing 2D slices in .tif format.

#     Returns:
#     - A 3D numpy array with the stacked image slices.
#     """
#     # List all .tif files in the folder and sort them
#     files = sorted([f for f in os.listdir(folder) if f.endswith('.tif')])

#     # Initialize an empty list to hold each 2D slice
#     slices = []
    
#     # Load and append each 2D image slice to the list
#     for file in tqdm(files, desc="Loading slices", unit="file"):
#         image_path = os.path.join(folder, file)
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         slices.append(image)
       
#     # Convert the list of 2D slices into a 3D numpy array
#     if slices:
#         slices_array = np.stack(slices, axis=0)
        
#         return slices_array
#     else:
#         print("No valid slices found.")
#         return None
'''

def load_dataset_custom(dataset_name):
    # Define the path where the dataset is cached
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    dataset_cache_path = os.path.join(cache_dir, dataset_name.replace("/", "___"))

    # Check if the dataset cache exists
    if os.path.exists(dataset_cache_path):
        print(f"Dataset '{dataset_name}' already exists in the cache. Loading from cache...")
        ds = load_dataset(dataset_name, cache_dir=cache_dir)
    else:
        print(f"Dataset '{dataset_name}' not found in cache. Loading from Hugging Face Hub...")
        ds = load_dataset(dataset_name)
    
    return ds


def save_images_to_folder_from_dataset(dataset, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Use tqdm to add a progress bar to the loop
    for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Saving images"):
        try:
            # Access the image data using the correct key (assuming it's 'image')
            image_data = sample['image']
            
            # If image_data is already a PIL Image
            if isinstance(image_data, Image.Image):
                pil_image = image_data
            else:
                # If image_data is a numpy array or similar, convert it to a PIL image
                pil_image = Image.fromarray(np.array(image_data))
            
            # Define the image file name
            image_file = os.path.join(output_folder, f'image_{i}.png')
            
            # Save the image
            pil_image.save(image_file)
           
        except Exception as e:
            print(f"Error saving image {i}: {e}")


def resize_images_in_directory(input_dir, output_dir, target_size):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Iterate over each file in the input directory
    for filename in tqdm(image_files, desc = 'Resizing Images'):
         # Load image
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping {filename}. Unable to load image.")
                continue

            # Resize image
            resized_image = cv2.resize(image, target_size)
            # Save resized image to the output directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, resized_image)
            # print(f"Resized {filename} saved to {output_path}") 


def image_to_tif(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    files = os.listdir(input_folder)

    # Filter for only .png files
    png_files = [file for file in files if file.endswith((".png" , ".jpeg" , ".mask"))]

    # Loop through files and process the .png images with a progress bar
    for file in tqdm(png_files, desc="Converting PNG to TIFF"):
        # Read the .png image
        img = cv2.imread(os.path.join(input_folder, file))
        
        if img is not None and isinstance(img, np.ndarray):
            # Create a new file name by changing the extension to .tif
            tif_filename = file.replace(".png", ".tif")
            
            # Save the image as a .tif file in the output folder
            cv2.imwrite(os.path.join(output_folder, tif_filename), img)
            
            # Uncomment if you want to log each saved image
            # print(f"Saved {tif_filename} to {output_folder}")
        else:
            print(f"Error: Image {file} is not readable or not a valid numpy array.")


def load_tiff_images_as_slices(folder):
    # Use natsorted to ensure natural order
    files = natsorted([f for f in os.listdir(folder) if f.endswith('.tif')])

    slices = []
    mask_slices = []
    for file in tqdm(files, desc="Loading slices", unit="file"):
        image_path = os.path.join(folder, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            slices.append(image)
            if "mask" in file.lower():
                mask_slices.append(image)
        else:
            tqdm.write(f"Error loading {file}")
    
    return np.array(slices), np.array(mask_slices)


def reconstruct_3d_mesh(data, threshold=0):
   
    # Apply the Marching Cubes algorithm to get the vertices and faces
    verts, faces, normals, _ = measure.marching_cubes(data, level=threshold)

    # PyVista expects a flat array where the first element of each face is the number of vertices (3 for triangles)
    faces_pyvista = np.hstack([np.full((faces.shape[0], 1), 3), faces])

    # Flatten the faces array for PyVista
    faces_pyvista = faces_pyvista.flatten()

    return verts, faces_pyvista

    
def visualize_3d_mesh_stretched_along_axis(verts, faces,verts_mask,faces_mask, gap=0):
    """
    Visualizes the 3D mesh using PyVista with a small gap between slices.

    Parameters:
    - verts: Vertices of the 3D mesh.
    - faces: Faces of the 3D mesh.
    - gap: Gap size between slices (default is 0.1).
    """
    # Adjust vertices to create a gap (if slices are known)
    # This example assumes slices are aligned along the z-axis.
    verts_with_gap = verts.copy()
    slice_thickness = len(verts) // 10  # Example: dividing vertices into slices

    for i in range(len(verts)):
        if i % slice_thickness == 0:
            verts_with_gap[i:, 0] += gap  # Add gap along the z-axis

    # Create a PyVista mesh object
    mesh = pv.PolyData(verts, faces)
    mesh_mask = pv.PolyData(verts_mask , faces_mask)
    # total_obj = pv.PolyData(verts,faces)
    # # total_obj.add_field_data(verts_mask,'vert_mask')
    # # total_obj.add_field_data(faces_mask ,'faces_mask')
   
    total_obj = pv.PolyData(verts,faces)
    total_obj.append_polydata(mesh_mask)
    
    filename = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\mesh.obj"
    filename_mask = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\mesh_mask.obj"
    filename_whole_brain = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\total_object.obj"
    mesh.save(filename)
    mesh_mask.save(filename_mask)
    total_obj.save(filename_whole_brain)

    
    # Use tqdm to simulate progress for each vertex being processed
    for _ in tqdm(range(len(verts)), desc="Processing vertices"):
        pass  # Simulating processing; replace with actual processing if needed

    # Visualize the mesh with a gap between slices
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', show_edges=True, smooth_shading=True)
    plotter.add_mesh(mesh_mask, color='red', show_edges=True, smooth_shading=True)
    
    plotter.show()
    
# Ftiaxnei to 3d object me ogko (volume)
def volume_rendering(data,data_mask):
    """
    Performs volume rendering of a 3D numpy array using PyVista.

    Parameters:
    - data: 3D numpy array representing the volume data.

    Returns:
    - Displays the volume rendering.   
    """

    # Wrap the NumPy array as a PyVista ImageData object
    volume = pv.wrap(data)
    volume_mask = pv.wrap(data_mask)
    # Define file paths for saving
    filename_volume = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\volume.vti"
    filename_mask = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\volume_mask.vti"

    # Save volume data to VTK format
    volume.save(filename_volume)
    volume_mask.save(filename_mask)
    
    print(f"Saved volume to: {filename_volume}")
    print(f"Saved mask volume to: {filename_mask}")


    plotter = pv.Plotter()# Create the plotter objects
    plotter.add_volume(volume, cmap="gray", opacity="linear")# Add the full volume to the first plotter
    plotter.add_volume(volume_mask, cmap="jet", opacity="linear")  # Add mask volume in red

    plotter.show()# Show the plots

def array_to_point_cloud(data_3d, threshold=0):
    """
    Converts a 3D numpy array into a point cloud based on a threshold.

    Parameters:
    - data_3d: 3D numpy array representing volume data.
    - threshold: Minimum value to consider a voxel as part of the point cloud.

    Returns:
    - points: Nx3 numpy array where each row is a point [x, y, z].
    """
    # Get the indices where data exceeds the threshold
    points = np.argwhere(data_3d > threshold)

    # Convert points to float for visualization
    return points.astype(float)

def visualize_point_cloud_from_array(points,points_mask,point_size=5,color="white", color_mask = "red"):
    """
    Visualizes a point cloud from a numpy array of points.

    Parameters:
    - points: Nx3 numpy array where each row is a point [x, y, z].
    - point_size: Size of the points in the visualization.
    - color: Color of the points.
    """
    # Convert points to a PyVista PolyData object
    point_cloud = pv.PolyData(points)
    point_cloud_mask = pv.PolyData(points_mask)
    filename = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\point_cloud.stl"
    point_cloud.save(filename)

    # Create the plotter
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, color=color, point_size=point_size, render_points_as_spheres=True)
    plotter.add_mesh(point_cloud_mask , color= color_mask ,point_size=point_size, render_points_as_spheres=True )
    
    # Add a camera orientation widget
    plotter.add_camera_orientation_widget()

    # Display the point cloud
    plotter.show()


def generate_voxel_grid(points, grid_size=32, bounds=None, binary=True):
    """
    Generates a voxel grid from a 3D point cloud.
    
    Parameters:
    - points (np.ndarray): Array of shape (N, 3), where N is the number of points. Each row contains the (x, y, z) coordinates.
    - grid_size (int): Number of voxels along each dimension.
    - bounds (tuple): Optional. ((x_min, x_max), (y_min, y_max), (z_min, z_max)) for the voxel grid bounds.
                      If not specified, it uses the min and max of the points.
    - binary (bool): If True, returns a binary grid. If False, returns a density grid.
    
    Returns:
    - voxel_positions (np.ndarray): Array of 3D coordinates for each occupied voxel if `binary` is True.
    - voxel_grid (np.ndarray): 3D numpy array representing the voxel grid if `binary` is False.
    """
    # Set bounds if none are provided
    if bounds is None:
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
    else:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds

    # Calculate voxel sizes
    voxel_size_x = (x_max - x_min) / grid_size
    voxel_size_y = (y_max - y_min) / grid_size
    voxel_size_z = (z_max - z_min) / grid_size
    
    # Initialize the voxel grid
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=int)
    
    # Map points to voxel grid indices
    for point in points:
        i = int((point[0] - x_min) / voxel_size_x)
        j = int((point[1] - y_min) / voxel_size_y)
        k = int((point[2] - z_min) / voxel_size_z)
        
        # Ensure indices stay within grid bounds
        i, j, k = min(grid_size - 1, max(i, 0)), min(grid_size - 1, max(j, 0)), min(grid_size - 1, max(k, 0))
        
        # Increment for density or mark for binary occupancy
        voxel_grid[i, j, k] += 1

    if binary:
        # Binary grid: mark occupied voxels
        voxel_grid = np.where(voxel_grid > 0, 1, 0)
        # Get the positions of the occupied voxels
        occupied_voxels = np.argwhere(voxel_grid > 0)
        # Convert voxel indices to real-world coordinates
        voxel_positions = (occupied_voxels * np.array([voxel_size_x, voxel_size_y, voxel_size_z])) + [x_min, y_min, z_min]
        return voxel_positions  # For binary, return occupied positions
    else:
        # For density, return the full grid
        return voxel_grid

def visualize_voxel_grid(voxel_positions,voxel_mask_positions):
    """
    Visualizes occupied voxels from a list of 3D positions.
    
    Parameters:
    - voxel_positions (np.ndarray): Array of 3D coordinates of occupied voxels.
    """
    point_cloud = pv.PolyData(voxel_positions)
    point_cloud_mask = pv.PolyData(voxel_mask_positions)
    
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, color="grey", point_size=5, render_points_as_spheres=True)
    plotter.add_mesh(point_cloud_mask, color="red", point_size=5, render_points_as_spheres=True)
    plotter.show()

import matplotlib.pyplot as plt

def visualize_mri_with_mask(mri_data, mask_data, num_cols=6):
    """
    Visualizes all MRI slices alongside their segmentation masks in a grid.
    
    Parameters:
    - mri_data: 3D NumPy array of the MRI volume.
    - mask_data: 3D NumPy array of the segmentation mask.
    - num_cols: Number of columns in the grid layout.
    """
    num_slices = min(mri_data.shape[0], mask_data.shape[0])  # Total number of slices
    num_rows = (num_slices // num_cols) + 1  # Compute required rows

    fig, axes = plt.subplots(num_rows, num_cols * 2, figsize=(20, num_rows * 3))
    fig.suptitle("MRI and Mask Slices", fontsize=16)

    for i in range(num_slices):
        row = i // num_cols
        col = (i % num_cols) * 2  # Space for MRI + Mask

        # MRI Slice
        axes[row, col].imshow(mri_data[i], cmap="gray")
        axes[row, col].set_title(f"MRI Slice {i}")
        axes[row, col].axis("off")

        # Mask Slice
        axes[row, col + 1].imshow(mask_data[i], cmap="Reds", alpha=0.7)
        axes[row, col + 1].set_title(f"Mask Slice {i}")
        axes[row, col + 1].axis("off")

    plt.show()



def visualize_obj_file(file_path):
    """
    Loads and visualizes a .obj file using PyVista.

    Parameters:
    - file_path: Path to the .obj file to load and visualize.
    """
    # Load the .obj file as a PyVista mesh
    mesh = pv.read(file_path)

    # Set up the PyVista plotter
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', show_edges=True, smooth_shading=True)
    plotter.add_axes()  # Adds an axis for reference
    plotter.show()

def main():
    output_folder_1 = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\sub_set_1_slices_from_pgn_to_tif"
    # output_folder_1 = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.2\Brain Tumor\Brain Tumor"
    output_folder_dataset_tiff = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\2d_images_from_dataset_totiff"  # Change this to your desired folder path
    # output_folder_dataset_tiff = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\images\images"
    output_dataset_folder = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\2d_images_from_dataset"
    point_cloud_directory = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\point_cloud_file"
    tumor_folder = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\tumor_dataset.10_segmented_tumors\brain_tumor_dataset\kaggle_3m\TCGA_CS_4941_19960909"
    
    masked_tumor_folder_output = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\tumor_dataset.10_segmented_tumors\brain_tumor_dataset\kaggle_3m\TCGA_DU_5854_19951104"
    masked_tumor_folder_input = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\tumor_dataset.10_segmented_tumors\brain_tumor_dataset\kaggle_3m\TCGA_CS_4942_19970222"
   
    #------------------------------------------------------------#
    # ------------Dataset Format and Items Debugging-------------#
    #------------------------------------------------------------#
    # ds = load_dataset_custom("TrainingDataPro/brain-mri-dataset")
    # print(ds) 
    # print(ds['train'][0]) #print first item
    # print(ds['train'].features) #columns names
    # print(f"Number of rows: {ds['train'].num_rows}")  # Print dataset size
    #------------------------------------------------------------#
  
    #------------------------------------------------------------#
    # ------------Saving Images from Dataset to Folder Debugging-------------#
    #------------------------------------------------------------#
    # save_images_to_folder_from_dataset(ds['train'],output_dataset_folder)
    #------------------------------------------------------------#

    #------------------------------------------------------------#
    #-----------------Resize images Debugging--------------------#
    #------------------------------------------------------------#
    # resize_images_in_directory(output_dataset_folder,output_dataset_folder,target_size=(224,224))
    # resize_images_in_directory(masked_tumor_folder_output,masked_tumor_folder_output,target_size=(224,224))
    #------------------------------------------------------------#

    #------------------------------------------------------------#
    #---------------PNG to tiff images Debugging-----------------#
    #------------------------------------------------------------#
    # image_to_tif(output_dataset_folder , output_folder_dataset_tiff)
    # image_to_tif(tumor_folder,tumor_folder)   
    # image_to_tif(output_folder_1,output_folder_1) 
    # image_to_tif(masked_tumor_folder_input,masked_tumor_folder_output)
    
    #------------------------------------------------------------#
    #------------------------------------------------------------#

    #------------------------------------------------------------#
    # -----------Loading Slices from ImagesDebugging-------------#
    #------------------------------------------------------------#
    # load_tiff_images_as_slices(output_folder_dataset_tiff)
    # load_tiff_images_as_slices(output_folder_1)
    # load_tiff_images_as_slices(tumor_folder)  
    data_3d ,data_3d_mask = load_tiff_images_as_slices(masked_tumor_folder_output)   
    #------------------------------------------------------------#
    points =[]
    points.append(data_3d)
    points.append(data_3d_mask)
    import os

    # Define your dataset folder path
    dataset_folder = masked_tumor_folder_output
    # Count the number of .tif files
    num_tiff_files = len([f for f in os.listdir(dataset_folder) if f.endswith('.tif')])

    print(f"✅ Number of TIFF slices found: {num_tiff_files}")

    # # Print intensity statistics
    # print(f"Min intensity: {np.min(data_3d)}")
    # print(f"Max intensity: {np.max(data_3d)}")
    # print(f"Mean intensity: {np.mean(data_3d)}")
    # print(f"Median intensity: {np.median(data_3d)}")

    
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(8,6))
    # plt.hist(data_3d.flatten(), bins=100, color='gray', alpha=0.7)
    # plt.title("Histogram of MRI Intensities")
    # plt.xlabel("Intensity Value")
    # plt.ylabel("Frequency")
    # plt.show()

    # Visualize MRI and mask slices
    visualize_mri_with_mask(data_3d, data_3d_mask)

    # # Adjust min_intensity based on the histogram results
    # data_3d_mask = recreate_mask(data_3d, min_intensity=150, max_intensity=255)

    

    #------------------------------------------------------------#
    #-------------Converting Slices to numpy.arrays--------------#
    #------------------------------------------------------------#
    # data_3d = convert_2d_slice_to_nparray(output_folder_dataset_tiff)
    # data_3d = convert_2d_slice_to_nparray(masked_tumor_folder_output)
    # tumor_data_3d = convert_2d_slice_to_nparray(tumor_folder)
    # if data_3d is not None:
    #     print(f"3D array shape: {data_3d.shape}")
    #     print(f"Array data type: {data_3d.dtype}")
    #     print(f"Min value in 3D array: {np.min(data_3d)}")
    #     print(f"Max value in 3D array: {np.max(data_3d)}")
        
    #     # Optionally, inspect a specific slice (e.g., the first one)
    #     print(f"First slice shape: {data_3d[0].shape}")
    #     print(f"First slice min value: {np.min(data_3d[0])}")
    #     print(f"First slice max value: {np.max(data_3d[0])}")
    # else:
        # print("Failed to create the 3D array.")
    #------------------------------------------------------------#
    #------------------------------------------------------------#
    
    
    #------------------------------------------------------------#
    #-----------------Visualize Single Slices--------------------#
    #------------------------------------------------------------#
    # visualize_slice(data_3d,0)
    #------------------------------------------------------------#

    

    
    #------------------------------------------------------------#
    #------------Constructing 3d mesh and Visualize--------------#
    #------------------------------------------------------------#
    for threshold in [55]:
        print(f"Reconstructing with threshold = {threshold}")
        verts, faces = reconstruct_3d_mesh(data_3d, threshold=threshold)
        verts_mask ,faces_mask = reconstruct_3d_mesh(data_3d_mask, threshold= threshold)
        visualize_3d_mesh_stretched_along_axis(verts ,faces ,verts_mask,faces_mask,gap = 0)
        
    # verts, faces = reconstruct_3d_mesh(tumor_data_3d, threshold=200)
    
    # verts, faces = reconstruct_3d_mesh(data_3d, threshold= 50)
    # visualize_3d_mesh(verts ,faces)

   


    #------------------------------------------------------------#
    #-------------Visualizing the Whole 3d object----------------#
    #------------------------------------------------------------#
    volume_rendering(data_3d,data_3d_mask) 
    # volume_rendering(data_3d_mask)
    #------------------------------------------------------------#
    points = array_to_point_cloud(data_3d, threshold=55)
    points_mask = array_to_point_cloud(data_3d_mask,threshold= 0)

    visualize_point_cloud_from_array(points,points_mask)

    voxel_positions = generate_voxel_grid(points, grid_size=100, binary=True)
    voxel_mask_positions = generate_voxel_grid(points_mask, grid_size=100, binary=True)
    
    #mallon dne boleuei
    visualize_voxel_grid(voxel_positions,voxel_mask_positions)
    filename = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\mesh.obj"
    visualize_obj_file(filename)
    
    
if __name__ == "__main__":
    main() 