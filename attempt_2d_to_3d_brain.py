import cv2
import math
import numpy as np
import os
from tqdm import tqdm
from skimage import measure
import pyvista as pv
from PIL import Image
from datasets import load_dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


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
    # List all files in the folder
    files = sorted([f for f in os.listdir(folder) if f.endswith('.tif')])

    # Load each image slice into a list
    slices = []
    for file in tqdm(files, desc="Loading slices", unit="file"):
        image_path = os.path.join(folder, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            slices.append(image)
        else:
            tqdm.write(f"Error loading {file}")  # Use tqdm.write() to print error messages without disrupting the progress bar
    
        if image is not None:
               
                slices.append(image)
                # # Debugging: Print summary information about each slice
                # print(f"Loaded slice: {file}")
                # print(f"  Original Shape: {image.shape}")
                # print(f"  Resized Shape: {image.shape}")
                # print(f"  Min pixel value: {np.min(image)}")
                # print(f"  Max pixel value: {np.max(image)}")
                # print(f"hj {type(image)}")
                
        else:
                print(f"Error loading {file}")
    # print(np.mean(image))
    print(np.array(slices[0]))
    return np.array(slices)


def convert_2d_slice_to_nparray(folder):
    """
    Converts a folder of 2D image slices into a 3D numpy array.

    Parameters:
    - folder: The folder containing 2D slices in .tif format.

    Returns:
    - A 3D numpy array with the stacked image slices.
    """
    # List all .tif files in the folder and sort them
    files = sorted([f for f in os.listdir(folder) if f.endswith('.tif')])

    # Initialize an empty list to hold each 2D slice
    slices = []
    
    # Load and append each 2D image slice to the list
    for file in tqdm(files, desc="Loading slices", unit="file"):
        image_path = os.path.join(folder, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        slices.append(image)
       
    # Convert the list of 2D slices into a 3D numpy array
    if slices:
        slices_array = np.stack(slices, axis=0)
        
        return slices_array
    else:
        print("No valid slices found.")
        return None

def reconstruct_3d_mesh(data, threshold=0):
   
    # Apply the Marching Cubes algorithm to get the vertices and faces
    verts, faces, normals, _ = measure.marching_cubes(data, level=threshold)

    # PyVista expects a flat array where the first element of each face is the number of vertices (3 for triangles)
    faces_pyvista = np.hstack([np.full((faces.shape[0], 1), 3), faces])

    # Flatten the faces array for PyVista
    faces_pyvista = faces_pyvista.flatten()

    return verts, faces_pyvista


    
def visualize_3d_mesh_with_gap(verts, faces, gap=0):
    """
    Visualizes the 3D mesh using PyVista with a small gap between slices.

    Parameters:
    - verts: Vertices of the 3D mesh.
    - faces: Faces of the 3D mesh.
    - gap: Gap size between slices (default is 0).
    """
    # Convert to numpy array if not already
    verts = np.array(verts)
    faces = np.array(faces)

    # Ensure faces are reshaped correctly
    if faces.ndim == 1:
        faces = faces.reshape((-1, 4))  # Reshape assuming triangular faces

    # Determine slice thickness (example uses number of vertices to divide slices)
    slice_thickness = len(verts) // 10  # Change this value based on your data

    # List to hold all the slices separately
    all_meshes = []

    # Iterate through slices and apply gap
    for i in range(0, len(verts), slice_thickness):
        # Select the vertices for the current slice
        slice_verts = verts[i:i + slice_thickness].copy()

        # Apply the gap to shift this slice along the x-axis
        slice_verts[:, 0] += (i // slice_thickness) * gap

        # Filter the faces that belong to the current slice
        slice_faces = []
        for face in faces:
            # Ensure the face references the correct vertices
            if all(i <= idx < i + slice_thickness for idx in face[1:]):
                adjusted_face = [face[0]] + [(idx - i) for idx in face[1:]]
                slice_faces.extend(adjusted_face)

        # Convert the slice_faces list to a numpy array with the correct shape
        slice_faces = np.array(slice_faces).reshape((-1, 4))  # Reshape based on triangular faces

        # Create a mesh only if the slice has valid faces
        if len(slice_faces) > 0:
            mesh = pv.PolyData(slice_verts, slice_faces)
            all_meshes.append(mesh)

    # Visualize all the separated slices with gaps
    plotter = pv.Plotter()
    for mesh in tqdm(all_meshes, desc="Adding slices to plot"):
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, smooth_shading=True)

    plotter.show()

def visualize_3d_mesh_stretched_along_axis(verts, faces, gap=0):
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
    mesh = pv.PolyData(verts_with_gap, faces)

    # Use tqdm to simulate progress for each vertex being processed
    for _ in tqdm(range(len(verts_with_gap)), desc="Processing vertices"):
        pass  # Simulating processing; replace with actual processing if needed

    # Visualize the mesh with a gap between slices
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', show_edges=True, smooth_shading=True)
    plotter.show()

def visualize_slice(data, slice_index):
    """
    Visualizes a single 2D slice from a 3D numpy array.

    Parameters:
    - data: 3D NumPy array containing the image slices.
    - slice_index: The index of the slice to visualize (along the z-axis).
    """
    # Get the dimensions of the 3D array
    depth = data.shape[0]
    
    # Ensure the slice index is within the valid range
    if slice_index < 0 or slice_index >= depth:
        print(f"Error: slice_index {slice_index} is out of range. Must be between 0 and {depth - 1}.")
        return
    
    # Extract the 2D slice at the specified index
    slice_data = data[slice_index]
    
    # Plot the slice
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_data, cmap='gray')  # Use 'gray' colormap for grayscale images
    plt.title(f"Slice {slice_index}")
    plt.axis('off')  # Hide the axis for a cleaner look
    plt.show()

# Ftiaxnei olo to 3d object
def volume_rendering(data):
    """
    Performs volume rendering of a 3D numpy array using PyVista.

    Parameters:
    - data: 3D numpy array representing the volume data.

    Returns:
    - Displays the volume rendering.   
    """

    # Wrap the NumPy array as a PyVista ImageData object
    volume = pv.wrap(data)
    
    plotter = pv.Plotter()# Create the plotter objects
    
    plotter.add_volume(volume, cmap="gray", opacity="linear")# Add the full volume to the first plotter
    
    plotter.show()# Show the plots
   

def visualize_point_cloud(filename):
    """
    Visualizes a point cloud from a PLY file using PyVista.

    Parameters:
    - filename: Path to the PLY file containing the point cloud.
    """
    # Read the point cloud from the PLY file
    point_cloud = pv.read(filename)

    # Create a plotter object
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, point_size=5, render_points_as_spheres=True, color="white")
    
   # Add a camera orientation widget
    plotter.add_camera_orientation_widget()
    center = point_cloud.center

    # Position the camera very close to the center of the point cloud
    plotter.camera_position = [
        (center[0] + 0.1, center[1] + 0.1, center[2] + 0.1),  # Camera position (very close to the center)
        center,  # Camera focal point (center of the point cloud)
        (0, 0, 1)  # View-up vector
    ]

    # Set a high zoom level
    plotter.camera.zoom(10000)
    
    
    plotter.show()

def center_point_cloud(arr):
    """
    Centers the point cloud around the origin.
    """
    centroid = np.mean(arr, axis=0)
    return arr - centroid

def createPointCloud(directory, arr):
    # Ensure that the directory exists
    if not os.path.exists(directory):
        print("Directory does not exist. Creating...")
        os.makedirs(directory)
    else:
        print("Directory already exists.")

    # Define the filename inside the given directory
    filename = os.path.join(directory, "output.ply")

    # Open file and write boilerplate header
    try:
        with open(filename, 'w') as file:
            file.write("ply\n") 
            file.write("format ascii 1.0\n") 

            # Count number of vertices
            num_verts = arr.shape[0] 
            file.write("element vertex " + str(num_verts) + "\n") 
            file.write("property float32 x\n") 
            file.write("property float32 y\n") 
            file.write("property float32 z\n") 
            file.write("end_header\n") 

            # Write points
            point_count = 0 
            for point in arr:
                # Progress check
                point_count += 1 
                if point_count % 1000 == 0:
                    print("Point: " + str(point_count) + " of " + str(len(arr))) 

                # Create file string
                out_str = " ".join(str(axis) for axis in point) + "\n"
                file.write(out_str)
        print(f"File created successfully at: {filename}")
    except PermissionError:
        print("Permission denied. Check the directory path and permissions.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    output_folder_1 = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\sub_set_1_slices_from_pgn_to_tif"
    # output_folder_1 = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.2\Brain Tumor\Brain Tumor"
    output_folder_dataset_tiff = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\2d_images_from_dataset_totiff"  # Change this to your desired folder path
    # output_folder_dataset_tiff = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\images\images"
    output_dataset_folder = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\2d_images_from_dataset"
    point_cloud_directory = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\point_cloud_file"
    tumor_folder = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\tumor_dataset.10_segmented_tumors\brain_tumor_dataset\kaggle_3m\TCGA_CS_4941_19960909"
    
    masked_tumor_folder_output = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\tumor_dataset.10_segmented_tumors\brain_tumor_dataset\lgg-mri-segmentation\kaggle_3m\TCGA_CS_4941_19960909"
    masked_tumor_folder_input = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\tumor_dataset.10_segmented_tumors\brain_tumor_dataset\lgg-mri-segmentation\kaggle_3m\TCGA_CS_4941_19960909"
   
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
    resize_images_in_directory(masked_tumor_folder_output,masked_tumor_folder_output,target_size=(224,224))
    #------------------------------------------------------------#

    #------------------------------------------------------------#
    #---------------PNG to tiff images Debugging-----------------#
    #------------------------------------------------------------#
    # image_to_tif(output_dataset_folder , output_folder_dataset_tiff)
    # image_to_tif(tumor_folder,tumor_folder)   
    # image_to_tif(output_folder_1,output_folder_1) 
    image_to_tif(masked_tumor_folder_input,masked_tumor_folder_output)
    
    #------------------------------------------------------------#
    #------------------------------------------------------------#

    #------------------------------------------------------------#
    # -----------Loading Slices from ImagesDebugging-------------#
    #------------------------------------------------------------#
    # load_tiff_images_as_slices(output_folder_dataset_tiff)
    # load_tiff_images_as_slices(output_folder_1)
    # load_tiff_images_as_slices(tumor_folder)  
    load_tiff_images_as_slices(masked_tumor_folder_output)   
    #------------------------------------------------------------#



    #------------------------------------------------------------#
    #-------------Converting Slices to numpy.arrays--------------#
    #------------------------------------------------------------#
    # data_3d = convert_2d_slice_to_nparray(output_folder_dataset_tiff)
    data_3d = convert_2d_slice_to_nparray(masked_tumor_folder_output)
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
        visualize_3d_mesh_stretched_along_axis(verts ,faces)
    # verts, faces = reconstruct_3d_mesh(tumor_data_3d, threshold=200)
    
    # verts, faces = reconstruct_3d_mesh(data_3d, threshold= 50)
    # visualize_3d_mesh(verts ,faces)

    
    #------------------------------------------------------------#
    #-------------Visualizing with differen colors---------------#
    #------------------------------------------------------------#
    # thresholds = [30, 100, 200]  # Different threshold values
    # colours = ['red', 'green', 'blue']  # Colours for each threshold
    # visualize_3d_mesh_different_colours(verts, faces, thresholds, colours, gap=0)
    #------------------------------------------------------------#


    #------------------------------------------------------------#
    #-------------Visualizing the Whole 3d object----------------#
    #------------------------------------------------------------#
    #volume_rendering(data_3d) 
    #------------------------------------------------------------#
     
   
    # createPointCloud(point_cloud_directory,data_3d)
    # filename = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\3d_reconstruction_from_2d_slices\point_cloud_file\output.ply"
    # visualize_point_cloud(filename)
if __name__ == "__main__":
    main() 