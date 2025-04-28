# load data
# seperate for training and evaluation 
# crop brain from image 
# resize for each training model 

import os
import cv2
import numpy as np
import matplotlib.pyplot as plts
import itertools
from tqdm import tqdm
import shutil
import random 
# import tensorflow as tf 



#  IMG_PATH = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\tumor_dataset.1\brain_tumor_dataset"

# Separate data into train/val/test subfolders 

def separate_data(img_path, output_path):
    for CLASS in os.listdir(img_path):
        if not CLASS.startswith('.'):
            CLASS_PATH = os.path.join(img_path, CLASS)  # Construct the class path
            if os.path.isdir(CLASS_PATH):  # Ensure CLASS_PATH is a directory
                file_names = os.listdir(CLASS_PATH)
                random.shuffle(file_names)  # Shuffle the list of file names
                IMG_NUM = len(file_names)  # Count the number of images in the class path
                for (n, FILE_NAME) in enumerate(tqdm(os.listdir(CLASS_PATH),desc= f'Separating Data {CLASS}')):
                    img = os.path.join(CLASS_PATH, FILE_NAME)  # Construct the full image path
                    if os.path.isfile(img):  # Check if img is a file
                        if n < 5:
                            destination_dir = os.path.join(output_path, 'test', CLASS.lower())
                        elif n < 0.8 * IMG_NUM:
                            destination_dir = os.path.join(output_path, 'train', CLASS.lower())
                        else:
                            destination_dir = os.path.join(output_path, 'val', CLASS.lower())

                        os.makedirs(destination_dir, exist_ok=True)  # Create the destination directory if it doesn't exist
                        shutil.copy(img, os.path.join(destination_dir, FILE_NAME))  # Copy file to destination directory

def resize_images_in_directory(input_dir, output_dir, target_size=(224, 224)):
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
            
def crop_brain(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for filename in tqdm(image_files , desc = 'Cropping Images'):
          # Load image
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping {filename}. Unable to load image.")
                continue
            
             # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Threshold the image, then perform a series of erosions + dilations to remove noise
            thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea)

            # Find extreme points
            x, y, w, h = cv2.boundingRect(c)
            extLeft = x
            extRight = x + w
            extTop = y
            extBot = y + h

            
            new_img = image[extTop:extBot, extLeft:extRight].copy()

            # Save cropped image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, new_img)
            # print(f"Cropped {filename} saved to {output_path}")

def copy_images(input_dir, output_dir):
   # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
     # Get list of files in the input directory
    filenames = os.listdir(input_dir)
    # Iterate over files in the input directory
    for filename in tqdm(filenames ,desc ='Copying Data'):
        # Construct the input and output file paths
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)   

        # Copy the file to the output directory
        shutil.copy(input_file_path, output_file_path)


def augment_data(input_dir, output_dir):
  
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of files in the input directory
    filenames = os.listdir(input_dir)

    # Wrap the loop with tqdm to track progress
    for filename in tqdm(filenames, desc="Augmenting Data"):
        # Construct the input and output file paths
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)
        shutil.copy(input_file_path, output_file_path)
    
        
        # Load image
        image = cv2.imread(input_file_path)
        image2 = cv2.imread(input_file_path)
        image3 =cv2.imread(input_file_path)
        if image is None:
            print(f"Skipping {filename}. Unable to load image.")
            continue
        elif image2 is None:
            print(f"Skipping {filename}. Unable to load image2.")
            continue
        elif image3 is None:
            print(f"Skipping {filename}. Unable to load image3.")
            continue
       
        try:
                
                image = cv2.imread(input_file_path)
                image2 = cv2.imread(input_file_path)
                image3 = cv2.imread(input_file_path)
                # Check if the image is loaded successfully and has the expected number of channels
                if image is None or image.shape[2] != 3:
                    raise ValueError("Invalid input image format or number of channels")

        
                i = 0
                while i <10 :
                    cv2.imwrite(output_file_path, image)
                    random_angle = random.randint(-360, 0)
                    # Compute the center of the image
                    center = (image.shape[1] // 2, image.shape[0] // 2)
                    # Perform rotation with proper aspect ratio handling
                    rotation_matrix = cv2.getRotationMatrix2D(center, random_angle, 1.0)
                    augmented_image_rot = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)              # Save the rotated image
                    cv2.imwrite(output_file_path[:-4] + f"_rotated{i}.jpg", augmented_image_rot)
                    i+=1  
               
        except Exception as e:
            print(f"Error processing image {input_file_path}: {str(e)}")
            continue  # Skip this file if image processing fails
    
    # Return the paths of the input and output directories after augmentation
    return (input_dir, output_dir)

def main():
    img_path = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset"
    output_path = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset"
   
    # Separate data into train/val/test subfolders
    # separate_data(img_path, output_path)

    # Input directory containing original images before cropping and resizing 
    input_dir_train_no = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\no"
    input_dir_train_yes = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\yes"
    
    input_dir_test_no = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\test\no"
    input_dir_test_yes = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\test\yes"
    
    input_dir_val_no = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\val\no"
    input_dir_val_yes = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\val\yes"
    
   
    
    # Output directory containing resized  images 
    output_dir_val_resized_no = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\val\resized_images_val_no"
    output_dir_val_resized_yes = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\val\resized_images_val_yes"
    
    
    output_dir_test_resized_no = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\test\resized_images_test_no"
    output_dir_test_resized_yes = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\test\resized_images_test_yes"
    
 
    output_dir_train_resized_no = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\resized_images_train_no"
    output_dir_train_resized_yes = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\resized_images_train_yes"
    


    # Output directory containing cropped images 
    output_dir_train_cropped_no = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\cropped_images_no"
    output_dir_train_cropped_yes = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\cropped_images_yes"
    
    output_dir_val_cropped_no = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\val\cropped_images_no"
    output_dir_val_cropped_yes = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\val\cropped_images_yes"
    

    output_dir_test_cropped_no = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\test\cropped_images_no"
    output_dir_test_cropped_yes = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\test\cropped_images_yes"
    
    #input_dir_augmented_data_yes = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\augmented_images_yes"
    # output_dir_augmented_data_no = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\augmented_images_no"
    
    # Crop train data
    # crop_brain(input_dir_train_no,output_dir_train_cropped_no)
    # crop_brain(input_dir_train_yes,output_dir_train_cropped_yes)  
    
    # #  Crop validation data
    # crop_brain(input_dir_val_no,output_dir_val_cropped_no)
    # crop_brain(input_dir_val_yes,output_dir_val_cropped_yes)  
    
    # # Crop  test data
    # crop_brain(input_dir_test_no,output_dir_test_cropped_no)
    # crop_brain(input_dir_test_yes,output_dir_test_cropped_yes)  
    



    # # Resize train data 
    # resize_images_in_directory(output_dir_train_cropped_no, output_dir_train_resized_no)
    # resize_images_in_directory(output_dir_train_cropped_yes, output_dir_train_resized_yes)
    
    # # Resize test data 
    # resize_images_in_directory(output_dir_test_cropped_no, output_dir_test_resized_no)
    # resize_images_in_directory(output_dir_test_cropped_yes, output_dir_test_resized_yes)
    
    # # Resize  val data 
    # resize_images_in_directory(output_dir_val_cropped_no, output_dir_val_resized_no)
    # resize_images_in_directory(output_dir_val_cropped_yes, output_dir_val_resized_yes)
    

    # all_images_resized_dir = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\test\all_resized_images"
    # copy_images(output_dir_test_resized_no , all_images_resized_dir)
    # copy_images(output_dir_test_resized_yes , all_images_resized_dir)
    input_dir = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.7\Brain_Tumor_Detection\New folder\yes"
    output_dir =r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.7\Brain_Tumor_Detection\New folder\yes"
    crop_brain(input_dir,output_dir)
    resize_images_in_directory(output_dir,output_dir)

    # output_dir_augmented_data_no = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\augmented_images_no"
    # output_dir_augmented_data_yes = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\augmented_images_yes"
    
    # augment_data(output_dir_train_resized_no, output_dir_augmented_data_no)
    # augment_data(output_dir_train_resized_yes, output_dir_augmented_data_yes)


    # all_augmented_images_dir = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\all_augmented_images"
    # copy_images(output_dir_augmented_data_no, all_augmented_images_dir)
    # copy_images(output_dir_augmented_data_yes, all_augmented_images_dir)
    

if __name__ == "__main__":  
    main()




