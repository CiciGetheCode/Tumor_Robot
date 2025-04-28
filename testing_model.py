import keras._tf_keras
import tensorflow as tf
import os
from tqdm import tqdm
import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_model(model_path):
    """
    Loads the trained model from the given path.
    
    :param model_path: Path to the .keras file containing the trained model.
    :return: Loaded Keras model.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_test_data(test_dir, img_size=(224, 224), batch_size=1):
    """
    Loads the test dataset from the specified directory.
    
    :param test_dir: Path to the directory containing test data.
    :param img_size: Target size for resizing the images (width, height).
    :param batch_size: Number of images per batch.
    """
    test_data = ImageDataGenerator(
        rescale=1.0/255,            # Normalize pixel values to [0, 1]
        rotation_range = 40,      # Random rotation between 0 and 40 degrees
        width_shift_range=0.2,     # Horizontal shifting
        height_shift_range=0.2,    # Vertical shifting
        shear_range=0.2,           # Shear transformation
        zoom_range=0.2,            # Random zoom
        horizontal_flip=True,      # Flip images horizontally
        fill_mode='nearest'       # Fill in missing pixels after transformations
    )
    test_generator= test_data.flow_from_directory(
        test_data_dir, 
        target_size=(224, 224), 
        batch_size= batch_size,
        class_mode='binary' ,
        subset = 'training'
    )
    return test_generator



def load_test_data_unlabeled(test_data_dir , img_size =(224,224) , batch_size = 1):
    
    test_data = keras._tf_keras.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode=None,  # For unlabeled data
        image_size= img_size,  # Adjust to your model's input size
        batch_size= batch_size
    )
    return test_data


def evaluate_model_labeled(model, test_data):
    """
    Evaluates the model using the test data.
    
    :param model: The loaded Keras model.
    :param test_data: The tf object containing test data.
    """
    if model is None:
        print("Model is not loaded. Evaluation cannot proceed.")
        return
    
    # Compile explicitly
    # compile_model(model)
    print("Evaluating model...")
    results = model.evaluate(test_data)
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")
    print(f"Test Precision: {results[2]}")
    print(f"Test Recall: {results[3]}")
    print(results)


def plot_confusion_matrix(model, test_data, class_names, binary=True):
    y_true = []
    y_pred = []
    num_images = sum(len(files) for _, _, files in os.walk(test_data_dir))
    total_batches = num_images

     # Collect all images and labels from the generator
    for i, (images, labels) in tqdm(enumerate(test_data), total=total_batches, desc="Processing Batches"):
        if i >= total_batches:
            break  # Stop after the exact number of images

        # Rest of the 
        # print(f"\nBatch {i+1}")
        # print("Images shape:", images.shape)
        # print("Labels shape:", labels.shape)
        
        y_true.extend(labels)
        
        # Predict in batch
        predictions = model.predict(images, verbose=0)
        # print("Predictions shape:", predictions.shape)
        # print("Predictions (first 5):", predictions[:5])

        # Convert predictions based on binary or multi-class
        if binary:
            batch_preds = (predictions > 0.5).astype(int).flatten()
            # print("Binary predictions (first 5):", batch_preds[:5])
            y_pred.extend(batch_preds)
        else:
            batch_preds = np.argmax(predictions, axis=1)
            # print("Multi-class predictions (first 5):", batch_preds[:5])
            y_pred.extend(batch_preds)
    
    y_true = np.array(y_true).flatten().astype(int)
    y_pred = np.array(y_pred).astype(int)

    # Final check for length mismatch
    print("\nFinal y_true length:", len(y_true))
    print("Final y_pred length:", len(y_pred))
    print("y_true (first 10):", y_true[:10])
    print("y_pred (first 10):", y_pred[:10])

    if len(y_true) != len(y_pred):
        raise ValueError("Mismatch in lengths of y_true and y_pred. Check your test data and predictions.")

    # Compute and plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


# def plot_predictions(model, test_data, binary=True):
#     # Get the total number of images in the test data directory
#     num_images = sum(len(files) for _, _, files in os.walk(test_data_dir))
#     print(f"Total number of images in test data: {num_images}")

#     y_pred = []
#     images_processed = 0


#     # Process images from the dataset
#     for images in tqdm(test_data, desc="Processing Batches"):
#         if images_processed >= num_images:
#             break  # Stop after the specified number of images

#         # Predict in batch
#         predictions = model.predict(images, verbose=0)

#         # Convert predictions based on binary or multi-class
#         if binary:
#             batch_preds = (predictions > 0.5).astype(int).flatten()
#             y_pred.extend(batch_preds)
#         else:
#             batch_preds = np.argmax(predictions, axis=1)
#             y_pred.extend(batch_preds)

#         images_processed += len(images)

#     # Convert predictions to a NumPy array for further use
#     y_pred = np.array(y_pred).astype(int)
#     print("\nFinal y_pred length:", len(y_pred))
#     print("y_pred (first 10):", y_pred[:10])

#     # Plot the distribution of predictions (for binary classification)
#     if binary:
#         plt.figure(figsize=(6, 4))
#         sns.countplot(x=y_pred, order=[0, 1])  # Specify order for binary classes
#         plt.xlabel('Predicted Labels')
#         plt.title('Prediction Distribution')
#         plt.show()

if __name__ == "__main__":
    # Define the paths
    model_path = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\classification_code\training_models\custom_training_model.keras"
    test_data_dir = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.5\brain_tumor_dataset\train\New folder"

    # Load the model
    model1 = load_model(model_path)

    # Load the test data
    test_data = load_test_data(test_data_dir)

    # Evaluate the model
    evaluate_model_labeled(model1, test_data)

    # Define class names based on the test folder structure
    # class_names = sorted(os.listdir(test_data_dir))
    # print(class_names)
    class_names = ["yes","no"]
    print(class_names)

    # test_data_unlabeled = load_test_data_unlabeled(test_data_dir)
    # Plot the confusion matrix
    plot_confusion_matrix(model1, test_data, class_names)
    # plot_predictions(model1,test_data_unlabeled,binary = True )