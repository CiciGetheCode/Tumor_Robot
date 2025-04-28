import tensorflow as tf 
import tqdm
import keras 
from keras._tf_keras.keras.applications.vgg19  import   VGG19 
from tensorflow.python.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping  # Import EarlyStopping
from keras.src import regularizers 
from keras.src import layers 
import os
from PIL import Image


class CustomTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_epoch=0):
        super(CustomTrainingCallback, self).__init__()
        self.epoch_successful = False
        self.global_epoch = initial_epoch  # Initialize global epoch count
        print("CustomTrainingCallback initialized with initial_epoch:", self.global_epoch)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_successful = True  # Assume the epoch will be successful unless stopped
        print(f"on_epoch_begin: Starting epoch {self.global_epoch + 1}.")  # Debug print for epoch start

    def on_epoch_end(self, epoch, logs=None):
        print(f"on_epoch_end: Epoch {self.global_epoch + 1} has ended.")
        if not self.epoch_successful:
            print(f"on_epoch_end: Epoch {self.global_epoch + 1} was aborted due to an OutOfRangeError. Dropping this epoch. \n")
            self.model.stop_training = False  # Reset for the next epoch
        else:
            print(f"on_epoch_end: Epoch {self.global_epoch + 1} completed successfully. \n")
            self.global_epoch += 1  # Increment the global epoch only if the epoch was successful

    def on_train_batch_end(self, batch, logs=None):
        # print(f"on_train_batch_end: Starting batch {batch}. Checking for errors... \n")
        try:
            # print(f"on_train_batch_end: Inside try block for batch {batch}. \n")
            if logs is not None and logs.get('loss') is not None:
                # print(f"on_train_batch_end: Batch {batch} completed successfully with loss: {logs.get('loss') } \n")
                return
        except tf.errors.OutOfRangeError:
            print(f"OutOfRangeError detected at batch {batch}. Aborting this epoch and dropping it. \n")
            self.model.stop_training = True
            self.epoch_successful = False
        except Exception as e:
            print(f"An unexpected error occurred at batch {batch}: {e} \n")
            self.model.stop_training = True
            self.epoch_successful = False
        finally:
            print(f"on_train_batch_end: Exiting try block for batch {batch}. \n")

    def on_train_begin(self, logs=None):
        print("on_train_begin: Training started. \n")

    def on_train_end(self, logs=None):
        print("on_train_end: Training completed. \n")

    def on_train_batch_begin(self, batch, logs=None):
        print(f"\non_train_batch_begin: Starting batch {batch}. \n")

    def on_epoch_start(self, epoch, logs=None):
        print(f"on_epoch_start: Preparing to start epoch {epoch + 1}. \n")

def check_images(directory):
    for subdir, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify if it is an image
            except (IOError, SyntaxError) as e:
                print(f"Corrupted or unrecognized image file: {file_path}")

def rename_images_in_directory(directory_path, new_name):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print("Directory does not exist.")
        return

    # List all files in the directory
    files = os.listdir(directory_path)
    
    # Filter for image files (add more extensions if needed)
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

    # Rename each image file
    for index, file_name in enumerate(image_files, start=1):
        # Get file extension
        file_extension = os.path.splitext(file_name)[1]
        # Create new file name
        new_file_name = f"{new_name}_{index}{file_extension}"
        # Define old and new file paths
        old_file_path = os.path.join(directory_path, file_name)
        new_file_path = os.path.join(directory_path, new_file_name)
        
        # Check if the current file name is already in the desired format
        if file_name == new_file_name:
            print(f"File already named as {new_file_name}, skipping.")
            continue

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {file_name} -> {new_file_name}")

def create_model():
    base_model = VGG19(
        input_shape = (224, 224, 3), # Shape of our images
        include_top = False, # Leave out the last fully connected layer
        weights = 'imagenet',
    )
    for layer in base_model.layers:
        layer.trainable = False

    # Build the model using the functional API
    x = base_model.output
    x = layers.Flatten()(x)  # Flatten the output from the base model
    x = layers.BatchNormalization()(x)  # Add batch normalization
    x = layers.Dense(256, kernel_initializer='he_uniform', kernel_regularizer = regularizers.L2(0.01) )(x)  # Add a Dense layer with 256 units and He initialization
    x = layers.BatchNormalization()(x)  # Add batch normalization again
    x = layers.Activation('relu')(x)  # Add ReLU activation
    x = layers.Dropout(0.4)(x)  # Add dropout with a rate of 0.5
    # Output layer with sigmoid activation for binary classification
    output = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs = base_model.input ,outputs = output)

    model.compile(
        optimizer = Adam(learning_rate = 0.001),
        loss='binary_crossentropy',
        metrics=['accuracy' ,"Precision" , "Recall"]
    )
    
    return model
    
def plot_metrics(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 5))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    # Directories used by the model
    train_dir = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\New folder"
    validation_dir = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\val\New folder"
    test_dir = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\test"
    
    # Directories for checking and renaming
    train_dir_no = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\New folder\augmented_images_no"
    train_dir_yes = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\New folder\augmented_images_yes"
    val_dir_no = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\val\New folder\resized_images_val_no"
    val_dir_yes = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\val\New folder\resized_images_val_yes"
    
    
    # Check for corrupted images in the directories
    check_images(train_dir)
    check_images(validation_dir)
    check_images(test_dir)
            
    model1 = create_model()
    model1.summary()
    
 
    #  Data augmentation with ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,            # Normalize pixel values to [0, 1]
        rotation_range = 40,      # Random rotation between 0 and 40 degrees
        width_shift_range=0.2,     # Horizontal shifting
        height_shift_range=0.2,    # Vertical shifting
        shear_range=0.2,           # Shear transformation
        zoom_range=0.2,            # Random zoom
        horizontal_flip=True,      # Flip images horizontally
        fill_mode='nearest'    ,    # Fill in missing pixels after transformations
        validation_split= 0.2
    )
    validation_datagen = ImageDataGenerator(rescale=1.0/255)
 
    batch_size_train = 64
    batch_size_val = 8

    train_generator = train_datagen.flow_from_directory(
        train_dir, 
        target_size=(224, 224), 
        batch_size= batch_size_train,
        class_mode='binary' ,
        subset = 'training'
    )

    validation_generator = train_datagen.flow_from_directory(
        
        train_dir,
        target_size = (224, 224),
        batch_size = batch_size_val,
        class_mode = 'binary' ,
        subset = 'validation'
    )
    
    target_epochs = 20
    epochs_completed = 0
    
    callback = CustomTrainingCallback(initial_epoch=0)  # Initialize the callback
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)# Set up early stopping to prevent overfitting

    while epochs_completed < target_epochs:
        print(f"Starting epoch {epochs_completed + 1}/{target_epochs}")

        history = model1.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size_train,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size_val,
            epochs=1,
            callbacks=[callback , early_stopping],
            verbose=1
        )

        # Check if the epoch was successful
        if callback.epoch_successful:
            epochs_completed += 1
    
    model1.save('custom_training_model.keras')
    

if __name__ == '__main__':
    main()