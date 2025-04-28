import tensorflow as tf 
import keras 
from keras import applications 
from keras._tf_keras.keras.applications.vgg19  import   VGG19 
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.applications import imagenet_utils
from keras.src.models import Functional
from keras.src.ops import operation_utils
from keras.src.utils import file_utils

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define a custom callback to print all epoch details
class DebugCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        
        print(f"\nBatch {batch + 1} (train) -> loss: {logs.get('loss', 'N/A'):.4f}, acc: {logs.get('acc', 'N/A'):.4f}")
        if 'steps' in self.params and 'batch_size' in self.params:
            remaining_images_train = (self.params['steps'] - (batch + 1)) * self.params['batch_size']
            print(f"Remaining training images in this epoch: {remaining_images_train}")

    def on_test_batch_end(self, batch, logs=None):
        print(f"\nBatch {batch + 1} (validation) -> val_loss: {logs.get('loss', 'N/A'):.4f}, val_acc: {logs.get('acc', 'N/A'):.4f}")
        if 'steps' in self.params and 'batch_size' in self.params:
            remaining_images_val = (self.params['steps'] - (batch + 1)) * self.params['batch_size']
            print(f"Remaining validation images in this epoch: {remaining_images_val}")
   
class EpochProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss', 'N/A')
        val_acc = logs.get('val_acc', 'N/A')
        print(f"Epoch {epoch + 1}:")
        print(f" - loss: {logs.get('loss', 'N/A'):.4f} - acc: {logs.get('acc', 'N/A'):.4f}")
        print(f" - val_loss: {val_loss} - val_acc: {val_acc}")
    # def on_batch_end(self, batch , logs = None):

base_model = VGG19(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')
for layer in base_model.layers:
    layer.trainable = False

# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output) 
# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)
# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])

# Directory paths for your specific case
train_dir = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\train\New folder"
validation_dir = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\val\New folder"  # Update with your path
test_dir = r"C:\Users\Aggelos\ECE-2017-\Diplomatiki\data\binary_tumor_existence\tumor_dataset.1\brain_tumor_dataset\test\New folder"
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize pixel values to [0, 1]
    # rotation_range=40,         # Random rotation between 0 and 40 degrees
    # width_shift_range=0.2,     # Horizontal shifting
    # height_shift_range=0.2,    # Vertical shifting
    # shear_range=0.2,           # Shear transformation
    # zoom_range=0.2,            # Random zoom
    # horizontal_flip=True,      # Flip images horizontally
    # fill_mode='nearest'        # Fill in missing pixels after transformations
    )

test_datagen = ImageDataGenerator( rescale = 1.0/255. )
validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size_train = 32
batch_size_val = 3 
image_size = (224,224)


train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size= batch_size_train,  class_mode='binary' )
# train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,shuffle=False,batch_size=batch_size_train,image_size = image_size)

validation_generator = validation_datagen.flow_from_directory( validation_dir, target_size = (224, 224), batch_size = batch_size_val,  class_mode = 'binary')

test_generator = test_datagen.flow_from_directory(test_dir , batch_size = 1 , class_mode = 'binary' , target_size = (224, 224) )

# train_size = sum([len(batch) for batch, _ in train_dataset])
train_size = train_generator.samples
valid_size = validation_generator.samples
epochs = 10
# Apply batching, repeating, and prefetching
# train_dataset = train_dataset.repeat().prefetch(tf.data.AUTOTUNE)
# Training
steps_per_epoch = 3* (train_size //batch_size_train)
# steps_per_epoch = train_dataset.cardinality().numpy()
validation_steps = valid_size // batch_size_val


# Train the model with limited steps for quick feedback
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[DebugCallback()]
    
)


model.save('path_to_save_model/my_model_from_dataset_1.h5')  # Save the model to an .h5 file


# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Predicting the test set results
predictions = model.predict(test_generator, steps=test_generator.samples)

# Convert probabilities to binary predictions (assuming binary classification)
y_pred = (predictions > 0.5).astype(int).flatten()

# True labels (since shuffle=False, the order will match the predictions)
y_true = test_generator.classes

# Calculate and display the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(cm)

# Classification report
target_names = ['Non-Tumor', 'Tumor']
print('\nClassification Report:')
print(classification_report(y_true, y_pred, target_names=target_names))

# Plotting the confusion matrix
plt.figure(figsize=(5,5))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.grid(False)

for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')

plt.show()
