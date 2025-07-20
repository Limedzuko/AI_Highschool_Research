from google.colab import drive

drive.mount('/content/drive')

import pathlib
import tensorflow as tf
import os

# Define hyperparameters
img_height = VARIED PER MODEL
img_width = VARIED PER MODEL 
batch_size = VARIED PER MODEL 
# Learning rate will be set in the model compilation cell

# Path to your dataset in Google Drive
data_dir = '/content/drive/MyDrive/AI Projects/flowers'


# Convert data_dir to a Path object for easier manipulation
data_dir = pathlib.Path(data_dir)

# Get a list of all image file paths
image_files = list(data_dir.glob('*/*'))
image_files = [str(path) for path in image_files]

# Get the class names from the directory structure
class_names = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])
print(f'Found {len(image_files)} images belonging to {len(class_names)} classes.')

# Create a mapping from class name to class index
class_indices = dict((name, i) for i, name in enumerate(class_names))

# Function to get the label from the file path
def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The class name is the second to last component
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)

# Function to decode the image and preprocess it
def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])

# Function to process the path to an image
def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    # Scale pixel values to [0, 1]
    img = img / 255.0
    return img, tf.one_hot(label, depth=len(class_names))

# Create a dataset from the image file paths
list_ds = tf.data.Dataset.from_tensor_slices(image_files)

# Process the dataset
AUTOTUNE = tf.data.AUTOTUNE
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# Determine the size of the dataset for splitting
dataset_size = len(image_files)
train_size = int(dataset_size * VARIED PER MODEL) # 80% for training usually, except the last 3 neural networks where it varied
val_size = dataset_size - train_size  # Remaining for validation

# Split the dataset
train_ds = labeled_ds.take(train_size)
val_ds = labeled_ds.skip(train_size)

# Batch and prefetch the datasets for performance
def configure_for_performance(ds):
    ds = ds.cache() # Cache data after preprocessing
    ds = ds.shuffle(buffer_size=1000) # Shuffle training data
    ds = ds.batch(batch_size) # Batch data
    ds = ds.prefetch(buffer_size=AUTOTUNE) # Fetch batches in advance
    return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

print("tf.data pipelines created.")

import tensorflow as tf

from tensorflow.keras import layers, models

# Define learning rate hyperparameter
learning_rate = VARIED PER MODEL 

model = models.Sequential([
    layers.Flatten(input_shape=(img_height, img_width, 3)),
    # Add a hidden Dense layer
    layers.Dense(128, activation='relu'), #THIS HIDDEN LAYER DOES NOT EXIST FOR FIRST 2 MODELS, AND THERE IS AN EXTRA HIDDEN LAYER IN THE 29TH MODEL 
    layers.Dense(len(class_names), activation='softmax') # Use len(class_names) for output layer size
])

# Use the defined learning rate in the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_ds, # Use the new tf.data training dataset
    validation_data=val_ds, # Use the new tf.data validation dataset
    epochs= VARIED PER MODEL 
)

import matplotlib.pyplot as plt

 

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.show()

 

plt.plot(history.history["accuracy"], label="Training Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

print(f"Epochs: {history.params['epochs']}")
print(f"Batch Size: {batch_size}")
print(f"Image Resolution: {img_height}x{img_width}")
print(f"Learning Rate: {learning_rate}")

# Get the training history data
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

# Final metrics
final_loss = train_loss[-1]
final_accuracy = train_accuracy[-1]
final_val_loss = val_loss[-1]
final_val_accuracy = val_accuracy[-1]

print("--- Final Metrics ---")
print(f"Final Training Loss: {final_loss:.4f}")
print(f"Final Training Accuracy: {final_accuracy:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
print("-" * 25)

# Best/Worst metrics and their epochs
max_train_accuracy = max(train_accuracy)
max_train_accuracy_epoch = epochs[train_accuracy.index(max_train_accuracy)]
min_train_loss = min(train_loss)
min_train_loss_epoch = epochs[train_loss.index(min_train_loss)]

max_val_accuracy = max(val_accuracy)
max_val_accuracy_epoch = epochs[val_accuracy.index(max_val_accuracy)]
min_val_loss = min(val_loss)
min_val_loss_epoch = epochs[val_loss.index(min_val_loss)]


print("--- Best/Worst Metrics ---")
print(f"Maximum Training Accuracy: {max_train_accuracy:.4f} (Epoch {max_train_accuracy_epoch})")
print(f"Minimum Training Loss: {min_train_loss:.4f} (Epoch {min_train_loss_epoch})")
print(f"Maximum Validation Accuracy: {max_val_accuracy:.4f} (Epoch {max_val_accuracy_epoch})")
print(f"Minimum Validation Loss: {min_val_loss:.4f} (Epoch {min_val_loss_epoch})")
print("-" * 25)
