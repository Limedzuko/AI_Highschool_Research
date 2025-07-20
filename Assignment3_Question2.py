from google.colab import drive

drive.mount('/content/drive')

import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Path to your dataset in Google Drive

data_dir = '/content/drive/MyDrive/AI Projects/faces'

# Image and batch parameters

img_height = VARIED BETWEEN MODELS

img_width = VARIED BETWEEN MODELS

batch_size = VARIED BETWEEN MODELS



# Create train and validation data generators

datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split= VARIED BETWEEN MODELS #was 0.2 for first two models

)



train_generator = datagen.flow_from_directory(

    data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical',

    subset='training'

)



val_generator = datagen.flow_from_directory(

    data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical',

    subset='validation'

)

learning_rate = VARIED BETWEEN MODELS
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Flatten(input_shape=(img_height, img_width, 3)),
    layers.Dense(128, activation='relu'), # WAS NOT PRESENT FOR FIRST MODEL
    layers.Dense(len(class_names), activation='softmax') # Use len(class_names) for the number of classes
])

# Use the defined learning rate in the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs= VARIED BETWEEN MODELS 

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
