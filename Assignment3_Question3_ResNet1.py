from google.colab import drive

drive.mount('/content/drive')

import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Path to your dataset in Google Drive

data_dir = '/content/drive/MyDrive/AI Projects/flowers'



# Image and batch parameters

img_height = 128

img_width = 128

batch_size = 32



# Create train and validation data generators

datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2

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


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load the pre-trained ResNet50 model, excluding the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




epochs = 10  # You can adjust the number of epochs

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
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
