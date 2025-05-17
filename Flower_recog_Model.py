
# Save this as train_model.py in your project directory

# Importing Libraries
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

# Fix path issues by using raw strings or forward slashes
base_path = r'C:\Users\tusha\OneDrive\Desktop\Flower recognition model'
flowers_path = os.path.join(base_path, 'flowers')

# Fetch Images count from Folders
count = 0
print(f"Looking for flower folders in: {flowers_path}")
try:
    dirs = os.listdir(flowers_path)
    for dir in dirs:
        dir_path = os.path.join(flowers_path, dir)
        if os.path.isdir(dir_path):  # Make sure it's a directory
            files = list(os.listdir(dir_path))
            print(f"{dir} Folder has {len(files)} Images")
            count = count + len(files)
    print(f"Images Folder has {count} Images in total")
except Exception as e:
    print(f"Error reading directories: {e}")
    print("Please make sure the flowers directory exists and contains flower category folders")
    exit(1)

# Load Images into Arrays as Dataset
img_size = 180
batch = 32

print("Loading and preparing the dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    flowers_path,
    seed=123,
    validation_split=0.2,
    subset='training',
    batch_size=batch,
    image_size=(img_size, img_size)
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    flowers_path,
    seed=123,
    validation_split=0.2,
    subset='validation',
    batch_size=batch,
    image_size=(img_size, img_size)
)

flower_names = train_ds.class_names
print(f"Flower categories detected: {flower_names}")

# Performance optimizations
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data Augmentation
print("Setting up data augmentation...")
data_augmentation = Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_size, img_size, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Model Creation
print("Creating the CNN model...")
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(flower_names))  # Make sure this matches the number of flower categories
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print("Model architecture summary:")
model.summary()

# Training the model
print("\nStarting model training...")
print(f"Training on {len(flower_names)} flower categories: {flower_names}")
print("This may take several minutes depending on your computer...")

history = model.fit(
    train_ds, 
    epochs=15, 
    validation_data=val_ds,
    verbose=1  # Show progress
)

# Define the classification function
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = f"The Image belongs to {flower_names[np.argmax(result)]} with a score of {np.max(result)*100:.2f}%"
    return outcome

# Test on a sample image if available
test_img_path = os.path.join(base_path, 'download.jpeg')
if os.path.exists(test_img_path):
    print("\nTesting model on sample image...")
    print(classify_images(test_img_path))
else:
    print("\nNo test image found at", test_img_path)

# Save the model
model_save_path = os.path.join(base_path, 'Flower_Recog_Model.h5')
print(f"\nSaving model to {model_save_path}...")
model.save(model_save_path)

print("\nDONE! The model has been trained and saved.")
print(f"Model file location: {model_save_path}")
print("\nYou can now run the Streamlit app using: streamlit run app.py")