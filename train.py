import cv2  # Import the OpenCV library for computer vision tasks
import tensorflow as tf  # Import the TensorFlow library for deep learning tasks
from tensorflow import keras  # Import the Keras module for building and training neural networks
import numpy as np  # Import the NumPy library for numerical operations
from sklearn.model_selection import train_test_split  # Import train_test_split function for splitting the dataset
from keras.utils import to_categorical  # Import to_categorical function for one-hot encoding
from keras.optimizers import Adam  # Import the Adam optimizer for model training
from keras.layers import BatchNormalization, Activation  # Import BatchNormalization and Activation layers
from keras import layers  # Import additional layers from Keras
from keras.models import Model  # Import the Model class for building the model architecture
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, SeparableConv2D  # Import convolutional and pooling layers from Keras
import os  # Import the os module for operating system-related functionalities

# Load the dataset
data_dir = "dataset_images"  # Define the directory containing the dataset
label_file = "label.txt"  # Define the path to the label file

# Read the labels from label.txt
labels = []
with open(label_file, "r") as file:
    lines = file.readlines()
    for line in lines:
        label = line.strip().split(" ")[1]
        labels.append(label)

# Initialize the data and labels arrays
data = []
labels_encoded = []

# Iterate over the dataset folders
for label_idx, folder in enumerate(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder)
    # Iterate over the images in the folder
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        # Read and preprocess the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # Resize to desired input shape
        img = img / 255.0  # Normalize pixel values
        data.append(img)
        labels_encoded.append(label_idx)

# Convert data and labels to numpy arrays
data = np.array(data)
labels_encoded = np.array(labels_encoded)

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
num_classes = len(np.unique(labels_encoded))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Create the CNN model
model = keras.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add the modified modules
residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(model.layers[-1].output)
residual = BatchNormalization()(residual)
x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(model.layers[-1].output)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

residual = Conv2D(512, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

residual = Conv2D(512, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

# Flatten and add dense layers
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)

# Create the modified model
model = Model(inputs=model.input, outputs=x)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save("facedetection.h5")
