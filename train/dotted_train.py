"""
Trains a CNN for identifying single characters from 'dotted' CAPTCHAS.

Character segmentation must be performed beforehand. For this purpose, see 
modules in the 'segmentation' directory.

The input folder, 'LETTER_IMAGES_FOLDER' must contain individual character 
images, in 60x60 png files, organized into appropriately named folders - i.e.,
each folder name must represent the number or letter (key-sensitive) depicted
in the png files it contains.
"""
import pickle
import os.path
import numpy as np
import cv2
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential
from functions.resize_to_fit import resize_to_fit

LETTER_IMAGES_FOLDER = "letters/dotted"
MODEL_FILENAME = "models/dotted_model.hdf5"
MODEL_LABELS_FILENAME = "models/dotted_labels.dat"

# Initializing lists with data and labels to be used in training
data = []
labels = []

# Looping over input images paths
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    
    # Loading image and converting it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resizing the letter so it fits in a 60x60 pixel box
    image = resize_to_fit(image, 60, 60)
    
    # Adding a third channel dimension to the image
    image = np.expand_dims(image, axis=2)
    
    # Grabbing the name of the character based on the folder it was in
    label = image_file.split(os.path.sep)[-2]
    
    # Adding the letter image and it's label to training data
    data.append(image)
    labels.append(label)

# Scaling the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Converting the labels (letters/numbers) into one-hot encodings
lb = LabelBinarizer().fit(labels)
labels = lb.transform(labels)

# Saving the mapping from labels to one-hot encodings
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Initializing and building the neural network
model = Sequential()
model.add(Conv2D(20, (3, 3), padding="same",
                 input_shape=(60, 60, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(15, (4, 4), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# Output layer with 57 nodes (one for each possible character)
model.add(Dense(57, activation="softmax"))

# Compiling the model
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# Training the neural network
model.fit(data, labels,
          batch_size=128, epochs=15, verbose=1)

# Saving the trained model to disk
model.save(MODEL_FILENAME)