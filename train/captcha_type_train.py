"""
Trains a CNN for sorting Comprasnet CAPTCHAS into the following categories:

    1) bubble
    2) bubble_cut
    3) dotted
    4) dotted_wave
    5) wave
    
Images in the train dataset must be organized in folders named according to
their category. Such folders, in turn, must be in the directory indicated
in 'CAPTCHAS_FOLDER'
"""
import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense

CAPTCHAS_FOLDER = "/home/andremenegatti/Downloads/classification_train"
MODEL_FILENAME = "models/captcha_type_model.hdf5"
MODEL_LABELS_FILENAME = "models/captcha_type_labels.dat"

# Initializing data and labels
data = []
labels = []

# Looping over CAPTCHA images
for image_file in paths.list_images(CAPTCHAS_FOLDER):
    
    # Loading image and converting to grayscale
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    
    # Thresholding
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Adding a third channel dimension (Keras expects 3 channles)
    image = np.expand_dims(image, axis=2)
    
    # Grabbing the CAPTCHA type based on the folder it was in
    label = image_file.split(os.path.sep)[-2]
    
    # Addinig the image and it's label to training data
    data.append(image)
    labels.append(label)

# Scaling the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)

# Converting the labels (types) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(labels)
labels = lb.transform(labels)

# Saving the mapping from labels to one-hot encodings
# (required to decode model's predictions afterwards)
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Initializing the Sequential model
model = Sequential()

# Single convolutional layer with max pooling
model.add(Conv2D(3, (5, 5), padding = "same",
                 input_shape = (90, 200, 1), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Flatten())
model.add(Dense(500, activation = "relu"))

# Output layer with 5 nodes (one for each possible CAPTCHA type)
model.add(Dense(5, activation = "softmax"))

# Compiling the model
model.compile(loss = "categorical_crossentropy",
              optimizer = "adam", metrics = ["accuracy"])

# Training the model
model.fit(data, labels, batch_size = 64, epochs = 10, verbose = 1)

# Saving the trained model to disk
model.save(MODEL_FILENAME)