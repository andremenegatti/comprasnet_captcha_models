"""
Uses a CNN to sort CAPTCHAs from Comprasnet into 4 categories and copies their
png files to folders named according to the predicted category.

The categories are:
    1) bubble
    2) bubble_cut
    3) dotted
    4) dotted_wave
    5) wave
"""
from keras.models import load_model
from imutils import paths
import numpy as np
import pickle
import cv2

MODEL_FILENAME = "models/captcha_type_model.hdf5"
MODEL_LABELS_FILENAME = "models/captcha_type_labels.dat"
INPUT_FOLDER = "captchas/sample"
OUTPUT_FOLDER = 'captchas/sorted'

# Defining variable with the total number of CAPTCHAS to be sorted
total = len(list(paths.list_images(INPUT_FOLDER)))

# Loading model labels to decode model predictions
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Loading the neural network
model = load_model(MODEL_FILENAME)

# Looping over image paths
for index, captcha_image_file in enumerate(paths.list_images(INPUT_FOLDER)):

    # Printing progress
    print(str(index + 1) + '/' + str(total))

    # Loading, grayscaling, and thresholding image
    captcha = cv2.imread(captcha_image_file)
    image = cv2.cvtColor(captcha, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Adding a third channel dimension to the image
    image = np.expand_dims(image, axis = 2)
    image = np.expand_dims(image, axis = 0)

    # Predicting CAPTCHA type and decoding the one-hot-encoded prediction
    prediction = model.predict(image)
    predicted_class = lb.inverse_transform(prediction)[0]
    
    # Saving captcha in the appropriate folder
    save_path = \
    OUTPUT_FOLDER + '/' + predicted_class + '/' + captcha_image_file[-10:]
    
    cv2.imwrite(save_path, captcha)