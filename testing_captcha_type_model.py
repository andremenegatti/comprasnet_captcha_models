"""
Tests a CNN used for sorting Comprasnet CAPTCHAS into the following categories:

    1) bubble
    2) bubble_cut
    3) dotted
    4) dotted_wave
    5) wave
    
Images in the test dataset must be organized in folders named according to
their category. Such folders, in turn, must be in the directory indicated
in 'INPUT_FOLDER'
"""
import os
import pickle
import numpy as np
from keras.models import load_model
from imutils import paths
import cv2

MODEL_FILENAME = 'models/captcha_type_model.hdf5'
MODEL_LABELS_FILENAME = 'models/captcha_type_labels.dat'
INPUT_FOLDER = 'captchas/test'

# Initializing list to store prediction errors
prediction_errors = []

# Defining variable with the total number of CAPTCHAS to be used in testing
total = len(list(paths.list_images(INPUT_FOLDER)))

# Loading model labels to decode model predictions
with open(MODEL_LABELS_FILENAME, 'rb') as f:
    lb = pickle.load(f)

# Loading the neural network
model = load_model(MODEL_FILENAME)

# Looping over image paths
for index, captcha_image_file in enumerate(paths.list_images(INPUT_FOLDER)):
    
    # Printing progress
    print(str(index + 1) + '/' + str(total))

    # Extracting CAPTCHA actual type from the folder it was in
    captcha_class = captcha_image_file.split(os.path.sep)[-2]

    # Loading image and converting it to grayscale
    image = cv2.imread(captcha_image_file, cv2.IMREAD_GRAYSCALE)

    # Thresholding
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)[1]
        
    # Adding a third channel dimension to the image
    image = np.expand_dims(image, axis = 2)
    image = np.expand_dims(image, axis = 0)
    
    # Predicting
    prediction = model.predict(image)
    
    # Converting the one-hot-encoded prediction back to a CAPTCHA type
    predicted_class = lb.inverse_transform(prediction)[0]

    # Printing CAPTCHA's actual and predicted class
    print('Actual class: {}'.format(captcha_class))
    print('Predicted class: {}'.format(predicted_class))
    
    # Printing SUCCESS/FAILURE message and recording errors as a tuple
    if predicted_class != captcha_class:
        print(' --------- FAILURE\n')
        prediction_errors.append((captcha_image_file.split(os.path.sep)[-1],
                                  captcha_class,
                                  predicted_class))
    else:
        print('--------- SUCCESS\n')

print(' ---------- ///////// ----------- ')

prediction_error_rate = round(len(prediction_errors) / total, ndigits = 3)

print(f'Total Error rate: {prediction_error_rate}')