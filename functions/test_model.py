import os
import time
import pickle
import numpy as np
import cv2
from keras.models import load_model
from functions.resize_to_fit import resize_to_fit
from functions.segmentation import split_captcha

def test_model(model_filename,
              model_labels_filename,
              input_folder,
              captcha_class,
              verbose=True,
              return_errors=False,
              dimensions_dict = {
            'dotted': {'width': 60, 'height': 60},
            'wave': {'width': 60, 'height': 60},
            'bubble': {'width': 86, 'height': 86},
            'bubble_cut': {'width': 85, 'height': 85}
            }):
    """
    Tests the performance of a CAPTCHA-breaking CNN built to handle one of
    the following types of CAPTCHAS:
        
        1) dotted
        2) wave
        3) bubble
        4) bubble_cut
    
    :param model_filename: path of the model .hdf5 file
    :param model_labels_filename: path of the .dat file with model labels
    :param input_folder: path of the folder with the test image dataset
    :param captcha_class: a string with one of the four supported classes
    :param verbose: if True, prints results for each CAPTCHA
    :param return_errors: if True, returns dict with files that caused errors
    :param dimencions_dict: dict with class-specific character dimensions
    :return: None, of a dict if return_errors=True
    """
    start_time = time.time()
    
    split_errors = []
    prediction_errors = []
    total = len(os.listdir(input_folder))

    # Load model labels to decode predictions
    with open(model_labels_filename, 'rb') as f:
        lb = pickle.load(f)

    # Load the trained neural network
    model = load_model(model_filename)
    
    letter_width = dimensions_dict[captcha_class]['width']
    letter_height = dimensions_dict[captcha_class]['height']
    
    # Looping over the image paths
    for index, captcha_file in enumerate(os.listdir(input_folder)):
        print(str(index + 1) + '/' + str(total))
        
        # Reading image as np.array, converting to grayscale and thresholding
        captcha_image_file = input_folder + '/' + captcha_file
        captcha_array = cv2.imread(captcha_image_file, cv2.IMREAD_GRAYSCALE)
        captcha_array = cv2.threshold(captcha_array,
                                      0, 255, cv2.THRESH_BINARY_INV)[1]
        
        # Performing character segmentation
        output = split_captcha(captcha_array, captcha_class)
        
        # Handling and recording errors
        if output == 'error':
            print('Unable to segment characters from ' \
                  + captcha_file + '\n')
            
            split_errors.append(captcha_file)
            continue

        # Creating a list to hold predicted characters
        predictions = []
        
        # Iterating over single-character arrays
        for image in output:

            # Adding additional padding to make images of same size
            image = resize_to_fit(image, letter_width, letter_height)

            # Add a third channel dimension to the image
            image = np.expand_dims(image, axis=2)
            image = np.expand_dims(image, axis=0)

            # Predicting, decoding result, and appending to predictions list
            prediction = model.predict(image)
            predicted_letter = lb.inverse_transform(prediction)[0]
            predictions.append(predicted_letter)

        # Printing CAPTCHA result
        captcha_text = ''.join(predictions)
        if captcha_text != captcha_file[0:6]:
            prediction_errors.append((captcha_text, captcha_file))
            status = 'FAILURE'
        else:
            status = 'SUCCESS'
        
        if verbose:
            print('Actual text: {}'.format(captcha_file[0:6]))
            print('Predicted text: {}'.format(captcha_text))
            print(' --------- ' + status + '\n')

    # Printing test summary
    elapsed_time = time.time() - start_time

    print('----------------------------------- \n')

    model.summary()

    print('----------------------------------- \n')

    print(str(total) + ' CAPTCHAs processed in ' + '%s seconds \n' \
          % round(elapsed_time, ndigits = 3))

    n_pred_errors = len(prediction_errors)
    n_split_errors = len(split_errors)
    n_total_errors = n_pred_errors + n_split_errors
    
    total_error_rate = round(n_total_errors / total,
                             ndigits=3)
    
    pred_error_rate = round(n_pred_errors / (total - n_split_errors),
                            ndigits=3)
   
    split_error_rate = round(n_split_errors / total,
                             ndigits=3)
    
    captchas_solved = total - n_split_errors - n_pred_errors

    print(f'Overall accuracy:                       {1 - total_error_rate}')
    print(f' - Char. segmentation error rate:       {split_error_rate}')
    print(f' - Prediction error rate:               {pred_error_rate}\n')

    print(f'Correct CAPTCHAs per second:            \
          {round(captchas_solved / elapsed_time, ndigits=3)}')
    
    # Returning dictionary with files that generated errors
    if return_errors:
        return({
                'prediction_errors': prediction_errors,
                'split_errors': split_errors
                })