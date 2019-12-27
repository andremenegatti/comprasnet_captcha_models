"""
This script uses the custom function 'test_model' to evaluate
the performance of a CAPTCHA-breaking CNN.
"""

from functions.test_model import test_model

MODEL_NAME = 'wave_model'
CAPTCHA_CLASS = 'wave'

MODELS_FOLDER = 'models/'

MODEL_FILENAME = MODELS_FOLDER + MODEL_NAME + '.hdf5'
MODEL_LABELS_FILENAME = MODELS_FOLDER + CAPTCHA_CLASS + '_labels.dat'
INPUT_FOLDER = 'captchas/test/' + CAPTCHA_CLASS

test_model(model_filename = MODEL_FILENAME,
          model_labels_filename = MODEL_LABELS_FILENAME,
          input_folder = INPUT_FOLDER,
          captcha_class = CAPTCHA_CLASS)