"""
This script iterates over 'wave' CAPTCHAs, performs character segmentation 
and stores individual character images in folders whose names corresponds
to the characters it contains.

Notes:
    1) The input CAPTCHA files must be named appropriately, i.e., the first
    6 characters of their names must match the CAPTCHA solution (key-sensitive)
    2) In order to use this script in Windows systems, it may be necessary to
    substitute '/' for '\\' in the cv2.imread command inside the loop.
"""

import time
import os
import numpy as np
import cv2

INPUT_FOLDER = 'captchas/train/wave'
OUTPUT_FOLDER = 'letters/wave'

counts = {}
cont_error = []
point_error = []
split_error = []
total = len(os.listdir(INPUT_FOLDER))

start_time = time.time()

# Iterating over CAPTCHA images
for index, captcha_image_file in enumerate(os.listdir(INPUT_FOLDER)):

    print(str(index + 1) + '/' + str(total) + \
          ' ------- ' + captcha_image_file + '\n')

    stop = False

    captcha_correct_text = os.path.splitext(captcha_image_file)[0][0:6]

    thresh = cv2.imread(INPUT_FOLDER + "/" + captcha_image_file,
                        cv2.IMREAD_GRAYSCALE)
    
    thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.copyMakeBorder(thresh, 8, 8, 8, 8, cv2.BORDER_CONSTANT, 0)

    hcontours = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP,
                                 cv2.CHAIN_APPROX_SIMPLE)
    contours = hcontours[0]

    conts = [(c,) for c, x in enumerate(hcontours[1][0]) if x[3] == -1]

    for pingij_index, pingij \
    in [(index, cc) for index, cc in enumerate(contours) \
        if (cv2.boundingRect(cc)[3] < 10) and (index, ) in conts]:

        pingij_coordx = max(pingij, key = lambda coords: coords[0][1])[0][0]

        dists = []

        for cont_item in [index[0] for index in conts if len(index) == 1]:
            l = contours[cont_item]
            l_index = cont_item
            if np.array_equal(l, pingij): continue
            dist = abs(pingij_coordx - \
                       min(l, key = lambda coords: coords[0][1])[0][0])
            dists.append((dist, l_index))

        min_index = min(dists)[1]

        try:
            conts.remove((min_index,))
            conts.remove((pingij_index,))
            conts.append((pingij_index, min_index))

        except:
            cont_error.append(captcha_image_file)
            stop = True
            break

    if stop:
        continue

    if len(conts) != 6:
        print("Error in " + captcha_image_file + ": discarding...")
        split_error.append(captcha_image_file)
        continue

    sorted_conts = []
    for i in conts:

        if len(i) == 1:
            (x, y, w, h) = cv2.boundingRect(contours[i[0]])
        else:
            (x_pingij, y_pingij, w_pingij, h_pingij) = \
            cv2.boundingRect(contours[i[0]])
            
            (x_ij, y_ij, w_ij, h_ij) = cv2.boundingRect(contours[i[1]])
            x = min(x_pingij, x_ij)
            y = y_pingij
            w = max(x_pingij + w_pingij, x_ij + w_ij) - x
            h = y_ij + h_ij - y

        letter = np.zeros((106, 216), np.uint8)

        for cont in i:

            letter = cv2.drawContours(image = letter, contours = contours,
                                      contourIdx = cont, color = 255,
                                      thickness = -1, hierarchy = hcontours[1],
                                      maxLevel = 2)

        letter_cut = letter[y-2 : y+h+2, x-2 : x+w+2]

        if letter_cut.shape[0] < 18:
            point_error.append(captcha_image_file)
            stop = True
            break

        sorted_conts.append((x, letter_cut))

    if stop:
        continue

    for i, letter_image in enumerate(sorted(sorted_conts)):

        letter_text = captcha_correct_text[i]

        if letter_text.isupper():
            case = "Upper"
        elif letter_text.islower():
            case = "Lower"
        else:
            case = "Number"

        save_path = os.path.join(OUTPUT_FOLDER, case, letter_text)

        # If the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        count = counts.get(letter_text, 1)
        final_letter_file = \
        os.path.join(save_path, f"{captcha_correct_text}_" \
                     + "{}.png".format(str(count).zfill(6)))
        
        cv2.imwrite(final_letter_file, letter_image[1])

        # Increment the count for the current key
        counts[letter_text] = count + 1



#### -------------------- ////// ------------------------ ###



print("\n------ ELAPSED TIME:\n" + "------ %s seconds \n" \
      % (time.time() - start_time))

cont_error_rate_perc = round( len(cont_error) * 100/ total, ndigits = 3)
split_error_rate_perc = round( len(split_error) * 100/ total, ndigits = 3)
point_error_rate_perc = round( len(point_error) * 100/ total, ndigits = 3)

print('cont error rate (perc):  ' + str(cont_error_rate_perc))
print('point error rate (perc): ' + str(point_error_rate_perc))
print('split error rate (perc): ' + str(split_error_rate_perc))
print('TOTAL ERROR RATE (PERC): ' + str(cont_error_rate_perc \
      + split_error_rate_perc + point_error_rate_perc))
