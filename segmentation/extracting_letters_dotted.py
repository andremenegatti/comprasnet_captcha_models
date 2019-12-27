"""
This script iterates over 'dotted' CAPTCHAs, performs character segmentation 
and stores individual character images in folders whose names corresponds
to the characters it contains.

Notes:
    1) The input CAPTCHA files must be named appropriately, i.e., the first
    6 characters of their names must match the CAPTCHA solution (key-sensitive)
    2) In order to use this script in Windows systems, it may be necessary to
    substitute '/' for '\\' in the cv2.imread command inside the loop.
    3) Unfortunately, the code is not DRY; there is a lot of copy and paste.
"""
import os
import time
import math as mt
import numpy as np
import pandas as pd
import cv2

start_time = time.time()

INPUT_FOLDER = 'captchas/train/dotted'
OUTPUT_FOLDER = 'letters/dotted'

counts = {}
f_first_count = 0
error_count_more = 0
error_count_less = 0
error_count_split_m = 0
error_count_split_n = 0
error_count_split_h = 0
error_count_merged_r = 0
error_count_merged_f = 0
error_count_split_V = 0

total = len(os.listdir(INPUT_FOLDER))

# Column patterns of potentially troublesome letters
pm1 = [0, 0, 0, 0, 510, 510, 0, 3060, 3060]
pm2 = [0, 0, 0, 0, 510, 510, 0, 2550, 2550]
pm3 = [0, 3060, 3060, 0, 510, 510, 0, 0, 0, 0]
pm4 = [0, 2550, 2550, 0, 510, 510, 0, 0, 0, 0]
ph1 = [0, 4080, 4080, 0, 0, 0, 0, 510, 510]


# Iterating over CAPTCHA images
for index, file in enumerate(os.listdir(INPUT_FOLDER)):

    print('\n' + str(index + 1) + "/" + str(total))

    src = INPUT_FOLDER + '/' + file
    captcha_correct_text = os.path.splitext(file)[0][0:6]

    captcha = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    captcha = cv2.threshold(captcha, 0, 255, cv2.THRESH_BINARY_INV)[1]

    image = captcha[19:46,]

    col_sum = np.sum(image, axis = 0)
    col_sum_list = list(col_sum)

    stop = False
    error = 'none'

    ############################################
    #  ------------  SPECIAL CASE ------------ #
    # -- LOWER CASE F IN THE FIRST POSITION -- #
    ############################################

    if col_sum_list[28:36] == [3570, 3570, 0 , 1020, 1020, 0, 510, 510]:
        print('Lower case F in the first position')
        f_first_count = f_first_count + 1
        continue

        # Redefining image so as the first letter (f) is not included
        image = image[:, 36:]

        col_sum = np.sum(image, axis = 0)
        col_sum_list = list(col_sum)

        # Finding all the dark regions
        # beggining and end of all dark regions)
        x = 1
        i = 0
        dark_regions = []
        while i < 164:
            if col_sum_list[i] == 0:
                dark_region_beg = i
                while col_sum_list[i + x] == 0:
                    x = x + 1
                    if (x + i) > 163:
                        break
                dark_region_end = i + x - 1
                dark_region = (dark_region_beg, dark_region_end)
                dark_regions.append(dark_region)
                i = x + i + 1
                x = 1
            else:
                i = i + 1

        # Identifying leftmost and rightmost dark regions and popping them out
        left_region = dark_regions[0]
        right_region = dark_regions[-1]
        dark_regions.pop(0)
        dark_regions.pop(-1)

        # Sorting dark regions according to their length
        four_regions = \
        sorted(dark_regions, key = lambda x: x[1] - x[0], reverse = True)

        gaps = []
        lines = []
        for i, region in enumerate(four_regions):
            gap = mt.ceil((region[1] - region[0]) / 2)
            if gap == 0:
                continue
            gaps.append(gap)
            lines.append(region[0] + gap)

        # If more than 4 remaining gaps are identified,
        # the problem may be due to split letters.
        # Some of the troublesome letters are m, n and h
        # We will try to fix this issue by completing gaps in these letters
        if len(lines) > 4:
            print("MORE than 4 remaining inner dark regions found\n \
                  Trying to solve the error...")

            for i in range(len(col_sum_list[:-9])):
                if col_sum_list[i:i+9] == pm1:
                    captcha[28:30, i+1:i+3] = 255
                if col_sum_list[i:i+9] == pm2:
                    captcha[31:33, i+1:i+3] = 255
                if col_sum_list[i:i+9] == pm3:
                    captcha[28:30, i+7:i+9] = 255
                if col_sum_list[i:i+9] == pm4:
                    captcha[31:33, i+7:i+9] = 255
                if col_sum_list[i:i+9] == ph1:
                    captcha[31:33, i+4:i+6] = 255

            # Reloading image (based on modified captcha) 
            # and redefining col_sum_list
            image = captcha[19:46, 36:]
            col_sum_list = list(np.sum(image, axis = 0))

            # Finding all the dark regions
            # beggining and end of all dark regions)
            x = 1
            i = 0
            dark_regions = []
            while i < 164:
                if col_sum_list[i] == 0:
                    dark_region_beg = i
                    while col_sum_list[i + x] == 0:
                        x = x + 1
                        if (x + i) > 163:
                            break
                    dark_region_end = i + x - 1
                    dark_region = (dark_region_beg, dark_region_end)
                    dark_regions.append(dark_region)
                    i = x + i + 1
                    x = 1
                else:
                    i = i + 1

            # Identifying leftmost/rightmost dark regions and popping them out
            left_region = dark_regions[0]
            right_region = dark_regions[-1]
            dark_regions.pop(0)
            dark_regions.pop(-1)

            # Sorting dark regions according to their length
            four_regions = \
            sorted(dark_regions, key = lambda x: x[1] - x[0], reverse = True)

            gaps = []
            lines = []
            for i, region in enumerate(four_regions):
                gap = mt.ceil((region[1] - region[0]) / 2)
                if gap == 0:
                    continue
                gaps.append(gap)
                lines.append(region[0] + gap)

            # If the errors persists, we move on to next captcha
            if len(lines) > 4:
                print('Unable to solve error: \
                      MORE than 4 remaining inner regions found')
                error_count_more = error_count_more + 1
                #error = 'more'
                continue

        # If the algorithm finds less letters than expected (merged letters),
        # we move on to next captcha
        if len(lines) < 4:
            print("LESS than 4 remaining inner dark regions found")
            error_count_less = error_count_less + 1
            #error = 'less'
            continue

        # Defining rightmost and leftmost lines, appending lines list, sorting
        left_line = 0
        right_line = right_region[0] + 2
        lines.append(left_line)
        lines.append(right_line)
        lines = sorted(lines)

        # Adjusting coordinates to account for deleting first letter
        lines = list(map(lambda x: x + 36, lines))

        # Finding letters x-coords (coords for initial f are already included)
        letters_xcoords = [(26, 37)]
        for i in range(len(lines)):
            if lines[i] == lines[-1]:
                break
            letter = (lines[i], lines[i + 1])
            letters_xcoords.append(letter)

        # Finding letters in the captcha, using the x-coordinates
        letters = []
        for i, letter in enumerate(letters_xcoords):
            letter_image = captcha[:60, letter[0]:letter[1]]
            letters.append(letter_image)

        # Looping over letters for labeling and saving
        for i, letter_image in enumerate(letters):

            # Assigning labels to images
            letter_text = captcha_correct_text[i]

            # Lower case R is a troublesome letter,
            # frequently 'merged' with other letters.
            # m, n, h and V are also problematic,
            # since sometimes they are 'split' into more than one letter
            # If an unusually wide or narrow image is identified
            # as one of such letters, it is probably due to an error.
            # We will discard captchas in which this occurs
            if letter_text == 'r' and letter_image.shape[1] > 16:
                stop = True
                error_count_merged_r = error_count_merged_r + 1
                break

            if letter_text == 'f' and letter_image.shape[1] > 16:
                stop = True
                error_count_merged_f = error_count_merged_f + 1
                break

            if letter_text == 'm' and letter_image.shape[1] < 26:
                stop = True
                error_count_split_m = error_count_split_m + 1
                break

            if letter_text == 'n' and letter_image.shape[1] < 16:
                stop = True
                error_count_split_n = error_count_split_n + 1
                break

            if letter_text == 'h' and letter_image.shape[1] < 16:
                stop = True
                error_count_split_h = error_count_split_h + 1
                break

            if letter_text == 'V' and letter_image.shape[1] < 20:
                stop = True
                error_count_split_V = error_count_split_V + 1
                break

            # Identifying categories for saving images in different folders
            if letter_text.islower():
                case = "Lower"
            elif letter_text.isupper():
                case = "Upper"
            else:
                case = "Number"

            # Defining output directory - if it does not exist, create it
            save_path = os.path.join(OUTPUT_FOLDER, case, letter_text)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Saving letter image file on disk
            count = counts.get(letter_text, 1)
            
            final_letter_file = \
            os.path.join(save_path, f"{captcha_correct_text}_" + \
                         "{}.png".format(str(count).zfill(6)))
            
            cv2.imwrite(final_letter_file, letter_image)

            # Increment the count for the current key
            counts[letter_text] = count + 1

        # Discard if a lower case 'r' seems to be merged with another letter
        if stop:
            continue

    ############################################
    #  ------------  GENERAL CASE ------------ #
    # ---- ALL REMAINING INITIAL LETTERS ----- #
    ############################################

    else:

        col_sum = np.sum(image, axis = 0)
        col_sum_list = list(col_sum)

        # Finding all the dark regions
        # beggining and end of all dark regions)
        x = 1
        i = 0
        dark_regions = []
        while i < 200:
            if col_sum_list[i] == 0:
                dark_region_beg = i
                while col_sum_list[i + x] == 0:
                    x = x + 1
                    if (x + i) > 199:
                        break
                dark_region_end = i + x - 1
                dark_region = (dark_region_beg, dark_region_end)
                dark_regions.append(dark_region)
                i = x + i + 1
                x = 1
            else:
                i = i + 1

        # Identifying leftmost/rightmost dark regions and popping them out
        left_region = dark_regions[0]
        right_region = dark_regions[-1]
        dark_regions.pop(0)
        dark_regions.pop(-1)

        # Sorting dark regions according to their length
        five_regions = \
        sorted(dark_regions, key = lambda x: x[1] - x[0], reverse = True)

        # Building a list of GAPS (lengths of the dark regions)
        # and LINES that split such gaps in half
        gaps = []
        lines = []
        for i, region in enumerate(five_regions):
            gap = mt.ceil((region[1] - region[0]) / 2)
            if gap == 0:
                continue
            gaps.append(gap)
            lines.append(region[0] + gap)


        # If more than 5 gaps are identified,
        # the problem may be due to split letters.
        # Some of the troublesome letters are m, n and h
        # We will try to fix this issue by completing gaps in these letters
        if len(lines) > 5:
            print("MORE than 5 inner dark regions found\n \
                  Trying to solve the error...")

            for i in range(len(col_sum_list[:-9])):
                if col_sum_list[i:i+9] == pm1:
                    captcha[28:30, i+1:i+3] = 255
                if col_sum_list[i:i+9] == pm2:
                    captcha[31:33, i+1:i+3] = 255
                if col_sum_list[i:i+9] == pm3:
                    captcha[28:30, i+7:i+9] = 255
                if col_sum_list[i:i+9] == pm4:
                    captcha[31:33, i+7:i+9] = 255
                if col_sum_list[i:i+9] == ph1:
                    captcha[31:33, i+4:i+6] = 255

            # Reloading image (based on modified captcha) 
            # and redefining col_sum_list
            image = captcha[19:46, ]
            col_sum_list = list(np.sum(image, axis = 0))

            # Finding all the dark regions
            # beggining and end of all dark regions)
            x = 1
            i = 0
            dark_regions = []
            while i < 200:
                if col_sum_list[i] == 0:
                    dark_region_beg = i
                    while col_sum_list[i + x] == 0:
                        x = x + 1
                        if (x + i) > 199:
                            break
                    dark_region_end = i + x - 1
                    dark_region = (dark_region_beg, dark_region_end)
                    dark_regions.append(dark_region)
                    i = x + i + 1
                    x = 1
                else:
                    i = i + 1

            # Identifying leftmost/rightmost dark regions and popping them out
            left_region = dark_regions[0]
            right_region = dark_regions[-1]
            dark_regions.pop(0)
            dark_regions.pop(-1)

            # Sorting dark regions according to their length
            five_regions = \
            sorted(dark_regions, key = lambda x: x[1] - x[0], reverse = True)

            # Building a list of GAPS (lengths of the dark regions)
            # and LINES that split such gaps in half
            gaps = []
            lines = []
            for i, region in enumerate(five_regions):
                gap = mt.ceil((region[1] - region[0]) / 2)
                if gap == 0:
                    continue
                gaps.append(gap)
                lines.append(region[0] + gap)

            # If the errors persists, we move on to next captcha
            if len(lines) > 5:
                print('Unable to solve error: MORE than 5 inner regions found')
                error_count_more = error_count_more + 1
                #error = 'more'
                continue

        # If the algorithm finds less letters than expected (merged letters),
        # we move on to next captcha
        if len(lines) < 5:
            print("LESS than 5 inner dark regions found")
            error_count_less = error_count_less + 1
            #error = 'less'
            continue

        # Defining rightmost/leftmost lines, appending lines list, and sorting
        left_line = left_region[1] - 2
        right_line = right_region[0] + 2
        lines.append(left_line)
        lines.append(right_line)
        lines = sorted(lines)

        # Finding letters x-coordinates
        letters_xcoords = []
        for i in range(len(lines)):
            if lines[i] == lines[-1]:
                break
            letter = (lines[i], lines[i + 1])
            letters_xcoords.append(letter)

        letters = []
        for i, letter in enumerate(letters_xcoords):
            letter_image = captcha[:60, letter[0]:letter[1]]
            letters.append(letter_image)

        # Looping over letters for labeling and saving
        for i, letter_image in enumerate(letters):

            # Assigning labels to images
            letter_text = captcha_correct_text[i]

            # Lower case R is a troublesome letter,
            # frequently 'merged' with other letters.
            # m, n, h and V are also problematic, 
            # since sometimes they are 'split' into more than one letter.
            # If an unusually wide or narrow image is identified
            # as one of such letters, it is probably due to an error.
            # We will discard captchas in which this occurs
            if letter_text == 'r' and letter_image.shape[1] > 16:
                stop = True
                error_count_merged_r = error_count_merged_r + 1
                break

            if letter_text == 'f' and letter_image.shape[1] > 16:
                stop = True
                error_count_merged_f = error_count_merged_f + 1
                break

            if letter_text == 'm' and letter_image.shape[1] < 26:
                stop = True
                error_count_split_m = error_count_split_m + 1
                break

            if letter_text == 'n' and letter_image.shape[1] < 16:
                stop = True
                error_count_split_n = error_count_split_n + 1
                break

            if letter_text == 'h' and letter_image.shape[1] < 16:
                stop = True
                error_count_split_h = error_count_split_h + 1
                break

            if letter_text == 'V' and letter_image.shape[1] < 20:
                stop = True
                error_count_split_V = error_count_split_V + 1
                break

            # Identifying categories for saving images in different folders
            if letter_text.islower():
                case = "Lower"
            elif letter_text.isupper():
                case = "Upper"
            else:
                case = "Number"

            # Defining output directory - if it does not exist, create it
            save_path = os.path.join(OUTPUT_FOLDER, case, letter_text)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Saving letter image file on disk
            count = counts.get(letter_text, 1)
            
            final_letter_file = \
            os.path.join(save_path, f"{captcha_correct_text}_" + \
                         "{}.png".format(str(count).zfill(6)))
            
            cv2.imwrite(final_letter_file, letter_image)

            # Increment the count for the current key
            counts[letter_text] = count + 1

        # Discard if a lower case 'r' seems to be merged with another letter
        if stop:
            continue

#### -------------- ////// ------------------ ###

print("\n------ ELAPSED TIME:\n" + "------ %s seconds \n" \
      % (time.time() - start_time))


## CHECKING ERRORS ##

error_summary_dict = {}

error_counts = [error_count_less, error_count_more, error_count_merged_r,
                error_count_merged_f, error_count_split_m, error_count_split_n,
                error_count_split_h, error_count_split_V]

error_types = ['less letters than expected', 'more letters than expected',
               'merged r', 'merged f', 'split m', 'split n',
               'split h', 'split V']

error_list = list(zip(error_types, error_counts))

for error in error_list:
    error_summary_dict[error[0]] = round(error[1] / total * 100, ndigits = 3)
error_summary_dict['TOTAL'] = sum(error_counts) / total * 100

error_summary_df = pd.DataFrame.from_dict(error_summary_dict,
                                          orient = 'index',
                                          columns = ['perc_error_rate'])

error_summary_df