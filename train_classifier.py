# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 03:07:16 2019

@author: user
"""

import numpy as np
import cv2
import glob


def Generate_ds(PEAK_COUNT_THRESHOLD, PEAK_VALUE_THRESHOLD):
    # Get all image path
    img_path_list = glob.glob('./data/' + '/*/*.jpg', recursive=True)
    label_list = []

    X = np.zeros((len(img_path_list), 1))
    i = 0
    for i_path in img_path_list:
        img = cv2.imread(i_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img /255.0
        horizontal_bin = np.mean(img, axis=1)
        horizontal_bin_diff = horizontal_bin[1:] - horizontal_bin[0:-1]
        peak_count = len(horizontal_bin_diff[horizontal_bin_diff>PEAK_VALUE_THRESHOLD])

        i_path = i_path.replace('\\','/')
        i_path = i_path.replace('.','/')
        label = i_path.split('/')[-3]
        label_list.append(int(label))

        X[i, :] = peak_count/len(horizontal_bin_diff)
        i = i+1

    # Evaluate
    correct_classification = 0
    for i ,i_data in enumerate(X):
        if i_data >= PEAK_COUNT_THRESHOLD:
            this_class = 1
        else:
            this_class = 0

        if this_class == label_list[i]:
            correct_classification += 1

    accuracy = correct_classification/len(label_list)

    return accuracy

# Grid search to find the parameters for minimum error classifier
peak_count_list = []
peak_value_list = []
acc_list = []
for i_peak_count in np.arange(0, 1, 0.01):
    for i_peak_value in np.arange(0, 1, 0.01):
        acc = Generate_ds(i_peak_count, i_peak_value)
        acc_list.append(acc)
        peak_count_list.append(i_peak_count)
        peak_value_list.append(i_peak_value)

max_index = np.argmax(acc_list)
print("Max accuracy: {}".format(np.max(acc_list)))
print("with peak value = {}".format(peak_value_list[max_index]))
print("with peak count = {}".format(peak_count_list[max_index]))