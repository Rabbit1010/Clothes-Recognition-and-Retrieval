# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 23:59:44 2019

@author: Wei-Hsiang, Shen
"""

import os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

dir_name = 'DeepFashion2/train/image/'
with open('./deepfashion2-train.txt', 'w') as train_txt:
    for filename in os.listdir(dir_name): # loop throught the entire folder
        if filename.lower().endswith('.jpg')==False:
            continue
        train_txt.write(current_dir + "/" + dir_name + filename + "\n")

dir_name = 'DeepFashion2/validation/image/'
with open('./deepfashion2-validation.txt', 'w') as train_txt:
    for filename in os.listdir(dir_name): # loop throught the entire folder
        if filename.lower().endswith('.jpg')==False:
            continue
        train_txt.write(current_dir + "/" + dir_name + filename + "\n")

## Percentage of images to be used for the test set
#percentage_test = 10;
#
## Create and/or truncate train.txt and test.txt
#file_train = open('train.txt', 'w')
#file_test = open('test.txt', 'w')
#
## Populate train.txt and test.txt
#counter = 1
#index_test = round(100 / percentage_test)
#
#for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
#    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
#
#if counter == index_test:
#        counter = 1
#        file_test.write(current_dir + "/" + title + '.jpg' + "\n")
#    else:
#        file_train.write(current_dir + "/" + title + '.jpg' + "\n")
#        counter = counter + 1