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