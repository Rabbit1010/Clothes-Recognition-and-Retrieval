# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 21:39:04 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import os

"""
Parse all .json file in a folder to Yolo format ,and save it to the same folder
of the training images.

Yolo format: (in .txt)
    <object-class> <x> <y> <width> <height>

<object-class> : integer number of object, range: [0, num_classes-1]
<x> = <absolute_x> / <image_width>, range: [0,1]
<y> = <absolute_y> / <image_height>, range: [0,1]
<width> = <absolute_width> / <image_width>, range: [0,1]
<height> = <absolute_height> / <image_height>, range: [0,1]
"""

in_dir_name = './DeepFashion2/train/annos'
out_dir_name = './DeepFashion2/train/image'

for filename in os.listdir(in_dir_name): # loop throught the entire folder
    if filename.lower().endswith('.json')==False:
        continue

    file_path = os.path.join(in_dir_name, filename)
#    print(file_path)
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        num_object = len(data)-2

        with open(os.path.join(out_dir_name, filename.split('.')[0]+'.txt'), 'w') as yolo_file:
            img = cv2.imread(os.path.join(out_dir_name, filename.split('.')[0]+'.jpg')) # load input image
            img_width = img.shape[1]
            img_height = img.shape[0]
            for i in range(1, num_object+1): # for each object
                class_ID = data['item{}'.format(i)]['category_id']-1
                x1 = data['item{}'.format(i)]['bounding_box'][0]
                y1 = data['item{}'.format(i)]['bounding_box'][1]
                x2 = data['item{}'.format(i)]['bounding_box'][2]
                y2 = data['item{}'.format(i)]['bounding_box'][3]

                yolo_x = (x1+x2)/img_width/2  # relative center x
                yolo_y = (y1+y2)/img_height/2 # relative center y
                yolo_width = (x2-x1)/img_width
                yolo_height = (y2-y1)/img_height

                # output YOLO format
                yolo_file.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_ID, yolo_x, yolo_y, yolo_width, yolo_height))

in_dir_name = './DeepFashion2/validation/annos'
out_dir_name = './DeepFashion2/validation/image'

for filename in os.listdir(in_dir_name): # loop throught the entire folder
    if filename.lower().endswith('.json')==False:
        continue

    file_path = os.path.join(in_dir_name, filename)
#    print(file_path)
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        num_object = len(data)-2

        with open(os.path.join(out_dir_name, filename.split('.')[0]+'.txt'), 'w') as yolo_file:
            img = cv2.imread(os.path.join(out_dir_name, filename.split('.')[0]+'.jpg')) # load input image
            img_width = img.shape[1]
            img_height = img.shape[0]
            for i in range(1, num_object+1): # for each object
                class_ID = data['item{}'.format(i)]['category_id']-1
                x1 = data['item{}'.format(i)]['bounding_box'][0]
                y1 = data['item{}'.format(i)]['bounding_box'][1]
                x2 = data['item{}'.format(i)]['bounding_box'][2]
                y2 = data['item{}'.format(i)]['bounding_box'][3]

                yolo_x = (x1+x2)/img_width/2  # relative center x
                yolo_y = (y1+y2)/img_height/2 # relative center y
                yolo_width = (x2-x1)/img_width
                yolo_height = (y2-y1)/img_height

                # output YOLO format
                yolo_file.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_ID, yolo_x, yolo_y, yolo_width, yolo_height))