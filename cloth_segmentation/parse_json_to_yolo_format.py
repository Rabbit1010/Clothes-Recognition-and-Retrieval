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

in_dir_name = './DeepFashion2/validation/annos'
out_dir_name = './DeepFashion2/validation/image'

for filename in os.listdir(in_dir_name): # loop throught the entire folder
    if filename.lower().endswith('.json')==False:
        continue

    print(filename)

    file_path = os.path.join(in_dir_name, filename)
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        num_object = len(data)-2

        with open(os.path.join(out_dir_name, filename.split('.')[0]+'.txt'), 'w') as yolo_file:
            img = cv2.imread(os.path.join(out_dir_name, filename.split('.')[0]+'.jpg')) # load input image
            img_width = img.shape[1]
            img_height = img.shape[0]
            for i in range(1, num_object+1): # for each object
                class_ID = data['item{}'.format(i)]['category_id']
                x1 = data['item{}'.format(i)]['bounding_box'][0]
                y1 = data['item{}'.format(i)]['bounding_box'][1]
                x2 = data['item{}'.format(i)]['bounding_box'][2]
                y2 = data['item{}'.format(i)]['bounding_box'][3]
                width = x2 - x1
                height = y2 - y1

                # output YOLO format
                yolo_file.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_ID, x1/img_width, y1/img_height, width/img_width, height/img_height))

                # show image for debugging
#                cv2.rectangle(img, (x1,y1), (x2,y2), (66,238,244), 3)

#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Change it to RGB for all models to work properly
#    plt.imshow(img)
#    plt.show()