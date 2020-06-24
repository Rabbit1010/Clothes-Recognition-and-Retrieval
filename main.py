# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 23:28:33 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from cloth_detection import Detect_Clothes_and_Crop
from utils_my import Read_Img_2_Tensor, Save_Image, Load_DeepFashion2_Yolov3

model = Load_DeepFashion2_Yolov3()

img_path = './images/test1.jpg'

# Read image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = Read_Img_2_Tensor(img_path)

# Clothes detection and crop the image
img_crop = Detect_Clothes_and_Crop(img_tensor, model)

# Transform the image to gray_scale
cloth_img = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)

# Pretrained classifer parameters
PEAK_COUNT_THRESHOLD = 0.02
PEAK_VALUE_THRESHOLD = 0.01

# Horizontal bins
horizontal_bin = np.mean(cloth_img, axis=1)
horizontal_bin_diff = horizontal_bin[1:] - horizontal_bin[0:-1]
peak_count = len(horizontal_bin_diff[horizontal_bin_diff>PEAK_VALUE_THRESHOLD])/len(horizontal_bin_diff)
if peak_count >= PEAK_COUNT_THRESHOLD:
    print("Class 1 (clothes wtih stripes)")
else:
    print("Class 0 (clothes without stripes)")


plt.imshow(img)
plt.title('Input image')
plt.show()

plt.imshow(img_crop)
plt.title('Cloth detection and crop')
plt.show()
Save_Image(img_crop, './images/test1_crop.jpg')