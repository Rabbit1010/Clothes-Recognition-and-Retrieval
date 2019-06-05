#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Extract eigen vewctor from image
# Jerry ZJ, 2019/6/2

from PIL import Image
from os import listdir, getcwd
from os.path import isfile, isdir, join
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import glob

# List .png files
def list_files(extension):
    list = []
    for file in glob.glob("*"+extension):
        list.append(file)
    return list

extension = ".jpg"
jpg_files = list_files(extension)
print(jpg_files)
for file in jpg_files:
    img = cv2.imread(file, 0)
    feature = np.zeros_like(img)
    orb = cv2.ORB_create()
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    img2 = cv2.drawKeypoints(img,kp,feature,color=(0,255,0), flags=0)
    print(feature)
    plt.imshow(img2),plt.show()

