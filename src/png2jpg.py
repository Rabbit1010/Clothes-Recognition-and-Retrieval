#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Convert RGBA png files to RGB JPG files
# Jerry ZJ, 2019/5/31

from PIL import Image
from os import listdir, getcwd
from os.path import isfile, isdir, join
from subprocess import call
import sys
import glob
import cv2

def list_files(extension):
    list = []
    for file in glob.glob("*.png"):
        list.append(file)
    return list

extension = "png"
png_files = list_files(extension)
print(png_files)
for file in png_files:
    rgba_img = cv2.imread(file) 
    rgb_img = cv2.cvtColor(rgba_img,cv2.COLOR_RGBA2RGB)
    name = file.split(".",1)
    cv2.imwrite(name[0]+".jpg",rgb_img)
