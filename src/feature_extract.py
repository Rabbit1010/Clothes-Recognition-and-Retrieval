#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Extract eigen vewctor from image
# Jerry ZJ, 2019/6/2

from PIL import Image
from os import listdir, getcwd
from os.path import isfile, isdir, join
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
    img = Image.open(file)
