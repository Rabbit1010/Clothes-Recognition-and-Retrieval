#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Convert RGBA png files to RGB JPG files
# Jerry ZJ, 2019/5/31

from PIL import Image
from os import listdir, getcwd
from os.path import isfile, isdir, join
import sys
import glob

# Backgroung fill color
fill_color = (255,255,255)

# List .png files
def list_files(extension):
    list = []
    for file in glob.glob("*"+extension):
        list.append(file)
    return list

extension = ".png"
png_files = list_files(extension)
print(png_files)
for file in png_files:
    img = Image.open(file)
    img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        background = Image.new(img.mode[:-1], img.size, fill_color)
        background.paste(img, img.split()[-1]) # omit transparency
        img = background
    name = file.split(".",1)
    img.convert("RGB").save(name[0]+".jpg")
