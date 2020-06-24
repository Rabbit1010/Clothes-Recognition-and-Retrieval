# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 16:14:17 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import time
import tensorflow as tf
import cv2

from yolov3_tf2.models import YoloV3


def Draw_Bounding_Box(img, list_obj):
    try:
        img = img.numpy() # convert tensor to numpy array
    except:
        pass

    img = np.squeeze(img)

    img_width = img.shape[1]
    img_height = img.shape[0]

    color_yellow = [244/255, 241/255, 66/255]
    color_green = [66/255, 241/255, 66/255]
    color_red = [241/255, 66/255, 66/255]

    # draw rectangle bounding box for cars
    for obj in list_obj:
        x1 = int(round(obj['x1']*img_width))
        y1 = int(round(obj['y1']*img_height))
        x2 = int(round(obj['x2']*img_width))
        y2 = int(round(obj['y2']*img_height))

        if obj['label'] == 'short_sleeve_top':
            color = color_yellow
        elif obj['label'] == 'trousers':
            color = color_red
        else:
            color = color_green

        text = '{}: {:.2f}'.format(obj['label'], obj['confidence'])

        # draw bounding box and labels
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        img = cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return img

def Read_Img_2_Tensor(img_path):
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=3, dtype=tf.dtypes.float32)
    img = tf.expand_dims(img, 0) # fake a batch axis

    return img

def Load_DeepFashion2_Yolov3():
    t1 = time.time()
    model = YoloV3(classes=13)
    model.load_weights('./built_model/deepfashion2_yolov3')
    t2 = time.time()
    print('Load DeepFashion2 Yolo-v3 from disk: {:.2f} sec'.format(t2 - t1))

    return model

def Save_Image(image_array, save_path):
    if image_array.dtype == 'float32':
        cv2.imwrite(save_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)*255)
    elif image_array.dtype == 'uint8':
        cv2.imwrite(save_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    else:
        raise ValueError('Unrecognize type of image array: {}', image_array.dtype)
