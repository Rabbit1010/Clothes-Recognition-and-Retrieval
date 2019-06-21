# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 23:11:51 2019

@author: Wei-Hsiang, Shen
"""

from tensorflow.keras import layers


class ResBlock(object):
    """
    Residual block (non bottleneck, 2 blocks)
    """
    def __init__(self, num_feature_in, num_feature_out, strides=(1,1)):
        self.num_feature_in = num_feature_in
        self.num_feature_out = num_feature_out
        self.strides = strides

    def __call__(self, x):
        shortcut = x

        x = layers.Conv2D(self.num_feature_out, (3,3), strides=self.strides, use_bias=False, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(self.num_feature_out, (3,3), use_bias=False, padding='same')(x)
        x = layers.BatchNormalization()(x)

        # projection shortcut is used when input and output are different dimensions
        if self.num_feature_in!=self.num_feature_out or self.strides != (1, 1):
            shortcut = layers.Conv2D(self.num_feature_out, (1, 1), strides=self.strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.add([x, shortcut]) # skip connection
        x = layers.ReLU()(x)
        return x

class ResBlock_Bottleneck(object):
    """
    Residual block (bottleneck, 3 blocks)
    """
    def __init__(self, num_feature_in, num_feature_mid, num_feature_out, strides=(1,1)):
        self.num_feature_in = num_feature_in
        self.num_feature_mid = num_feature_mid
        self.num_feature_out = num_feature_out
        self.strides = strides

    def __call__(self, x):
        shortcut = x

        x = layers.Conv2D(self.num_feature_mid, kernel_size=(1,1), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(self.num_feature_mid, kernel_size=(3,3), use_bias=False, strides=self.strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(self.num_feature_out, kernel_size=(1,1), use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        # projection shortcut is used when input and output are different dimensions
        if self.num_feature_in!=self.num_feature_out or self.strides != (1, 1):
            shortcut = layers.Conv2D(self.num_feature_out, kernel_size=(1, 1), use_bias=False, strides=self.strides)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        return x