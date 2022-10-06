# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 09:12:55 2022

@author: dietlmj
"""

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,InputSpec
import tensorflow.keras.layers as kl
import tensorflow as tf
import numpy as np

class spatial_attention_CBAM(tf.keras.layers.Layer):
    """ spatial attention module 
        
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, kernel_size=7, **kwargs):
        self.kernel_size = kernel_size
        super(spatial_attention_CBAM, self).__init__(**kwargs)

    def get_config(self):
        config = super(spatial_attention_CBAM, self).get_config().copy()
        config.update({
            'kernel_size': self.kernel_size
        })
        return config

    def build(self, input_shape):
        self.conv3d = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size,
                                             strides=1, padding='same', activation='sigmoid',# was 'sigmoid'
                                             kernel_initializer='he_normal', use_bias=False)
        super(spatial_attention_CBAM, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(inputs)
        
        concat = tf.keras.layers.Concatenate()([max_pool, avg_pool])

        feature = self.conv3d(concat)	
            
        return tf.keras.layers.multiply([inputs, feature])