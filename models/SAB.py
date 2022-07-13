# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 08:24:55 2022

@author: dietlmj
"""

import tensorflow as tf

class spatial_attention_block(tf.keras.layers.Layer):

    def __init__(self, kernel_size=3, **kwargs):
        self.kernel_size = kernel_size
        super(spatial_attention_block, self).__init__(**kwargs)

    def get_config(self):
        config = super(spatial_attention_block, self).get_config().copy()
        config.update({
            'kernel_size': self.kernel_size
        })
        return config

    def build(self, input_shape):
        
        self.convolve = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size,
                                             strides=1, padding='same', activation='sigmoid',
                                             kernel_initializer='he_normal', use_bias=False)
        
        super(spatial_attention_block, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        
        min_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.min(x, axis=-1, keepdims=True))(inputs)
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(inputs)
                
        min_pool = tf.keras.layers.UnitNormalization()(min_pool)
        max_pool = tf.keras.layers.UnitNormalization()(max_pool)
        
        sub = tf.keras.layers.Subtract()([max_pool, min_pool])

        conv = self.convolve(sub)	
        
        result = tf.keras.layers.multiply([inputs, conv])
            
        return result



