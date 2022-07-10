# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 15:24:55 2022

@author: dietlmj
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm

def model_S():# VGG16* model from the paper
# Block 1
    inp=keras.Input(shape=(256,256,3))
    x1=layers.Conv2D(8, (3,3), kernel_constraint=max_norm(2.0), activation='relu')(inp)
    x1=layers.Conv2D(8, (3,3), kernel_constraint=max_norm(2.0), activation='relu')(x1)
    x1=layers.MaxPooling2D((2,2))(x1)   
# Block 2   
    x2=layers.Conv2D(16, (3,3), kernel_constraint=max_norm(2.0), activation='relu')(x1)
    x2=layers.Conv2D(16, (3,3), kernel_constraint=max_norm(2.0), activation='relu')(x2)
    x2=layers.MaxPooling2D((2,2))(x2)
# Block 3
    x3=layers.Conv2D(32, (3,3), kernel_constraint=max_norm(2.0), activation='relu')(x2)
    x3=layers.Conv2D(32, (3,3), kernel_constraint=max_norm(2.0), activation='relu',name='last_conv')(x3)
    x3=layers.MaxPooling2D((2,2))(x3)

    x4=layers.Flatten()(x3)
    x4=layers.Activation("relu")(x4)

    x5=layers.Dense(1024,"relu")(x4)
   
    x6=layers.Dense(256,"relu")(x5)

    x7=layers.Dense(3)(x6)

    out=layers.Activation("softmax")(x7)

    model=keras.Model(inputs=inp,outputs=out)
    
    return model