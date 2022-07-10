# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 13:21:59 2022

@author: dietlmj
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def model_AlexNet():

    inp=keras.Input(shape=(256,256,3))
    
    x1=layers.Conv2D(64,(25,25),strides=2)(inp)
    x1=layers.MaxPooling2D((4,4))(x1)

    x2=layers.Conv2D(128,(13,13),strides=2,padding="same")(x1)
    x2=layers.MaxPooling2D((2,2))(x2)

    x3=layers.Conv2D(256,(9,9),strides=2,padding="same",name='last_conv')(x2)
    x3=layers.MaxPooling2D((2,2))(x3)

    x4=layers.Flatten()(x3)
    x4=layers.Activation("relu")(x4)

    x5=layers.Dense(1024,"relu")(x4)
   
    x6=layers.Dense(256,"relu")(x5)

    x7=layers.Dense(3)(x6)

    out=layers.Activation("softmax")(x7)

    model=keras.Model(inputs=inp,outputs=out)
    
    return model
