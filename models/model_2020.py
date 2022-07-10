# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 15:35:11 2022

@author: dietlmj
"""

"""
We have replicated this network from: Milica M. Badza and Marko C. Barjaktarovic, 
”Classification of Brain Tumors from MRI Images Using a Convolutional Neural Network”,
Applied Sciences, 2020

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def model_2020():

    inp=keras.Input(shape=(256,256,3))

    x1=layers.Conv2D(16,(5,5),strides=2,padding='same',name='last_conv3',activation='relu')(inp)
    x1=layers.Dropout(0.5)(x1)
    x1=layers.MaxPooling2D((2,2),strides=2,padding='same')(x1)
    x1=layers.Conv2D(32,(3,3),strides=2,padding='same',name='last_conv2',activation='relu')(x1)
    x1=layers.Dropout(0.5)(x1)
    x1=layers.MaxPooling2D((2,2),strides=2,padding='same')(x1)
    x1=layers.Conv2D(64,(3,3),strides=1,padding='same',name='last_conv1',activation='relu')(x1)
    x1=layers.Dropout(0.5)(x1)
    x1=layers.MaxPooling2D((2,2),strides=2,padding='same')(x1)
    x1=layers.Conv2D(128,(3,3),strides=1,padding='same',name='last_conv',activation='relu')(x1)
    x1=layers.Dropout(0.5)(x1)
    x1=layers.MaxPooling2D((2,2),strides=2,padding='same')(x1)
    x4=layers.Flatten()(x1)
    x4=layers.Activation("relu")(x4)

    x5=layers.Dense(1024,"relu")(x4)
   
    x6=layers.Dense(3,"relu")(x5)
    out=layers.Activation("softmax")(x6)
    model=keras.Model(inputs=inp,outputs=out)
    
    return model