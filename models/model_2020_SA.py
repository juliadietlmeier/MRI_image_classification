# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:04:08 2022

@author: dietlmj
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from SAB import spatial_attention_block

def model_2020_SA():

    inp=keras.Input(shape=(256,256,3))

    x1=layers.Conv2D(16,(5,5),strides=2,padding='same',name='last_conv3',activation='relu')(inp)
    
    K1=spatial_attention_block(kernel_size=16)(x1)
    K1s=layers.Multiply()([K1,x1])
    K1s=layers.UnitNormalization()(K1s)
    x1=layers.UnitNormalization()(x1)
    K=layers.Add()([K1s,x1]) 
    
    #x1=layers.Dropout(0.5)(K)
    x1=layers.MaxPooling2D((2,2),strides=2,padding='same')(K)
    x1=layers.Conv2D(32,(3,3),strides=2,padding='same',name='last_conv4',activation='relu')(x1)
    
    K1=spatial_attention_block(kernel_size=8)(x1)
    K1s=layers.Multiply()([K1,x1])
    K1s=layers.UnitNormalization()(K1s)
    x1=layers.UnitNormalization()(x1)
    K=layers.Add()([K1s,x1]) 
    
    #x1=layers.Dropout(0.5)(K)
    x1=layers.MaxPooling2D((2,2),strides=2,padding='same')(K)
    x1=layers.Conv2D(64,(3,3),strides=1,padding='same',name='last_conv5',activation='relu')(x1)
    
    K1=spatial_attention_block(kernel_size=4)(x1)
    K1s=layers.Multiply()([K1,x1])
    K1s=layers.UnitNormalization()(K1s)
    x1=layers.UnitNormalization()(x1)
    K=layers.Add()([K1s,x1]) 
    
    #x1=layers.Dropout(0.5)(K)
    x1=layers.MaxPooling2D((2,2),strides=2,padding='same')(K)
    x1=layers.Conv2D(128,(3,3),strides=1,padding='same',name='last_conv6',activation='relu')(x1)
    
    K1=spatial_attention_block(kernel_size=4)(x1)
    K1s=layers.Multiply()([K1,x1])
    K1s=layers.UnitNormalization()(K1s)
    x1=layers.UnitNormalization()(x1)
    K=layers.Add()([K1s,x1]) 
    
    #x1=layers.Dropout(0.5)(K)
    x1=layers.MaxPooling2D((2,2),strides=2,padding='same')(K)
    x4=layers.Flatten()(x1)
    x4=layers.Activation("relu")(x4)

    x5=layers.Dense(1024,"relu")(x4)
   
    x6=layers.Dense(3,"relu")(x5)
    pred=layers.Activation("softmax")(x6)
    model=keras.Model(inputs=inp, outputs=pred)
    
    return model

