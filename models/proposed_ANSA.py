# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 08:51:40 2022

@author: dietlmj
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from SAB import spatial_attention_block

def model_ANSA():

    inp=keras.Input(shape=(256,256,3))
#--------- l2-SAB1 ------------------------------------------------------------  
  
    x1=layers.Conv2D(64,(25,25),strides=2,name='last_conv3',activation='relu')(inp)
    x1=layers.Conv2D(64,(1,1),strides=1,activation='relu')(x1)    
    K1=spatial_attention_block(kernel_size=16)(x1)
    K1s=layers.Multiply()([K1,x1])
    
#------------------------------------------------------------------------------

    K1s=layers.UnitNormalization()(K1s)
    x1=layers.UnitNormalization()(x1)
    K=layers.Add()([K1s,x1])    
    K1s=layers.MaxPooling2D((4,4))(K)
    
#----------l2-SAB2 ------------------------------------------------------------

    conv2=layers.Conv2D(128,(13,13),strides=2,padding="same",name='last_conv2',activation='relu')(K1s)
    conv2=layers.Conv2D(128,(1,1),strides=1,activation='relu')(conv2)
    K2s=spatial_attention_block(kernel_size=8)(conv2)
    K2=layers.Multiply()([K2s,conv2])
    
#-------- Skip A -------------------------------------------------------------- 
   
    K1=layers.MaxPooling2D((2,2))(K1)
    K12=layers.Resizing(15,15,interpolation="nearest")(K1)
    K12=layers.Conv2D(128,(1,1))(K12)
    K2=layers.Multiply()([K2,K12])
    
#-------------------------------------------------------------------------------    
   
    K2=layers.UnitNormalization()(K2)
    conv2=layers.UnitNormalization()(conv2)    
    K=layers.Add()([K2,conv2])
    K2=layers.MaxPooling2D((2,2))(K)
    
#---------l2-SAB3 -------------------------------------------------------------

    conv3=layers.Conv2D(256,(9,9),strides=2,padding="same",name='last_conv1',activation='relu')(K2)
    conv3=layers.Conv2D(256,(1,1),strides=1,activation='relu')(conv3)
    K3=spatial_attention_block(kernel_size=4)(conv3)
    K3=layers.Multiply()([K3,conv3])

#--------Skip C ---------------------------------------------------------------
    
    K1=layers.MaxPooling2D((2,2))(K1)
    K13=layers.Resizing(4,4,interpolation="nearest")(K1)
    K13=layers.Conv2D(256,(1,1))(K13)
    K3=layers.Multiply()([K3,K13])
    
#--------Skip B --------------------------------------------------------------- 
   
    K2s=layers.MaxPooling2D((2,2))(K2s)
    K23=layers.Resizing(4,4,interpolation="nearest")(K2s)
    K23=layers.Conv2D(256,(1,1))(K23)    
    K3=layers.Multiply()([K3,K23])
    
#-------------------------------------------------------------------------------   

    K3=layers.UnitNormalization()(K3)
    conv3=layers.UnitNormalization()(conv3)       
    K=layers.Add()([K3,conv3])
    K3=layers.MaxPooling2D((2,2))(K)

#-------Classifier Head ------------------------------------------------------- 
   
    x4=layers.Flatten()(K3)
    x4=layers.Activation("relu")(x4)
    x5=layers.Dense(1024,"relu")(x4)   
    x6=layers.Dense(256,"relu")(x5)
    x7=layers.Dense(3)(x6) 

    out=layers.Activation("softmax")(x7)

    model=keras.Model(inputs=inp,outputs=out)
    
    return model
