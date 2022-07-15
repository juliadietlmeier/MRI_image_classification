# -*- coding: utf-8 -*-
"""
Created on Wed July 13 13:53:48 2022

@author: dietlmj
"""
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2
import os
from skimage import io
from skimage.transform import resize
from softattention_model import SoftAttention
from tensorflow.keras.layers import concatenate, Activation

from skimage import io
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Dense, Dropout

tf.keras.backend.clear_session()
        
batch_size=64

acc_array=[]
for k in range(20):
    
    train_datagen = ImageDataGenerator(rescale=1/255)

    validation_datagen = ImageDataGenerator(rescale=1/255)
                                        
    train_generator = train_datagen.flow_from_directory(
            '.../mri_t1_classification_dataset2/train/',  
            classes = ['meningioma', 'glioma', 'pituitary'],
            target_size=(256, 256),  
            batch_size=batch_size,
            shuffle=True,
            class_mode='sparse')

    validation_generator = validation_datagen.flow_from_directory(
            '.../mri_t1_classification_dataset2/validation/',  
            classes = ['meningioma', 'glioma', 'pituitary'],
            target_size=(256, 256),  
            batch_size=32,
            class_mode='sparse',
            shuffle=False)

    from models.AlexNet import model_AlexNet()
    #from models.proposed_ANSA import model_ANSA

#=== COMPILE ==================================================================

    opt1 = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9, clipnorm=1.)
    opt2 = tf.keras.optimizers.RMSprop(lr=0.001)
    opt3 = tf.keras.optimizers.Adam(learning_rate=0.000001,clipvalue=1.)# for vgg16*
    opt4 = tf.keras.optimizers.Adam(learning_rate=0.01,epsilon=0.1)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt4,
                  metrics=['accuracy'])

#=== TRAINING =================================================================

    class_weights = {0: 1.4461315979754157,
                     1: 0.7183908045977011,
                     2: 1.0911074740861975}

    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)
    model_checkpoint = ModelCheckpoint('model_classification.hdf5', monitor='loss',verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)


    history = model.fit(train_generator,
                        steps_per_epoch=(2145)//64,
                        epochs=50, # 200 epochs for model_2020()
                        verbose=1,
                        validation_data = validation_generator,
                        validation_steps=306//32, callbacks=[lr_reduce, model_checkpoint, es], class_weight=class_weights)
    
     
    model.load_weights('model_classification.hdf5')

    txtfile_write = ".../predictedclass.txt"
    test_dir      = ".../mri_t1_classification_dataset2/test/"
    txt_file_test = ".../mri_t1_classification_dataset2/test_labels.txt"
    f=open(txtfile_write, "w")
    P=[]
    L=np.loadtxt(txt_file_test)

    for i in range(613):
        fname=L.astype('int')[i,0]
        img = resize(io.imread(test_dir+str(fname)+'.png'),(256,256))
        img = np.expand_dims(img,axis = 0)
        p = model.predict(img)
        print("p = ", p)
        ind = np.where(p==np.max(p))
        predicted_class = ind[1][0]+1 # {1,2,3}
        P.append(predicted_class)
        print("predicted class = ", predicted_class)
        
        f.write(str(fname)+ ' ' + str(predicted_class) +"\n")
        print('iteration = ', i)
    f.close()
    P=np.asarray(P) # predicted_labels

#=== compute the classification accuracy ======================================

    txt_file_test=".../mri_t1_classification_dataset2/test_labels.txt"
    L=np.loadtxt(txt_file_test)# true labels
    classification_accuracy=accuracy_score(L.astype('int')[:,1], P)
    print('Classification accuracy = ', classification_accuracy)
    acc_array.append(classification_accuracy)
    print('iteration = ',k)

#------------------------------------------------------------------------------

acc_array=np.asarray(acc_array)
mean_acc=np.mean(acc_array)
std_acc=np.std(acc_array)
best_acc=np.max(acc_array)
print('Best result = ', best_acc)


