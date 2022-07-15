# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:53:48 2021

@author: jdiet_000
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from skimage import io
from skimage.transform import resize
from keras.applications import VGG16
from SAB import spatial_attention_block

tf.keras.backend.clear_session()

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

def two_image_generator(train_datagen, 
                        directory, 
                        batch_size,
                        shuffle = False,
                        img_size1 = (256,256), 
                        img_size2 = (256,256)):

    gen1 = train_datagen.flow_from_directory(
        directory,
        classes = ['meningioma', 'glioma', 'pituitary'],
        target_size=img_size1,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle = shuffle,
        seed = 7)

    gen2 = train_datagen.flow_from_directory(
        directory,
        classes = ['meningioma', 'glioma', 'pituitary'],
        target_size=img_size2,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle = shuffle,
        seed = 7)  
  
    while True:
        X1i = gen1.next()
        X2i = gen2.next()
        yield [X1i[0], X2i[0]], X2i[1]    

train_generator = two_image_generator(train_datagen, 
                                      '.../mri_t1_classification_dataset2/train/',
                                      batch_size = 64,  
                                      shuffle = True)

validation_generator = two_image_generator(validation_datagen, 
                                      '.../mri_t1_classification_dataset2/validation/',
                                      batch_size = 32,  
                                      shuffle = False)

"===== pre-trained VGG16 ======================================================"

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

for layer in vgg_model.layers[:15]:
    layer.trainable = False
    

for i, layer in enumerate(vgg_model.layers):
    print(i, layer.name, layer.trainable)
    
    
x = vgg_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x) 

x = layers.Dropout(0.5)(x) 
x1vgg = layers.Dense(768, activation='relu')(x)

"===== Proposed ANSA model ===================================================="

inp=keras.Input(shape=(256,256,3))
#-----l2-SAB1 -----------------------------------------------------------------
x1=layers.Conv2D(64,(25,25),strides=2,name='last_conv3',activation='relu')(inp)
x1=layers.Conv2D(64,(1,1),strides=1,activation='relu')(x1)
K1=spatial_attention_block(kernel_size=16)(x1)
K1s=layers.Multiply()([K1,x1])
K1s=layers.UnitNormalization()(K1s)
x1=layers.UnitNormalization()(x1)
K=layers.Add()([K1s,x1])
# the first max pooling layer is changed to (2,2)
K1s=layers.MaxPooling2D((2,2))(K)
#------l2-SAB2 ----------------------------------------------------------------
conv2=layers.Conv2D(128,(13,13),strides=2,padding="same",name='last_conv2',activation='relu')(K1s)
conv2=layers.Conv2D(128,(1,1),strides=1,activation='relu')(conv2)
K2s=spatial_attention_block(kernel_size=8)(conv2)
K2=layers.Multiply()([K2s,conv2])   
#------Skip A -----------------------------------------------------------------    
K1=layers.MaxPooling2D((2,2))(K1)
K12=layers.Resizing(29,29)(K1)
K12=layers.Conv2D(128,(1,1))(K12)
K2=layers.Multiply()([K2,K12])  
#-------------------------------------------------------------------------------    
K2=layers.UnitNormalization()(K2)
conv2=layers.UnitNormalization()(conv2)
K=layers.Add()([K2,conv2])
K2=layers.MaxPooling2D((2,2))(K)
#------l2-SAB3 ----------------------------------------------------------------
conv3=layers.Conv2D(256,(9,9),strides=2,padding="same",name='last_conv1',activation='relu')(K2)
conv3=layers.Conv2D(256,(1,1),strides=1,activation='relu')(conv3)
K3=spatial_attention_block(kernel_size=4)(conv3)
K3=layers.Multiply()([K3,conv3])
#-----Skip C ------------------------------------------------------------------    
K1=layers.MaxPooling2D((2,2))(K1)
K13=layers.Resizing(7,7)(K1)
K13=layers.Conv2D(256,(1,1))(K13)
K3=layers.Multiply()([K3,K13])
#-----Skip B ------------------------------------------------------------------    
K2s=layers.MaxPooling2D((2,2))(K2s)
K23=layers.Resizing(7,7)(K2s)
K23=layers.Conv2D(256,(1,1))(K23)
K3=layers.Multiply()([K3,K23])
#-------------------------------------------------------------------------------   
K3=layers.UnitNormalization()(K3)
conv3=layers.UnitNormalization()(conv3)
K=layers.Add()([K3,conv3])
K3=layers.MaxPooling2D((2,2))(K)
#-----Classifier Head ---------------------------------------------------------    
x4=layers.Flatten()(K3)
x4=layers.Activation("relu")(x4)
x5=layers.Dense(1024,"relu")(x4)
x6=layers.Dense(256,"relu")(x5)

"=============================================================================="

x1vgg=layers.Dense(1024,"relu")(x1vgg)    
concat=layers.concatenate([x1vgg,x6])# concatenate both models
output=layers.Dropout(0.5)(concat)

"=============================================================================="

output=layers.Dense(256,"relu")(output)
output=layers.Dense(3)(output)
output=layers.Activation("softmax")(output)

full_model=Model(inputs=[vgg_model.input, inp], outputs=[output])

"=== COMPILE =================================================================="

opt1 = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9, clipnorm=1.)
opt2 = tf.keras.optimizers.RMSprop(lr=0.001)
opt3 = tf.keras.optimizers.Adam(learning_rate=0.000001,clipvalue=1.)
opt4 = tf.keras.optimizers.Adam(learning_rate=0.01,epsilon=0.1)
full_model.compile(loss='sparse_categorical_crossentropy',
optimizer=opt4,
metrics=['accuracy'], run_eagerly=True)

"=== TRAINING ================================================================="

class_weights = {0: 1.4461315979754157,
                1: 0.7183908045977011,
                2: 1.0911074740861975}
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)
model_checkpoint = ModelCheckpoint('Ensemble_classification.hdf5', monitor='loss',verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

history = full_model.fit(train_generator,
steps_per_epoch=(2145)//64,
epochs=80,
verbose=1,
validation_data = validation_generator,
validation_steps=306//32, callbacks=[lr_reduce, model_checkpoint, es])

"=== PREDICTIONS =============================================================="
full_model.load_weights('Ensemble_classification.hdf5')

txtfile_write = ".../predictedclass_joint.txt"# where to write the predictions
test_dir      = ".../mri_t1_classification_dataset2/test/"
txt_file_test = ".../mri_t1_classification_dataset2/test_labels.txt"# true labels created through data preparation
f=open(txtfile_write, "w")
P=[]
L=np.loadtxt(txt_file_test)

for i in range(613):
    fname=L.astype('int')[i,0]
    img = resize(io.imread(test_dir+str(fname)+'.png'),(256,256))
    img = np.expand_dims(img,axis = 0)
    p = full_model.predict([img,img])
    print("p = ", p)
    ind = np.where(p==np.max(p))
    predicted_class = ind[1][0]+1 # {1,2,3}
    P.append(predicted_class)
    print("predicted class = ", predicted_class)
    f.write(str(fname)+ ' ' + str(predicted_class) +"\n")
    
    print('iteration = ', i)
f.close()
P=np.asarray(P) # predicted_labels

"=== compute the classification accuracy ======================================"
from sklearn.metrics import accuracy_score

txt_file_test=".../mri_t1_classification_dataset2/test_labels.txt"

L=np.loadtxt(txt_file_test)# true labels
classification_accuracy=accuracy_score(L.astype('int')[:,1], P)
print('Classification accuracy = ', classification_accuracy)



