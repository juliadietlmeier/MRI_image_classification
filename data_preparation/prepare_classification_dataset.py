# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 07:58:04 2022

@author: dietlmj
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 13:35:07 2022

@author: dietlmj
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
from skimage import io
import cv2

"== create the following empty directories ===================================="
train_mdir      = ".../mri_t1_classification_dataset2/train/meningioma/"
train_gdir      = ".../mri_t1_classification_dataset2/train/glioma/"
train_pdir      = ".../mri_t1_classification_dataset2/train/pituitary/"

val_mdir        = ".../mri_t1_classification_dataset2/validation/meningioma/"
val_gdir        = ".../mri_t1_classification_dataset2/validation/glioma/"
val_pdir        = ".../mri_t1_classification_dataset2/validation/pituitary/"

test_dir        = ".../mri_t1_classification_dataset2/test/"

"=============================================================================="

src_dir=".../Brain_tumor_dataset/"# this is where all original .mat files are
path, dirs, files = next(os.walk(os.path.join(src_dir)))

"=============================================================================="
"=== our SPLIT is 70% train 10% validation and 20% test ======================="

txt_file_train=".../mri_t1_classification_dataset2/data_split_train.txt"
txt_file_val  =".../mri_t1_classification_dataset2/data_split_val.txt"
txt_file_test =".../mri_t1_classification_dataset2/data_split_test.txt"

"=============================================================================="

idx_train=np.loadtxt(txt_file_train)
idx_val=np.loadtxt(txt_file_val)
idx_test=np.loadtxt(txt_file_test)

idx_train=idx_train.astype('int')
idx_val=idx_val.astype('int')
idx_test=idx_test.astype('int')

"== TRAINing data preparation ================================================="
txt_file=".../mri_t1_classification_dataset2/train_labels.txt"
fl = open(txt_file, 'w')

for i in range(len(idx_train)):
    f=files[idx_train[i]]
    fname=f.split('.')[0]+'.png'
    data=h5py.File(os.path.join(src_dir,f),"r")
    image=list(data["cjdata"]["image"])#.value
    image=np.asarray(image)
    image=np.stack((image,)*3, axis=-1)
    label = list(data["cjdata"]["label"])
    label=round(label[0][0])
    if label==1:
        txt='1'#meningioma
        io.imsave(train_mdir+fname,image)
    elif label==2:
        txt='2'#glioma
        io.imsave(train_gdir+fname,image)
    elif label==3:
        txt='3'#pituitary 
        io.imsave(train_pdir+fname,image)
        
    fl.write(f.split('.')[0] + ' ' + txt+'\n')    

    
    print('i=',i)
fl.close()    
"== VALIDATION data preparation ==============================================="
txt_file=".../mri_t1_classification_dataset2/val_labels.txt"
fl = open(txt_file, 'w')

for i in range(len(idx_val)):
    f=files[idx_val[i]]
    fname=f.split('.')[0]+'.png'
    data=h5py.File(os.path.join(src_dir,f),"r")
    image=list(data["cjdata"]["image"])#.value
    image=np.asarray(image)
    image=np.stack((image,)*3, axis=-1)
    label = list(data["cjdata"]["label"])
    label=round(label[0][0])
    if label==1:
        txt='1'#meningioma
        io.imsave(val_mdir+fname,image)
    elif label==2:
        txt='2'#glioma'
        io.imsave(val_gdir+fname,image)
    elif label==3:
        txt='3'#pituitary 
        io.imsave(val_pdir+fname,image)
        
    fl.write(f.split('.')[0] + ' '+ txt+'\n') 
    
    print('i=',i)    
fl.close()    
"== TEST data preparation ====================================================="
txt_file=".../mri_t1_classification_dataset2/test_labels.txt"
fl = open(txt_file, 'w')

for i in range(len(idx_test)):
    f=files[idx_test[i]]
    fname=f.split('.')[0]+'.png'
    data=h5py.File(os.path.join(src_dir,f),"r")
    image=list(data["cjdata"]["image"])#.value
    image=np.asarray(image)
    image=np.stack((image,)*3, axis=-1)
    label = list(data["cjdata"]["label"])
    label=round(label[0][0])
    if label==1:
        txt='1'#meningioma
        io.imsave(test_dir+fname,image)
    elif label==2:
        txt='2'#glioma'
        io.imsave(test_dir+fname,image)
    elif label==3:
        txt='3'#pituitary 
        io.imsave(test_dir+fname,image)
        
    fl.write(f.split('.')[0] + ' '+ txt+'\n')
    
    print('i=',i)        
fl.close()        
