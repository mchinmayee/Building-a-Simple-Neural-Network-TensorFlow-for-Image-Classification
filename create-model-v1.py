#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:15:07 2020

@author: Chinmayee Ojha
"""

from pathlib import Path
from random import shuffle
import os
import cv2
import shutil
import numpy as np
import tensorflow_hub as hub

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.models import load_model


train_dir = '../cats_and_dogs_filtered/train'
test_dir = '../cats_and_dogs_filtered/validation'
model_file = '../cats_and_dogs_filtered/cats_vs_dogs_V2.h5'

image_base_dir = '../cats_and_dogs_filtered/'
train_data_npz = image_base_dir + '/catdogData/cats_vs_dogs_training_data2.npz'
train_label_npz = image_base_dir + '/catdogData/cats_vs_dogs_training_label2.npz'
test_data_npz = image_base_dir + '/catdogData/cats_vs_dogs_testing_data2.npz'
test_label_npz = image_base_dir + '/catdogData/cats_vs_dogs_testing_label2.npz'


model_file = '../cats_and_dogs_filtered/cats_vs_dogs_V1.h5'

def get_filenames(mypath):
     image_file_ext = '*.jpg'
     file_names = [os.fspath(f) for f in Path(mypath).rglob(image_file_ext)]
     print('images loaded = {}'.format(len(file_names)))
     return file_names


def resize_image_save_image_label(image_count, img_dir, data_file, tag_file):
    dog_count=0
    cat_count = 0
    # image_count=1000
    images=[]
    labels = []
    size=150
    dog_label = [1,0]
    cat_label = [0,1]
    
    file_names = get_filenames(img_dir)
    shuffle(file_names)

    for i,file in enumerate(file_names):
        if os.path.split(file_names[i])[-1][0] == 'd' and dog_count < image_count:
            dog_count += 1
            image = cv2.imread(file)
            image = cv2.resize(image,(size,size), interpolation = cv2.INTER_AREA)
            labels.append(np.array(dog_label))
            images.append(image)
            
        if os.path.split(file_names[i])[-1][0] == 'c' and cat_count < image_count:
            cat_count += 1
            image = cv2.imread(file)
            image = cv2.resize(image,(size,size), interpolation = cv2.INTER_AREA)
            labels.append(np.array(cat_label))
            images.append(image)
            
        print('image dir: {}, dog count: {}, cat count: {}'.format(img_dir, dog_count, cat_count))
        if dog_count >= image_count and cat_count >= image_count:
            break
        
    np.savez(data_file, np.array(images))
    np.savez(tag_file, np.array(labels))        

            
def load_data_training_test(train_data_npz, train_label_npz, test_data_npz, test_label_npz):
    
    if os.path.exists(train_data_npz):
        npzfile = np.load(train_data_npz)
        train_data = npzfile['arr_0']
    
    if os.path.exists(train_label_npz):
        npzfile = np.load(train_label_npz)
        train_labels = npzfile['arr_0']
    
    if os.path.exists(test_data_npz):
        npzfile = np.load(test_data_npz)
        test_data = npzfile['arr_0']
    
    if os.path.exists(test_label_npz):
        npzfile = np.load(test_label_npz)
        test_labels = npzfile['arr_0']
    
    return (train_data, train_labels), (test_data, test_labels)
            
            
def generate_model(x_train, y_train):

    img_rows = x_train[0].shape[0]
    img_cols = x_train[1].shape[0]
    input_shape = (img_rows, img_cols, 3)
    base_model= tf.keras.applications.MobileNetV2(input_shape=input_shape,include_top=False,weights='imagenet')
    base_model.trainable = False
    model = Sequential([ base_model,keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(2, activation='sigmoid')])
       
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    print(model.summary())

    return model

def train_model(model,x_train, y_train, x_test, y_test):
    
    batch_size = 16
    epochs = 10
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
    model.save(model_file)
    return model
                
def main():
    if not os.path.exists(train_data_npz) and \
      not os.path.exists(train_label_npz) and \
      not os.path.exists(test_data_npz) and \
      not os.path.exists(test_label_npz):
       #training data
       resize_image_save_image_label(1000, train_dir, train_data_npz, train_label_npz)    

       #testing data
       resize_image_save_image_label(70, test_dir, test_data_npz, test_label_npz)
    
    (x_train, y_train), (x_test, y_test) = load_data_training_test(train_data_npz, train_label_npz, test_data_npz, test_label_npz)  
               
    y_train = y_train.reshape(y_train.shape[0], 2)
    y_test = y_test.reshape(y_test.shape[0], 2)
   
   #change image type to float32 data type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
   
   #normalize data by changing the range from (0 to 255) to (0 to 1)

    x_train /= 255
    x_test /= 255
   
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)    
    

    if os.path.exists(model_file):
       model = load_model(model_file)
    else:
       model = generate_model(x_train, y_train)
       model = train_model(model, x_train, y_train, x_test, y_test)
       
    # scores = model.evaluate(x_test,y_test, verbose=1)
    # print('Test loss:{}'.format(scores[0]))
    # print('Test accuracy:{}'.format(scores[1]))
            
main()
