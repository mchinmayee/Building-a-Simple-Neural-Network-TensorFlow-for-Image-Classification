#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:15:07 2020

@author: chinu
"""

from pathlib import Path
import os,cv2
import shutil
import numpy as np
import tensorflow_hub as hub
import keras
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = '../cats_and_dogs_filtered/train'
test_dir = '../cats_and_dogs_filtered/validation'

model_file = '../cats_and_dogs_filtered/cats_vs_dogs_V1.h5'

train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(
                    batch_size=32,
                    directory=train_dir,
                    shuffle=True,
                    target_size=(224, 224),
                    class_mode='binary')

val_data_gen = train_image_generator.flow_from_directory(
                    batch_size=32,
                    directory=test_dir,
                    shuffle=False,
                    target_size=(224, 224),
                    class_mode='binary')
 
    
base_model= tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')
base_model.trainable = False

model = Sequential([ base_model,keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(1, activation='sigmoid')])
    
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
          loss='binary_crossentropy',
          metrics=['accuracy'])

print(model.summary())    
    

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=32,epochs=5,validation_data=val_data_gen, validation_steps=70)

model.save(model_file)

