# -*- coding: utf-8 -*-
"""
Created on Sat May 15 22:38:30 2021

@author: Sharaaf.Nazeer
"""

from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import regularizers
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import cv2

import matplotlib.pyplot as plt
plt.style.use('dark_background')

import os


data_path = 'trainingDataSet/train'

directories = ['/Closed', '/Open', '/no_yawn', '/yawn']

batch_size = 128

image_size = 256

classes = 4

num_epochs = 100

test_datagen = ImageDataGenerator(rescale = 1./255)



test_data_path = 'trainingDataSet/train'

test_set = test_datagen.flow_from_directory(test_data_path, target_size = (256,256),
                                              batch_size = batch_size, 
                                              color_mode = 'grayscale',
                                              class_mode = 'categorical')

model = load_model('myModels/detection_new2.h5')

#predict model
#prediction = model.predict(test_set)

print(test_set)


test_steps_per_epoch = np.math.ceil(test_set.samples / test_set.batch_size)

prediction = model.predict(test_set)
# Get most likely class
prediction = np.argmax(prediction,axis=1)
#print(prediction)

#print('Confusion Matrix')
#print(confusion_matrix(test_set.classes, prediction))

true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())  



report = classification_report(true_classes, prediction, target_names=class_labels)
print(report)  


def prepare(filepath):
    image = load_img(filepath)
    print(image)
    img_array = img_to_array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    eyeImage = cv2.resize(gray, (image_size, image_size))      
    resized_array = cv2.resize(eyeImage, (image_size, image_size))
    resized_array = resized_array / 255
    #print(resized_array)
    resized_array = np.expand_dims(resized_array, axis=0)
    resized_array = np.expand_dims(resized_array, axis=3)
    print(resized_array.shape)
    return resized_array

#print(prepare("trainingDataSet/test/Closed/_3.jpg"))
print(class_labels)
prediction = model.predict([prepare("neededData\dataset\set_01\Closed_Eyes\s0001_01480_0_1_0_0_0_01.png")])
print(np.argmax(prediction))