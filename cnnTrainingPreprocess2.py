# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:36:08 2021

@author: Sharaaf.Nazeer
"""

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
import matplotlib.image as mpimg
import dlib
import os
from helpers import faceCropper


train_data_path = 'neededData/dataset/set_03/train'

train_save_path = 'trainingDataSet/train/'

test_data_path = 'neededData/dataset/set_03/test'

test_save_path = 'trainingDataSet/test/'

image_size = 256

def readImage(imagePath):
  image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
  #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  return image

def crop_face_for_yawn():
    
    cropper = faceCropper.FaceCropper("models/dlib/shape_predictor_68_face_landmarks.dat")
    categories = ["yawn", "no_yawn"]
    
    otherCategories = ["Opened_mouth", "Closed_mouth"]
    for category in categories:
        path_link_train = os.path.join(train_data_path, category)
        for image in os.listdir(path_link_train):
            print(image)           
            
            image_array = readImage(os.path.join(path_link_train, image))
            shape, gray, face = cropper.runDetector(image_array)  
            
            if shape is not None:
                faceImage = cropper.extractFace(face, gray)            
                if faceImage is not None:
                    cv2.imwrite(train_save_path + category + '/' + image, faceImage)
                    
                    mouthImage = cropper.extractMouth(shape, gray)   
                    plt.imshow(cv2.cvtColor(mouthImage, cv2.COLOR_BGR2RGB))
                    if(category == 'yawn'):
                        print(train_save_path + otherCategories[0] + '/' + image)
                        cv2.imwrite(train_save_path + otherCategories[0] + '/' + image, mouthImage)          
                    else:
                        cv2.imwrite(train_save_path + otherCategories[1] + '/' + image, mouthImage)          
            
        
    for category in categories:
        path_link_test = os.path.join(test_data_path, category)
        for image in os.listdir(path_link_test):
            print(image)
            
            image_array = readImage(os.path.join(path_link_test, image))
            shape, gray, rects = cropper.runDetector(image_array)

            if shape is not None:               
                faceImage = cropper.extractFace(rects, gray)            
                if faceImage is not None:
                    cv2.imwrite(test_save_path + category + '/' + image, faceImage)
                    
                    mouthImage = cropper.extractMouth(shape, gray)                
                    if(category == 'yawn'):
                        cv2.imwrite(test_save_path + otherCategories[0] + '/' + image, mouthImage)          
                    else:
                        cv2.imwrite(test_save_path + otherCategories[1] + '/' + image, mouthImage) 


def resize_eyes():
    categories = ["Open", "Closed"]
    for category in categories:
        path_link_train = os.path.join(train_data_path, category)
        for image in os.listdir(path_link_train):
            print(image)
            image_array = cv2.imread(os.path.join(path_link_train, image), cv2.IMREAD_COLOR)
            
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            eyeImage = cv2.resize(gray, (image_size, image_size))         
            cv2.imwrite(train_save_path + category + '/' + image, eyeImage)
            
        
    for category in categories:
        path_link_test = os.path.join(test_data_path, category)
        for image in os.listdir(path_link_test):
            print(image)
            image_array = cv2.imread(os.path.join(path_link_test, image), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            eyeImage = cv2.resize(gray, (image_size, image_size))         
            cv2.imwrite(test_save_path + category + '/' + image, eyeImage)

crop_face_for_yawn()
#resize_eyes()


