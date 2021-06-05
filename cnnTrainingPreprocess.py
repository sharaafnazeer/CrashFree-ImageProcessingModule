# -*- coding: utf-8 -*-
"""
Created on Sat May 15 22:38:30 2021

@author: Sharaaf.Nazeer
"""

import cv2
from helpers import faceCropper

import matplotlib.pyplot as plt
plt.style.use('dark_background')

import os


train_data_path = 'neededData/dataset/set_03/train'

train_save_path = 'trainingDataSet/train/'

test_data_path = 'neededData/dataset/set_03/test'

test_save_path = 'trainingDataSet/test/'

image_size = 256

def crop_face_for_yawn():
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link_train = os.path.join(train_data_path, category)
        for image in os.listdir(path_link_train):
            print(image)
            image_array = cv2.imread(os.path.join(path_link_train, image), cv2.IMREAD_COLOR)
            plt.imshow(image_array) 
            cropper = faceCropper.FaceCropper()
            faceImage = cropper.generate(image_array, size = image_size)   
            print(faceImage)    
            if faceImage is not None:
                cv2.imwrite(train_save_path + category + '/' + image, faceImage)
            
        
    for category in categories:
        path_link_test = os.path.join(test_data_path, category)
        for image in os.listdir(path_link_test):
            print(image)
            image_array = cv2.imread(os.path.join(path_link_test, image), cv2.IMREAD_COLOR)
            plt.imshow(image_array) 
            cropper = faceCropper.FaceCropper()
            faceImage = cropper.generate(image_array, size = image_size)
            print(faceImage)            
            if faceImage is not None:
                cv2.imwrite(test_save_path + category + '/' + image, faceImage)


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
resize_eyes()


