# -*- coding: utf-8 -*-
"""
Created on Sat May 15 22:38:30 2021

@author: Sharaaf.Nazeer
"""

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

import os
import visualkeras

data_path = 'trainingDataSet/train'

directories = ['/Closed', '/Open', '/no_yawn', '/yawn']

batch_size = 128

image_size = 256

classes = 4

num_epochs = 100

model_path="myModels/detection_new3.h5"

def plot_imgs(directory, top=10):
    all_item_dirs = os.listdir(directory)
    item_files = [os.path.join(directory, file) for file in all_item_dirs][:5]
  
    plt.figure(figsize=(20, 20))
  
    for i, img_path in enumerate(item_files):
        plt.subplot(10, 10, i+1)
    
        img = plt.imread(img_path)
        plt.tight_layout()         
        plt.imshow(img, cmap='gray') 
        
for j in directories:
    plot_imgs(data_path+j)   
    
    

train_datagen = ImageDataGenerator(horizontal_flip = True, 
                                  rescale = 1./255, 
                                  zoom_range = 0.2, 
                                  validation_split = 0.1)

test_datagen = ImageDataGenerator(rescale = 1./255)



train_data_path = 'trainingDataSet/train'
test_data_path = 'trainingDataSet/test'

train_set = train_datagen.flow_from_directory(train_data_path, target_size = (256,256),
                                              batch_size = batch_size, 
                                              color_mode = 'grayscale',
                                              class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_data_path, target_size = (256,256),
                                              batch_size = batch_size, 
                                              color_mode = 'grayscale',
                                              class_mode = 'categorical')

print(train_datagen)


model = Sequential()
model.add(Conv2D(128, (3,3), padding = 'same', input_shape = (256,256,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(64, (3,3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(64,(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Dropout(0.4))

model.add(Conv2D(32,(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))

## for mulitclassification
model.add(Dense(classes, kernel_regularizer=regularizers.l2(0.01),activation
             ='softmax'))

model.compile(optimizer = 'adam', loss = 'squared_hinge', metrics = ['accuracy'])

print(model.summary())

visualkeras.layered_view(model, legend=True).show() # display using your system viewer

#exit()

checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, 
                              save_best_only=True, mode='max')

callbacks_list = [checkpoint]

training_steps=train_set.n//train_set.batch_size
validation_steps =test_set.n//test_set.batch_size


history = model.fit_generator(train_set, 
                              epochs=num_epochs, 
                              steps_per_epoch=training_steps,
                              validation_data=test_set,
                              validation_steps=validation_steps, 
                              callbacks = callbacks_list)

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()

#predict model
prediction = model.predict(test_set)
prediction = np.argmax(prediction,axis=1)

print(prediction)


true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())  

report = classification_report(true_classes, prediction, target_names=class_labels)
print(report) 

