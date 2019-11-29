# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 11:26:29 2019

@author: Preeti
"""

import os, shutil

base_dir = 'D:/Users/Preeti/Desktop/6 Oct AI/cats and dogs small'
train_dir = 'D:/Users/Preeti/Desktop/6 Oct AI/cats and dogs small/train'
validation_dir = 'D:/Users/Preeti/Desktop/6 Oct AI/cats and dogs small/validation'
test_dir = 'D:/Users/Preeti/Desktop/6 Oct AI/cats and dogs small/test'
train_cats_dir = 'D:/Users/Preeti/Desktop/6 Oct AI/cats and dogs small/train/cats'
train_dogs_dir = 'D:/Users/Preeti/Desktop/6 Oct AI/cats and dogs small/train/dogs'
validation_cats_dir = 'D:/Users/Preeti/Desktop/6 Oct AI/cats and dogs small/validation/cats'
validation_dogs_dir = 'D:/Users/Preeti/Desktop/6 Oct AI/cats and dogs small/validation/dogs'
test_cats_dir = 'D:/Users/Preeti/Desktop/6 Oct AI/cats and dogs small/test/cats'
test_dogs_dir = 'D:/Users/Preeti/Desktop/6 Oct AI/cats and dogs small/test/dogs'


print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

from keras import optimizers
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=20,class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150),batch_size=20,class_mode='binary')

import PIL
from IPython.display import display
from PIL import Image
from sklearn.preprocessing import LabelEncoder
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)


model.save('cats_and_dogs_small_1.h5')
os.getcwd()

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()