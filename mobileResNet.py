# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 04:42:44 2019

@author: rilak
"""
import keras
import numpy as np
import keras.backend as K
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential

def model():
    model = Sequential()
    model.add(keras.applications.mobilenet.MobileNet(input_shape=x_train.shape[1:], alpha=1.0, depth_multiplier=3, dropout=1e-3, include_top=True, weights=None, input_tensor=None, pooling='max', classes=10))
    return model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
model = model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=10, epochs=1, validation_split=0.1)
