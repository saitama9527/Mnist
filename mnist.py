import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import os

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28)/255.
print(X_train)
X_test = X_test.reshape(-1, 1,28, 28)/255.
print(X_test)
y_train = np_utils.to_categorical(y_train, num_classes=10)
print(y_train)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print(y_test)

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
                       pool_size=2,
                       strides=2,
                       padding='same',    # Padding method
                       data_format='channels_first',
                       ))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))




