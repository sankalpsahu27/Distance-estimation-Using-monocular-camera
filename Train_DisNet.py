# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 01:38:53 2021

@author: Admin
"""
import os
import numpy as np
import json
import csv 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import keras.backend as k
from keras.layers import Dense, Activation
from keras.layers import InputLayer, Input
from keras.models import Sequential, Model, load_model
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.constraints import max_norm


def construct_DisNet_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(6,)))
    model.add(Dense(100, kernel_constraint=max_norm(2.), activation="selu"))
    model.add(Dense(100, kernel_constraint=max_norm(2.), activation="selu"))
    model.add(Dense(100, kernel_constraint=max_norm(2.), activation="selu"))
    model.add(Dense(1, activation="selu"))
    optimizer = Adam(1e-4)
    model.compile(optimizer=optimizer,
                  loss="mean_absolute_error",metrics=['accuracy'])
    return model 


df_train = pd.read_csv("train.csv")
X_train = df_train[['height','width','diagonal', 'size_h', 'size_w', 'size_d']].values
y_train = df_train[['zloc']].values
    
df_test = pd.read_csv("test.csv") 
X_test = df_test[['height','width','diagonal', 'size_h', 'size_w', 'size_d']].values
y_test = df_test[['zloc']].values


print("Train data shape: {}".format(X_train.shape))

DisNet_model_dir = "DisNet_Model"
DisNet_checkpoints = os.path.join(DisNet_model_dir, "best_disnet_model.keras")
if not os.path.exists(DisNet_model_dir):
    os.makedirs(DisNet_model_dir)

print("*******************************************************************************************")
if os.path.exists(DisNet_checkpoints):
    print("Continue training from checkpoints ...")
    model = load_model(DisNet_checkpoints)
else:
    print("No model checkpoints founded, construct new model...")
    model = construct_DisNet_model()
print("*******************************************************************************************")
callbacks = [EarlyStopping(monitor='val_loss', patience=200, verbose=1),
             ModelCheckpoint(filepath=DisNet_checkpoints, verbose=1, save_best_only=True)]

history = model.fit(x=X_train,
                        y=y_train,
                        epochs=100,
                        callbacks = callbacks,
                        verbose=1,
                        batch_size=50,
                        validation_data=(X_test, y_test))
    
scores = model.evaluate(X_train, y_train, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save_weights('my_model_weights.h5')
plt.plot(history.history['accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

