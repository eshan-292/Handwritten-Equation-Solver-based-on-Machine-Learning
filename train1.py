#!/usr/bin/env python
# coding: utf-8

# # PART 1

# In[ ]:


#IMPORTING THE LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#READING THE DATA AND SPLITTING THE CSV FILE
train = pd.read_csv('annotations.csv').iloc[0:45000,:]
test = pd.read_csv('annotations.csv').iloc[45000:,:]
# IMPORTING LIBRARIES FOR CNN
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#K.set_image_dim_ordering('th')
K.set_image_data_format("channels_first")
# TO CONVERT IMAGES TO TENSOR
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_gen = ImageDataGenerator(rescale = 1.255)
train_data = train_datagen.flow_from_dataframe(dataframe = train, 
directory = 'data/training_set', x_col = 'Image', 
y_col = 'Label', seed = 42,
batch_size = 100, shuffle = True, 
class_mode="categorical",target_size = (64, 64),color_mode='grayscale')

test_data = test_gen.flow_from_dataframe(dataframe = test, 
directory = 'data/test_set', x_col = 'Image', 
y_col ='Label',
batch_size = 100, shuffle = True, 
class_mode="categorical",target_size = (64, 64),color_mode='grayscale')

# CREATING AN OBJECT OF SEQUENTIAL CLASS
cnn = tf.keras.models.Sequential()

# ADDING CONVULUTION LAYERS
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 1]))

#POOLING
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
#ADDING MORE LAYERS
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#FLATTENING 
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# ADDING OUTPUT LAYER
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

#COMPILING VARIOUS LAYERS
cnn.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])

# FITTING THE MODEL WITH TRAINING SET AND PREDICTING ACCURACY ON TEST SET
cnn.fit(x = train_data,validation_data=test_data, epochs = 10)


#SAVING THE MODEL

model_json = cnn.to_json()
with open("model_final_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn.save_weights("model_final_1.h5")


