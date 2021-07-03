#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# PART 2

# # IMPORTING LIBRARIES FOR FUTURE USE AND IMAGE MODIFICATION
# 

# In[ ]:


import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
np.random.seed(1212)

import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
#K.set_image_dim_ordering('th')
K.set_image_data_format("channels_first")


# # SPLITTING THE IMAGES

# # MANUAL LABELLING OF AROUND 3300 IMAGES

# # GENERATING TRAINING SET FROM SPLITTED IMAGES(RECOGNISING ALL 14 CHARACTERS(10 DIGITS AND 4 OPERATORS))

# In[ ]:


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory(directory='NEW/',
                                    
                                                 target_size = (64, 64),
                                                 batch_size = 5,
                                                 class_mode = "categorical",color_mode = 'grayscale')


# # CREATING THE VARIOUS LAYERS AND TRAINING THE MODEL ON TRAINING SET

# In[ ]:


cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 1]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=200, activation='relu'))

cnn.add(tf.keras.layers.Dense(units=14, activation='softmax'))

cnn.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])

cnn.fit(x = training_set,validation_data=training_set, epochs = 50)


# The following code was used to segregate more images from the dataset and later use them to retrain our model for better results 

from PIL import Image,ImageOps
import cv2
import numpy as np


# GET DIGIT FUNCTION TO GET INDICE CORRESPONDING TO THE DIGIT
def get_digit(arr):
    for i in range(0,len(arr)):
        if arr[i]==1:
            return i
    return 0
# TEST SIZE 
 #should be less than 5000
# FOR EVERY IMAGE IN TEST_SET
for i in range(1,2000):

 
    # Opens a image in RGB mode
        im = Image.open("data/test_set/"+str(i+45000)+".jpg")
        im = ImageOps.grayscale(im)
#         im.show()

        # Size of the image in pixels (size of original image)
        # (This is not mandatory)
        width, height = im.size

        # Setting the points for cropped image
        left = 0
        top = 0
        right = width/3
        bottom = height
        # Cropped image of above dimension
        # (It will not change original image)
        im1 = im.crop((left, top, right, bottom))
        im1.save('im1'+str(i)+'.jpg')
        test_image = image.load_img('im1'+str(i)+'.jpg', target_size = (64, 64),color_mode = 'grayscale')
#         test_image.show()
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = -1)
        #PREDCITING ON FIRST PART
        result1 = cnn.predict(test_image)


        # Shows the image in image viewer



        # Setting the points for cropped image
        left = width/3
        top = 0
        right = 2*width/3
        bottom = height

        # Cropped image of above dimension
        im2 = im.crop((left, top, right, bottom))
        im2.save('im2'+str(i)+'.jpg')
        test_image = image.load_img('im2'+str(i)+'.jpg', target_size = (64, 64),color_mode = 'grayscale')
#         test_image.show()
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = -1)
        #PREDICTING ON SECOND PART
        result2 = cnn.predict(test_image)

        # Shows the image in image viewer


        # Setting the points for cropped image
        left = 2*width/3
        top = 0
        right = width
        bottom = height

        # Cropped image of above dimension
        im3 = im.crop((left, top, right, bottom))
        im3.save('im3'+str(i)+'.jpg')
        test_image = image.load_img('im3'+str(i)+'.jpg', target_size = (64, 64),color_mode = 'grayscale')
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = -1)
        #PREDICTING IN THIRD PART
        result3 = cnn.predict(test_image)
        
        # DECIDING WHAT IS OPERATOR AND WHAT IS OPERAND
        if result1[0][0]==1 or result1[0][1]==1 or result1[0][12]==1 or result1[0][13]==1:
            operator = result1
            op1 = get_digit(result2[0]) # GET INDICES CORRESPONDING TO DIGIT
            im2.save(str(op1-2)+'/im2'+str(i)+'.jpg')
            op2 = get_digit(result3[0]) # ACTUAL DIGIT WILL BE INDICE-
            im3.save(str(op2-2)+'/im3'+str(i)+'.jpg')
            # COMPUTE THE VALUE DEPENDING ON THE OPERATOR 
            if operator[0][0]==1:       
                im1.save('+/'+'im1'+str(i)+'.jpg')
            elif operator[0][1]==1:
                im1.save('-/'+'im1'+str(i)+'.jpg')
            elif operator[0][12]==1:
                im1.save('divide/'+'im1'+str(i)+'.jpg')
                if(float(op2)-2==0):value=0
                    #TO AVOID DIVISION BY ZERO ERROR
                else : value = (float(op1)-2)/(float(op2)-2)
            elif operator[0][13]==1:
                im1.save('multiply/'+'im1'+str(i)+'.jpg')
        elif result2[0][0]==1 or result2[0][1]==1 or result2[0][12]==1 or result2[0][13]==1:
            operator = result2
            op1 = get_digit(result1[0])
            op2 = get_digit(result3[0])
            im1.save(str(op1-2)+'/im1'+str(i)+'.jpg')
            op2 = get_digit(result3[0]) # ACTUAL DIGIT WILL BE INDICE-
            im3.save(str(op2-2)+'/im3'+str(i)+'.jpg')
            if operator[0][0]==1:
                im2.save('+/'+'im2'+str(i)+'.jpg')
            elif operator[0][1]==1:
                im2.save('-/'+'im2'+str(i)+'.jpg')
            elif operator[0][12]==1:
                im2.save('divide/'+'im2'+str(i)+'.jpg')
                if(float(op2)-2==0):value=0
                else : value = (float(op1)-2)/(float(op2)-2)
            elif operator[0][13]==1:
                im2.save('multiply/'+'im2'+str(i)+'.jpg')
        else :
            operator = result3
            op1 = get_digit(result1[0])
            op2 = get_digit(result2[0])
            im1.save(str(op1-2)+'/im1'+str(i)+'.jpg')
            op2 = get_digit(result3[0]) # ACTUAL DIGIT WILL BE INDICE-
            im2.save(str(op2-2)+'/im2'+str(i)+'.jpg')
            if operator[0][0]==1:
                im3.save('+/'+'im3'+str(i)+'.jpg')
            elif operator[0][1]==1:
                im3.save('-/'+'im3'+str(i)+'.jpg')
                value = float(op1)-2-(float(op2)-2)
            elif operator[0][12]==1:
                im3.save('divide/'+'im3'+str(i)+'.jpg')
                if(float(op2)-2==0):value=0
                else : value = (float(op1)-2)/(float(op2)-2)
            elif operator[0][13]==1:
                im3.save('multiply/'+'im3'+str(i)+'.jpg')



# SAVING THE MODEL
model_json = cnn.to_json()
with open("model_final_2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn.save_weights("model_final_2.h5")

