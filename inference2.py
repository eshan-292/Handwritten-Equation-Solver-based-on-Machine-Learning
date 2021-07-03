#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#IMPORTING THE LIBRARIES
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
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

from keras.models import model_from_json

#lOADING THE SAVED MODEL
json_file = open('model_final_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn = model_from_json(loaded_model_json)
# load weights into new model
cnn.load_weights("model_final_2.h5")


# Python code to demonstrate the use of 'sys' module
# for command line arguments
  
import sys
  
# command line arguments are stored in the form
# of list in sys.argv
argumentList = sys.argv
#print argumentList
  
# Print the name of file
#print sys.argv[0]
  
# Print the first argument after the name of file
testfolder_path=sys.argv[1]

# FUNCTION TO SPLIT THE IMAGES OF TEST SET AND THEN PREDICT OUTPUT ON EACH INDIVIDUAL IMAGE AND COMBINE TO GENERATE FINAL VALUE
from PIL import Image,ImageOps
import cv2
import numpy as np
import os

correct=0
results =[]
img_names=[]
# GET DIGIT FUNCTION TO GET INDICE CORRESPONDING TO THE DIGIT
def get_digit(arr):
    for i in range(0,len(arr)):
        if arr[i]==1:
            return i
    return 0

# TEST SIZE 
test_size = 300
n = test_size #should be less than 5000
# FOR EVERY IMAGE IN TEST_SET 
#for i in range(1,n+1):
for filename in os.listdir(testfolder_path):
        img_names.append(filename)
        im = cv2.imread(os.path.join(folder,filename))

 
    # Opens a image in RGB mode
        #im = Image.open(testfolder_path+"/"+str(i+45000)+".jpg")
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
        
        
        
        #PREDICTING ON THIRD PART
        result3 = cnn.predict(test_image)
        
        # DECIDING WHICH IS OPERATOR AND WHICH IS OPERAND
        if result1[0][0]==1 or result1[0][1]==1 or result1[0][12]==1 or result1[0][13]==1:
            operator = result1
            op1 = get_digit(result2[0]) # GET INDICES CORRESPONDING TO DIGIT
            op2 = get_digit(result3[0]) # ACTUAL DIGIT WILL BE INDICE-
            # COMPUTE THE VALUE DEPENDING ON THE OPERATOR 
            if operator[0][0]==1:       
                value=float(op1)-2+float(op2)-2
            elif operator[0][1]==1:
                value = float(op1)-2-(float(op2)-2)
            elif operator[0][12]==1:
                if(float(op2)-2==0):value=0
                    #TO AVOID DIVISION BY ZERO ERROR
                else : value = (float(op1)-2)/(float(op2)-2)
            elif operator[0][13]==1:
                value = (float(op1)-2)*(float(op2)-2)
        elif result2[0][0]==1 or result2[0][1]==1 or result2[0][12]==1 or result2[0][13]==1:
            operator = result2
            op1 = get_digit(result1[0])
            op2 = get_digit(result3[0])
            if operator[0][0]==1:
                value=float(op1)-2+float(op2)-2
            elif operator[0][1]==1:
                value = float(op1)-2-(float(op2)-2)
            elif operator[0][12]==1:
                if(float(op2)-2==0):value=0
                else : value = (float(op1)-2)/(float(op2)-2)
            elif operator[0][13]==1:
                value = (float(op1)-2)*(float(op2)-2)
        else :
            operator = result3
            op1 = get_digit(result1[0])
            op2 = get_digit(result2[0])
            if operator[0][0]==1:
                value=float(op1)-2+float(op2)-2
            elif operator[0][1]==1:
                value = float(op1)-2-(float(op2)-2)
            elif operator[0][12]==1:
                if(float(op2)-2==0):value=0
                else : value = (float(op1)-2)/(float(op2)-2)
            elif operator[0][13]==1:
                value = (float(op1)-2)*(float(op2)-2)
        
        # adding the value to list of results 
        results.append(value)
        # counting the no of correct predictions 
        #if dataset[45000+i-1]==value:
         #   correct+=1
# combining the results(predicted value) with the actual value in 2d array 
array_2d= np.column_stack((img_names,results))

#import matplotlib.pyplot as plt
# plot y=x (orange) with the points corresponing to prediction
#plt.scatter(array_2d[:,0],array_2d[:,1])
#plt.scatter(array_2d[:,1],array_2d[:,1])

# converting 2d array to dataframe and then converting dataframe to csv
import pandas as pd
(pd.DataFrame(array_2d)).to_csv("Dreams of Reality_2.csv")

