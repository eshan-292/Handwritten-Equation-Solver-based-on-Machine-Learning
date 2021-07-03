#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#IMPORTING THE LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# IMPORTING LIBRARIES FOR CNN
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#K.set_image_dim_ordering('th')
K.set_image_data_format("channels_first")

from keras.models import model_from_json


#lOADING THE SAVED MODEL
json_file = open('model_final_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_final_1.h5")


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

#Testing

results = []
img_names=[]
test_size = 100
value = ""
correct = 0
n = test_size
#for i in range (1,n+1):
for filename in os.listdir(testfolder_path):
        img_names.append(filename) 
        #im = cv2.imread(os.path.join(folder,filename))
        test_image = image.load_img(os.path.join(folder,filename), target_size = (64, 64),color_mode = 'grayscale')
#        test_image.show()
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = -1)
        #PREDICTING ON FIRST PART
        result = cnn.predict(test_image)
        if result[0][0]==1:
            value = "infix"
        elif result[0][1]==1:
            value = "postfix"
        else:
            value = "prefix"
        #if value==dataset[45000+i-1]:
        #    correct+=1
        results.append(value)
    
dataset = pd.read_csv('annotations.csv').iloc[:,1].values
array_2d= np.column_stack((img_names,results))
import pandas as pd
(pd.DataFrame(array_2d)).to_csv("Dreams of Reality_1.csv")

