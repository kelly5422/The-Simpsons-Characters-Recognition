import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import time

from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras import utils as np_utils
import pandas as pd


dataframe = pd.read_csv("name.txt",header=None)
dataset=dataframe.values
loadtxt=dataset[:,0]

def read_images(path):
    images=[]
    for i in range(990):
        image = cv2.resize(cv2.imread(path+str(i+1)+'.jpg'), (64,64))
        images.append(image)
    images = np.array(images,dtype=np.float32)/255
    return images


test_images = read_images('test/test/')

def transform(name,label,lenSIZE):
    label_str = []
    for i in range (lenSIZE):
        temp = name[label[i]]
        label_str.append(temp)
    return label_str

# load model
reload_model = keras.models.load_model('./my_model')
predict = reload_model.predict_classes(test_images, verbose=1)
label_str = transform(loadtxt, predict ,test_images.shape[0])

raw_data={'id':range(1,991),
          'character':label_str    
}
df=pd.DataFrame(raw_data,columns=['id','character'])
df.to_csv('predict.csv',index=False,float_format='%.0f')


