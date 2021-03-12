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

images = []
labels = []
name = []

#read images
def read_images_labels(path, i):
    for file in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file))
        if os.path.isdir(abs_path):
            i += 1
            temp = os.path.split(abs_path)[-1]
            name.append(temp)
            read_images_labels(abs_path,i)
            amount = int(len(os.listdir(path)))
            sys.stdout.write('\r'+'>'*(i)+' '*(amount-i)+'[%s%%]'%(i*100/amount)+temp)
        else:
            if file.endswith('.jpg'):
                image = cv2.resize(cv2.imread(abs_path),(64,64))
                images.append(image)
                labels.append(i-1)
    return images, labels, name

def read_main(path):
    images, labels, name = read_images_labels(path, i=0)
    images = np.array(images, dtype=np.float32)/255 #normalization
    labels = np_utils.to_categorical(labels, num_classes=20)
    np.savetxt('name.txt', name, delimiter = ' ', fmt="%s")
    return images, labels

images, labels = read_main('/home/lab1323/Desktop/ml_classifivation_homework2/train/characters-20')

# split train and valid
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)

model = Sequential()
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# from jpeg to data
datagen = ImageDataGenerator(zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

epochs = 100
batch_size = 512
file_name = str(epochs) + '_' + str(batch_size)

# training
start_time = time.time()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, validation_data=(X_test, y_test))
print("\n")
print("--- %s seconds ---" % (time.time() - start_time))
model.save('./my_model')
score = model.evaluate(X_test, y_test, verbose=0)
print("\n")
print(model.metrics_names)
print(score)