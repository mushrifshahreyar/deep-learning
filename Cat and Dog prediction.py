#!/usr/bin/env python
# coding: utf-8

# In[27]:


import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random as random

Categories = ['Cat','Dog']
Dir = 'Cats and Dog/PetImages/'
size = 50
training_data = []

def Create_training_data():
    for category in Categories:
        class_index = Categories.index(category)
        path = os.path.join(Dir,category)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_img = cv2.resize(img_arr,(size,size))
                training_data.append([new_img,class_index])
            except Exception as e:
                pass

Create_training_data()


# In[65]:


# print(training_data[24000])
random.shuffle(training_data)
# for samples in training_data[:10]:
#     print(samples[1])
x = []
y = []
for features, labels in training_data:
    x.append(features)
    y.append(labels)
    
# print(x[0])
x = np.array(x).reshape(-1,size,size,1)
print(x.shape[0])


# In[66]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
X = x/255.0

model = Sequential()

model.add(Conv2D(64,(3,3),input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss= 'binary_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])
model.fit(X,y,batch_size = 32,epochs = 10,validation_split = 0.1)


# In[84]:


imgg = cv2.imread('dog.jpeg',cv2.IMREAD_GRAYSCALE)
new_imgg = cv2.resize(imgg,(50,50))


# In[92]:


def pre_predict(filepath):
    size_img = 50
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_img_array = cv2.resize(img_array,(size_img,size_img))
    return np.array(new_img_array).reshape(-1,size_img,size_img,1)

prediction = model.predict([pre_predict('cat.jpg')])
print(Categories[int(prediction[0][0])])

