#!/usr/bin/env python
# coding: utf-8

# In[64]:


import tensorflow as tf
tf.__version__

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train,axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))

model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])
model.fit(x_train,y_train,epochs = 5)


# In[57]:


val_loss,val_acc = model.evaluate(x_test,y_test)
print(val_loss,val_acc)


# In[58]:


model.save('digits_recognizer.model')


# In[59]:


predictions = model.predict([x_test])


# In[60]:


print(predictions)


# In[61]:


import numpy as np
print(np.argmax(predictions[1]))


# In[62]:


import matplotlib.pyplot as plt
plt.imshow(x_test[1],cmap = plt.cm.binary)
# print(x_test[0])


# In[ ]:




