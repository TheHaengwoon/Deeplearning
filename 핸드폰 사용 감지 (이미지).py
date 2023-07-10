#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# In[2]:


# 이미 학습된 VGG-16 모델 weights 장착
vgg16_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 새로운 모델 제작
model = Sequential()

# 새로운 모델에 사용하고자하는 VGG-16모델 제작 
for layer in vgg16_model.layers:
    model.add(layer)

# 이미 학습된 부분의 층들의 weights 사용 중지 
for layer in model.layers:
    layer.trainable = False

# CNN의 층 개수 설정 
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))  # Two classes: Right hand texting and Left hand texting


# In[3]:


# 입력된 이미지들 전처리하기 
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img


# In[4]:


labels = ['Right Hand Texting', 'Left Hand Texting']

# 입력된 이미지로 운전자 핸드폰 사용 감지
def predict_driver_distraction(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_label = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    print('Predicted Label:', predicted_label)
    print('Confidence:', confidence, '%')


# In[5]:


image_path = 'texting.jpeg'
predict_driver_distraction(image_path)

