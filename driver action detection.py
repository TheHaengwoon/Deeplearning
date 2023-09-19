#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image


# In[ ]:


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[ ]:


model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[ ]:


cap = cv2.VideoCapture(0)


# In[ ]:


action_labels = ['Safe driving', 'Texting(Right)', 'Calling(Right)', 'Texting(Left)', 'Calling(Left)', 'Radio Control', 'Drinking/Eating', 'Looking Back', 'Hair & makeup', 'Conversation with passenger']
print(action_labels)


# In[ ]:


while True:
    ret, frame = cap.read()

    if ret:
       #vgg-16모델 인풋의 사이즈 맞춰주기 
        resized_frame = cv2.resize(frame, (224, 224))

        # 이미지 전처리 해주기 
        preprocessed_frame = image.img_to_array(resized_frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        #preprocessed_frame = VGG16.preprocess_input(preprocessed_frame)

        # 모델 이용하여 모션 감지 
        prediction = model.predict(preprocessed_frame)
        predicted_label = action_labels[np.argmax(prediction)]

        # 프레임에 특정 행동 알려주기 
        cv2.putText(frame, f'Action: {predicted_label}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        cv2.imshow('Driver Action Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


# In[ ]:





# In[ ]:




