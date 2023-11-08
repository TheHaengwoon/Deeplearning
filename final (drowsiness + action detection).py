#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image

# DNN's Caffe model for face detection
prototxt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# dlib's facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# EAR threshold for eye blink detection
EAR_THRESHOLD = 0.2

# Load VGG16 model for action recognition
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

cap = cv2.VideoCapture(0)

action_labels = ['Safe driving', 'Texting(Right)', 'Calling(Right)', 'Texting(Left)', 'Calling(Left)', 'Radio Control', 'Drinking/Eating', 'Looking Back', 'Hair & makeup', 'Conversation with passenger']

while True:
    ret, frame = cap.read()

    if ret:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x_end, y_end) = box.astype("int")

                roi = frame[y:y_end, x:x_end]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                shape = predictor(gray, dlib.rectangle(0, 0, x_end - x, y_end - y))
                shape = np.array([[p.x + x, p.y + y] for p in shape.parts()])

                left_eye = shape[36:42]
                right_eye = shape[42:48]

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)

                ear = (left_ear + right_ear) / 2.0

                if ear < EAR_THRESHOLD:
                    cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 0, 255), 2)

        # VGG-16 model input size adjustment
        resized_frame = cv2.resize(frame, (224, 224))

        preprocessed_frame = image.img_to_array(resized_frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)

        # Action recognition
        prediction = model.predict(preprocessed_frame)
        predicted_label = action_labels[np.argmax(prediction)]

        # Display the action label
        cv2.putText(frame, f'Action: {predicted_label}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Driver Action Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

