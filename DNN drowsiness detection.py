import cv2
import numpy as np
import dlib
import time
from scipy.spatial import distance as dist

#DNN의 cafffe model 사용
prototxt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

#dlib의 facial landmark으로 얼굴에 좌표찍기
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#눈의종화비(EAR) 함수
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

#EAR임계값 설정
EAR_THRESHOLD = 0.2

#frame안의 양쪽눈의 EAR<임계값이면 빨간 박스로 표시
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
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
            
            shape = predictor(gray, dlib.rectangle(0, 0, x_end-x, y_end-y))
            shape = np.array([[p.x + x, p.y + y] for p in shape.parts()])
            
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            ear = (left_ear + right_ear) / 2.0
            
            if ear < EAR_THRESHOLD:
                cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 0, 255), 2)
