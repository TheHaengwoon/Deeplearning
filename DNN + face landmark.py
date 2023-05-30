import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Load the required deep learning face detector
prototxt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load the facial landmark predictor using dlib
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define a function to compute the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define the threshold for drowsiness detection
EAR_THRESHOLD = 0.25

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Start the main loop
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # Extract the height and width of the frame
    (h, w) = frame.shape[:2]
    
    # Construct a blob from the frame and perform face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    # Loop over the detected faces
    for i in range(0, detections.shape[2]):
        # Extract the confidence (probability) associated with the detection
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections by thresholding the confidence
        if confidence > 0.5:
            # Extract the bounding box coordinates of the detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x_end, y_end) = box.astype("int")
            
            # Extract the region of interest (ROI) containing the face
            roi = frame[y:y_end, x:x_end]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Detect the facial landmarks in the ROI using dlib
            shape = predictor(gray, dlib.rectangle(0, 0, x_end-x, y_end-y))
            shape = np.array([[p.x + x, p.y + y] for p in shape.parts()])
            
            # Extract the left and right eye landmarks
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            
            # Compute the EAR for the left and right eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            # Compute the average EAR for both eyes
            ear = (left_ear + right_ear) / 2.0
            
            # Check if the EAR is below the threshold
            if ear < EAR_THRESHOLD:
                # Draw a red rectangle around the face
                cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 0, 255), 2)
                # Display a message indicating drowsiness
