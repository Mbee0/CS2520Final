import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('facial_emotion_model.h5')

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to perform emotion recognition
def predict_emotion(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Crop the face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Resize and preprocess the face region
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.reshape((1, 48, 48, 3)) / 255.0
        
        # Perform prediction
        prediction = model.predict(face_roi)
        
        # Get the predicted emotion label
        emotion_label = ['angry', 'fear', 'happy', 'neutral', 'sad'][np.argmax(prediction)]
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the emotion label on the frame
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Perform emotion recognition
    frame_with_emotion = predict_emotion(frame)
    
    # Display the frame with emotion recognition
    cv2.imshow('Emotion Recognition', frame_with_emotion)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
