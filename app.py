import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model("attendance_model.h5")

# Define categories (the names of the classes in your dataset)
categories = ['shrejal', 'aastha', 'shristina', 'sresta']  # Update this list with your class names

# Initialize face detector (Haar Cascade Classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face from the frame
        face = frame[y:y+h, x:x+w]

        # Resize the face to match the input size of the model (128x128)
        face_resized = cv2.resize(face, (128, 128))

        # Normalize the image and expand dimensions for the model
        face_normalized = face_resized / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)  # Add batch dimension

        # Get model predictions
        predictions = model.predict(face_expanded)
        max_index = np.argmax(predictions[0])  # Get the class with highest probability

        # Get the name of the predicted class
        predicted_name = categories[max_index]

        # Display the predicted name on the frame
        cv2.putText(frame, predicted_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
