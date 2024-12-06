import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import pandas as pd
import datetime
import pickle

# Load the class mapping from ResultMap.pkl
with open("ResultMap.pkl", "rb") as file:  # Replace with your file path
    Result_class = pickle.load(file)
print("Loaded Class Mapping:", Result_class)

# Load your trained model
model = load_model("face_recognition_model1.h5")  # Replace with your model's file path

# Attendance log file
attendance_log = "attendance2.csv"

# Preprocessing function
def preprocess_image(img, target_size):
    """
    Preprocess the image: resize, normalize, and expand dimensions.
    """
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)
    return img

# Function to mark attendance
def mark_attendance(name):
    """
    Log attendance for the detected name in a CSV file.
    """
    try:
        df = pd.read_csv(attendance_log)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Name", "Time"])

    # Check if attendance is already marked for today
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if not ((df["Name"] == name) & (df["Time"].str.contains(today_date))).any():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Append a new row
        new_row = pd.DataFrame({"Name": [name], "Time": [timestamp]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_log, index=False)
        print(f"Attendance marked for {name} at {timestamp}")
    else:
        print(f"Attendance already marked for {name} today.")

# Initialize webcam
camera = cv2.VideoCapture(0)
target_size = (100, 100)  # Adjust based on your model's input size

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press 'q' to exit the attendance system.")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face = frame[y:y+h, x:x+w]
        preprocessed_face = preprocess_image(face, target_size)

        # Predict the class
        result = model.predict(preprocessed_face, verbose=0)
        predicted_class = Result_class[np.argmax(result)]

        # Mark attendance for recognized faces
        if predicted_class != "unknown":  # Ensure only known faces are logged
            mark_attendance(predicted_class)
            cv2.putText(frame, f"Detected: {predicted_class}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the video frame
    cv2.imshow("Attendance System", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
