import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import pandas as pd
import datetime

# Load your trained model
model = load_model("face_recognition_model.h5")  # Replace with your model's file path

# Class mapping (as defined in your training process)
Result_class = {0: 'aastha', 1: 'shrejal', 2: 'shristina', 3: 'sresta'}
# Attendance log
attendance_log = "attendance.csv"

# Preprocessing function
def preprocess_image(img, target_size):
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize if necessary
    img = np.expand_dims(img, axis=0)
    return img


def mark_attendance(name):
    try:
        df = pd.read_csv(attendance_log)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Name", "Time"])

    # Check if already marked for today
    if not ((df["Name"] == name) & (df["Time"].str.contains(datetime.datetime.now().strftime("%Y-%m-%d")))).any():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Create a new row as a DataFrame
        new_row = pd.DataFrame({"Name": [name], "Time": [timestamp]})
        # Use pd.concat to add the new row
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_log, index=False)
        print(f"Attendance marked for {name} at {timestamp}")


# Initialize webcam
camera = cv2.VideoCapture(0)
target_size = (100, 100)  # Adjust based on your model's input size

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Detect and preprocess the face (simplistic approach: using entire frame)
    face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preprocessed_face = preprocess_image(face, target_size)

    # Predict the class
    result = model.predict(preprocessed_face, verbose=0)
    predicted_class = Result_class[np.argmax(result)]

    # Mark attendance if a known face is recognized
    if predicted_class != "unknown":  # Adjust based on your model
        mark_attendance(predicted_class)

    # Display the frame
    cv2.putText(frame, f"Detected: {predicted_class}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

camera.release()
cv2.destroyAllWindows()
