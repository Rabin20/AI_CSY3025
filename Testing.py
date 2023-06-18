import cv2  # Import the OpenCV library for computer vision tasks
import tensorflow as tf  # Import the TensorFlow library for deep learning tasks
from tensorflow import keras  # Import the Keras module for building and training neural networks
import numpy as np  # Import the NumPy library for numerical operations
import datetime  # Import the datetime module for timestamping

attendance_log = "attendance.txt"  # Define the path to the attendance log file

def get_class_name(class_no):
    # Function to map class numbers to class names
    if class_no == 0:
        return "Dhanbir"
    elif class_no == 1:
        return "Hridaya"
    elif class_no == 2:
        return "Nabin"
    elif class_no == 3:
        return "Rabin"
    elif class_no == 4:
        return "Umang"
    elif class_no == 5:
        return "Unacha"

model = keras.models.load_model('facedetection.h5')  # Load the trained face detection model

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Load the face cascade classifier

cap = cv2.VideoCapture(0)  # Open the video capture device (webcam)
cap.set(3, 640)  # Set the width of the captured video
cap.set(4, 480)  # Set the height of the captured video
font = cv2.FONT_HERSHEY_COMPLEX  # Define the font for displaying text on the image

while True:
    success, img_original = cap.read()  # Read a frame from the video capture device
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces in the grayscale frame

    for (x, y, w, h) in faces:
        cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle around the detected face
        crop_img = img_original[y:y + h, x:x + w]  # Crop the face region from the frame
        img = cv2.resize(crop_img, (224, 224))  # Resize the cropped image to the input size of the model
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add an extra dimension to represent the batch size
        prediction = model.predict(img)  # Perform face recognition on the cropped image
        class_index = np.argmax(prediction)  # Get the index of the predicted class
        class_name = get_class_name(class_index)  # Get the class name based on the index

        cv2.putText(img_original, class_name, (x, y + h + 20), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)  # Display the class name above the detected face

        with open(attendance_log, "a") as file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current timestamp
            attendance_entry = f"{timestamp} - {class_name}\n"  # Create an attendance entry
            file.write(attendance_entry)  # Write the attendance entry to the log file

        cv2.putText(img_original, f"{class_name} - Attendance Registered", (x, y - 10), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)  # Display the attendance registration message

    cv2.imshow("Result", img_original)  # Display the result image with faces and labels
    cv2.waitKey(1)  # Wait for a key press (1 millisecond delay)

    if len(faces) > 0:
        cv2.waitKey(3000)  # Wait for 3 seconds if faces are detected
        break

cap.release()  # Release the video capture device
cv2.destroyAllWindows()  # Close all windows
