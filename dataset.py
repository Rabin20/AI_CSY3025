import cv2 #use the OpenCV library to import_data for_computer vision tasks.
import os #The os module should be imported for operating system-related features.

def capture_images_from_webcam():
    # Prompt the user to enter the person title
    title = input("Please enter the person title: ")
    location = "dataset_images/{}".format(title)
    os.makedirs(location, exist_ok=True)

    # Capture video from webcam
    video_capture = cv2.VideoCapture(0)

    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 1

    while True:
        # Read frame from video capture
        frame = video_capture.read()

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the captured frame
        cv2.imshow("Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('s') and len(faces) > 0:
            # Save the captured image with incremental filenames
            image_path = "{}/{}_{}.jpg".format(location, title, str(count).zfill(2))
            cv2.imwrite(image_path, frame)
            print("Image saved:", image_path)

            count += 1

        if cv2.waitKey(1) & 0xFF == ord('q') or count > 100:
            break

    # Release the video capture and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

def load_image_from_file():
    # Prompt the user to enter the image file path
    file_path = input("Enter the image file path: ")

    if os.path.isfile(file_path):
        # Prompt the user to enter the folder title
        title = input("Enter the folder title: ")
        location = "dataset_images/{}".format(title)
        os.makedirs(location, exist_ok=True)

        # Load the image
        image = cv2.imread(file_path)

        count = 1
        while count <= 100:
            # Save the loaded image with incremental filenames
            image_path = "{}/{}_{}.jpg".format(location, title , str(count).zfill(2))
            cv2.imwrite(image_path, image)
            print("Image saved:", image_path)
            count += 1

    else:
        print("File not found!")

def generate_label_file():
    # Create the label.txt file
    label_file = open("label.txt", "w")

    # Get the list of folder names in the 'dataset_images' directory
    folders = os.listdir("dataset_images")

    for i, folder in enumerate(folders):
        # Write the label in the format: <index> <folder name>
        label = "{} {}\n".format(i, folder)
        label_file.write(label)

    # Close the label.txt file
    label_file.close()
    print("label.txt generated successfully!")

def main():
    # Prompt the user to select an option
    option = input("Select one option you prefer to choose:\n1. Load image by capturing from Webcam\n2. Load image from image file location\n")

    if option == '1':
        capture_images_from_webcam()
    elif option == '2':
        load_image_from_file()
    else:
        print("Invalid option selected!")

    # Generate the label.txt file
    generate_label_file()

if __name__ == "__main__":
    main()
