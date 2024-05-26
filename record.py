import cv2 as cv
import os
import gdown
import uuid
import csv
from common import ROOT_FOLDER

# from cascade import create_cascade

# Quellen
#  - How to open the webcam: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
#  - How to run the detector: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
#  - How to download files from google drive: https://github.com/wkentaro/gdown
#  - How to save an image with OpenCV: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
#  - How to read/write CSV files: https://docs.python.org/3/library/csv.html
#  - How to create new folders: https://www.geeksforgeeks.org/python-os-mkdir-method/

# This is the data recording pipeline
def record(args):
    # TODO: Implement the recording stage of your pipeline
    #   Create missing folders before you store data in them (os.mkdir)
    if not os.path.exists(ROOT_FOLDER):
        os.makedirs(ROOT_FOLDER)

    foldername = (args.folder).replace(" ", "")
    folder_path = os.path.join(ROOT_FOLDER, foldername)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    #   Open The OpenCV VideoCapture Device to retrieve live images from your webcam (cv.VideoCapture)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    #   Initialize the Haar feature cascade for face recognition from OpenCV (cv.CascadeClassifier)
    # face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    #   If the cascade file (haarcascade_frontalface_default.xml) is missing, download it from google drive
    if os.path.exists(cv.data.haarcascades + "haarcascade_frontalface_default.xml"):
        HAAR_CASCADE = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    else:
        print("No HAAR_CASCADE-file found. Downloading it from Google Drive.")
        url = "PERSONAL_LINK_TO_GOOGLE_DRIVE"
        output = "haarcascade_frontalface_default.xml"
        gdown.download(url, output, fuzzy=True)
        HAAR_CASCADE = output

    face_cascade = cv.CascadeClassifier(HAAR_CASCADE)

    frame_counter = 0
    saving_blocker = False

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        #   Run the cascade on every image to detect possible faces (CascadeClassifier::detectMultiScale)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        frame_rectangle = frame.copy()

        for (x, y, w, h) in faces:
            frame_rectangle = cv.rectangle(
                frame_rectangle, (x, y), (x + w, y + h), (255, 0, 0), 2
            )

        cv.imshow("webcam", frame_rectangle)
        if cv.waitKey(1) == ord("q"):
            break

        #   If there is exactly one face, write the image and the face position to disk in two seperate files (cv.imwrite, csv.writer)
        #   If you have just saved, block saving for 30 consecutive frames to make sure you get good variance of images.
        if len(faces) == 1 and not saving_blocker:
            filename = str(uuid.uuid4())
            file_path = os.path.join(folder_path, filename)

            cv.imwrite(file_path + ".jpg", frame)

            csvfile = open(file_path + ".csv", "w", newline="", encoding="utf-8")
            c = csv.writer(csvfile)
            for (x, y, w, h) in faces:
                c.writerow([x, y, w, h])
            csvfile.close()
            saving_blocker = True

        else:
            frame_counter += 1
            if frame_counter >= 30:
                frame_counter = 0
                saving_blocker = False

    if args.folder is None:
        print("Please specify folder for data to be recorded into")
        exit()
