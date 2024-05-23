import cv2 as cv
import torch
import os
#from network import Net
#from cascade import create_cascade
from transforms import ValidationTransform
from PIL import Image

import gdown

# NOTE: This will be the live execution of your pipeline

def live(args):
    # TODO: 
    #   Load the model checkpoint from a previous training session (check code in train.py)
        ## TODO



    '''
    #   Initialize the face recognition cascade again (reuse code if possible)
    if os.path.exists(cv.data.haarcascades + "haarcascade_frontalface_default.xml"):
        HAAR_CASCADE = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    else:
        print("No HAAR_CASCADE-file found. Downloading it from Google Drive.")
        url = "PERSONAL_LINK_TO_GOOGLE_DRIVE"
        output = "haarcascade_frontalface_default.xml"
        gdown.download(url, output, fuzzy=True)
        HAAR_CASCADE = output

    face_cascade = cv.CascadeClassifier(HAAR_CASCADE)
    '''

    #   Also, create a video capture device to retrieve live footage from the webcam.
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    #   Attach border to the whole video frame for later cropping.
    #   Run the cascade on each image, crop all faces with border.
    #   Run each cropped face through the network to get a class prediction.
    #   Retrieve the predicted persons name from the checkpoint and display it in the image

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        h, w = frame.shape[:2]
        top_bottom_border = int((h * float(args.border)) / 2)
        left_right_border = int((w * float(args.border)) / 2)
        
        frame_border = cv.copyMakeBorder(
                            frame, top_bottom_border, top_bottom_border, left_right_border, left_right_border, cv.BORDER_REFLECT
                        )



        '''
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        frame_rectangle = frame.copy()

        for (x,y,w,h) in faces:
            frame_rectangle = cv.rectangle(frame_rectangle,(x,y),(x+w,y+h),(255,0,0),2)
        '''
        cv.imshow('webcam',frame_border)
        if cv.waitKey(1) == ord('q'):
            break




    if args.border is None:
        print("Live mode requires a border value to be set")
        exit()

    args.border = float(args.border)
    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()