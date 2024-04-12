import cv2 as cv
from common import ROOT_FOLDER, TRAIN_FOLDER, VAL_FOLDER
import os
import csv
import random

# Quellen
#  - How to iterate over all files/folders in one directory: https://www.tutorialspoint.com/python/os_walk.htm
#  - How to add border to an image: https://www.geeksforgeeks.org/python-opencv-cv2-copymakeborder-method/

# This is the cropping of images
def crop(args):
    # TODO: Crop the full-frame images into individual crops
    #   Create the TRAIN_FOLDER and VAL_FOLDER is they are missing (os.mkdir)
    #   Clean the folders from all previous files if there are any (os.walk)
    if os.path.exists(TRAIN_FOLDER):
        delete_folder_contents(TRAIN_FOLDER)
    else:
        os.makedirs(TRAIN_FOLDER)

    if os.path.exists(VAL_FOLDER):
        delete_folder_contents(VAL_FOLDER)
    else:
        os.makedirs(VAL_FOLDER)


    #   Iterate over all object folders and for each such folder over all full-frame images 
    #   Read the image (cv.imread) and the respective file with annotations you have saved earlier (e.g. CSV)
    #   Attach the right amount of border to your image (cv.copyMakeBorder)

    for folder in os.listdir(ROOT_FOLDER):
        folder_path = os.path.join(ROOT_FOLDER, folder)

        if os.path.isdir(folder_path):
            for image_file_name in os.listdir(folder_path):
                if image_file_name.lower().endswith('.jpg'):
                    image_path = os.path.join(folder_path, image_file_name)
                    csv_file_name = image_file_name[:-4] + '.csv'
                    csv_path = os.path.join(folder_path, csv_file_name)










    #   Crop the face with border added and save it to either the TRAIN_FOLDER or VAL_FOLDER
    #   You can use 
    #
    #       random.uniform(0.0, 1.0) < float(args.split) 
    #
    #   to decide how to split them.
    if args.border is None:
        print("Cropping mode requires a border value to be set")
        exit()

    args.border = float(args.border)
    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()

def delete_folder_contents(folder):
    
    # Go through every directory and folder and delete all files
    for root, dirs, files in os.walk(folder):
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
    
    # Delete every folder
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)