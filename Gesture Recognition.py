#Importing required libraries

import os
import cv2
import imutils
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split # Helps with organizing data for training
from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

#Declaring global variables

bg = None   #This is the variable for the first frame.

#-------------------------------------------------------
# This function will compute the running average between
# the background model and the current frame.
#-------------------------------------------------------

def run_avg(image, aWeight):
    global bg
    # Here, we need to take into account the first frame.
    if bg is None:
        bg = image.copy().astype("float")
        return

    # We use this OpenCV function to compute the running
    # average of the background models against the
    # current frame.
    cv2.accumulateWeighted(image, bg, aWeight)

#-------------------------------------------------------
# This function will segment the hands in the current
# image.
#-------------------------------------------------------

def segment(image, threshold=25):
    global bg
    # This finds the difference between the background
    # and the current frame. 
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # This uses the threshold so we can extract the
    # foreground from the image.
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # This finds any available contours in the current
    # image.
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # This returns none if no contours detected, or
    # returns results if they are.
    if len(cnts) == 0:
        return
    else:
        # This picks the object with the most contours,
        #which should be the hand.
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#-------------------------------------------------------
# This function is used more for debugging and showing
# results later. It plots the image into the notebook.
#-------------------------------------------------------

def plot_image(path):
  img = cv2.imread(path) # Reads the image into a numpy.array
  img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the correct colorspace (RGB)
  print(img_cvt.shape) # Prints the shape of the image just to check
  plt.grid(False) # Without grid so we can see better
  plt.imshow(img_cvt) # Shows the image
  plt.xlabel("Width")
  plt.ylabel("Height")
  plt.title("Image " + path)

#-------------------------------------------------------
# This is the main function that will use the previously
# defined functions to capture image.
#-------------------------------------------------------
if __name__ == "__main__":
    
    # This is an arbitrary value to begin the running
    # average and should change fairly quickly.
    aWeight = 0.5

    # This finds the webcam.
    camera = cv2.VideoCapture(0)

    # This sets a Region of Interest (ROI).
    top, right, bottom, left = 10, 350, 225, 590

    # This simply initializes the number of frames.
    num_frames = 0

    #Loads the model we built in our training script.
    model = keras.models.load_model(r"C:\Users\frank\Desktop\handrecognition_model.h5")

    # This loop will run until it is interrupted.
    while(True):
        # This gets the current frame or image.
        (grabbed, frame) = camera.read()

        kernel = np.ones((3,3),np.uint8)

        # This resizes the frame for our purposes.
        frame = imutils.resize(frame, width=700)

        # This flips the frame. 
        frame = cv2.flip(frame, 1)

        # This makes a copy of the frame. 
        clone = frame.copy()

        # This gets the height and width of the frame.
        (height, width) = frame.shape[:2]

        # This gets the new ROI.
        roi = frame[top:bottom, right:left]

        # This segment will convert the RoI to grayscale and blur it.
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        #gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)

        #extract skin color image
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)

        #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100)
        mask = cv2.resize(mask,(128,128))
        img_array = np.array(mask)

        # This calibrates our running average model until a
        # threshold is reached.
        if num_frames < 30:
            run_avg(hsv, aWeight)
        else:
            # This segments the hand region.
            hand = segment(hsv)

            # This checks whether hand region is segmented
            if hand is not None:
                # If so, unpack the thresholded image and
                # segmented region.
                (thresholded, segmented) = hand

                # This draws the segmented region and displays the frame.
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

        # This draws the segmented hand.
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # This increments the number of frames.
        num_frames += 1

        # This displays the frame with segmented hand.
        frame_analysis = cv2.imshow("Video Feed", clone)

        # Changing dimension from 128x128 to 128x128x3
        img_array = np.stack((img_array,)*3, axis=-1)
        
        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        img_array_ex = np.expand_dims(img_array, axis=0)

        #This analyzes the frame to interpret the gesture.
        try:
            prediction = model.predict(img_array_ex)
        except:
            prediction = "Nothing detected!"

        #This outputs the interpreted gesture if captured
        print(prediction)

        # This will capture any key pressed by the user.
        keypress = cv2.waitKey(1) & 0xFF

        # If the user pressed "q", then stop looping!
        if keypress == ord("q"):
            break

# This section frees up memory after the script is halted.
camera.release()
cv2.destroyAllWindows()
