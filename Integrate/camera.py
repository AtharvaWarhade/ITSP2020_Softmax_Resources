import cv2
import time
import numpy as np
import keras
from keras.models import load_model
import os

#load your model
model = load_model(r"unique2.h5")

#1. Create an object. Zero for external camera
cap = cv2.VideoCapture(r"test.mp4")

##2. Choose the dimensions for your frames
width = 256
height = 256
dim = (width, height)

#Variable a will calculate the number of frames
a = 0

#Loop for continuous video feed
while cap.isOpened():
    a = a+1 ##increase frame count

    #3. Create a frame object
    check, frame = cap.read()
    #6. Convreting to grascale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the image to the desired dataset value
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    #print(resized.shape)
    #resized = cv2.flip(resized, 1)
    #convert the image into a numpy array and reshape it to the required format
    array = np.array(resized)
    array = array.reshape((1, 256, 256, 1))
    #print(array.shape)

    #get your model to predict the given image and round it off
    X = np.round(model.predict(array))
    if (X == [1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0]).all():
        print("SAFE")
        cv2.putText(frame,"Safe",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4)
    elif (X == [0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0]).all() or (X == [0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0]).all():
        print("Driver is texting!")
        cv2.putText(frame,"Driver is texting!",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
    elif (X == [0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0]).all() or (X == [0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0]).all():
        print("Driver is talking on the phone!")
        cv2.putText(frame,"Driver is talking on the phone!",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
    elif (X == [0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0]).all():
        print("Driver is operating the radio!")
        cv2.putText(frame,"Driver is operating the radio!",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
    elif (X == [0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0]).all():
        print("Driver is drinking!")
        cv2.putText(frame,"Driver is drinking!",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
    elif (X == [0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0]).all():
        print("Driver is reaching behind!")
        cv2.putText(frame,"Driver is reaching behind!",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
    elif (X == [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0]).all():
        print("Driver is putting on Makeup!")
        cv2.putText(frame,"Driver is putting on Makeup!",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
    elif (X == [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1]).all():
        print("Driver is talking to the passenger!")
        cv2.putText(frame,"Driver is talking to the passenger!",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
    #4. Show the frame
    cv2.imshow("Capturing", frame)

    #5. For press any key to out (milliseconds)
    #cv2.waitKey(0)

    #7. For playing
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
print(a)
#2. Shutdown the camera
cap.release()

cv2.destroyAllWindows()
