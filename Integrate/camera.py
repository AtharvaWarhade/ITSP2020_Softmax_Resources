import cv2
import time
import numpy as np
import keras
from keras.models import load_model
import os

#load your model
model = load_model("unique2.h5")

#1. Create an object. 0 for external camera, path for video
cap = cv2.VideoCapture("test.mp4")	#Video
#cap = cv2.VideoCapture(0)

##2. Choose the dimensions for your frames
width = 256
height = 256
dim = (width, height)

#Variable a will calculate the number of frames
a = 0

#Loop for continuous video feed
while True:
	a = a + 1							##increase frame count

	#3. Create a frame object
	check, frame = cap.read()			##check is the bool values
										##frame is the image
	#os.system("mpg123 " + file)
	#print(check)
	#print(frame.shape) #Representing image


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
		print('\n')
	if (X == [0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0]).all() or (X == [0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0]).all():
		print("Driver is texting!")
	elif (X == [0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0]).all() or (X == [0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0]).all():
		print("Driver is talking on the phone!")
	elif (X == [0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0]).all():
		print("Driver is operating the radio!")
	elif (X == [0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0]).all():
		print("Driver is drinking something!")
	elif (X == [0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0]).all():
		print("Driver is reaching behind!")
	elif (X == [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0]).all():
		print("Driver is putting on Makeup!")
	elif (X == [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1]).all():
		print("Driver is talking to the passenger!")
	#4. Show the frame
	cv2.imshow("Capturing", resized)

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
