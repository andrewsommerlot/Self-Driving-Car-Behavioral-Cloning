#====================================================================
# Project 3 in Udactiy Self Driving Car Nano Degree 
# Using a Convolutional Neural Network to Predict Steering Angles
# syntax upgraded to keras 2
#====================================================================

#====================================================================
# import libraries 
# for data read in and CNN
import csv 
import cv2
import numpy  as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D 	
from keras.regularizers import l2
from keras.optimizers import Adam 

# for balancing the data by steering angle
import matplotlib.pyplot as plt
import pylab
import random
#====================================================================

#====================================================================
# Read in training data 
# from Udacity class material, reads in y
lines = []
with open('/home/andrewrs/Desktop/udacity/data/train3/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
	
# from Udacity class material, reads in X
images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		# split character modified because image data was collected on windows 10
		filename = source_path.split('\\')[-1]
		current_path = '/home/andrewrs/Desktop/udacity/data/train3/IMG/' + filename
		image = cv2.imread(current_path)
		# add cropping
		image_c = image[60:160,0:320,:]
		images.append(image_c)
		measurement = float(line[3])
		measurements.append(measurement)

# from Udacity class material, doubles X by making a mirror image of each
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image, 1))
	augmented_measurements.append(measurement*-1.0)
#====================================================================
	
#====================================================================
# further processing by balancing steering angle values to get a more
#   uniform distribution

# down sample small steering angels
# Get training data for steering angles above and below 0.3. 
#   0.3 was chosen through trial an error to produce a more 
#    visually flat distribution. 

# absolute value greater than 0.3
y_trim_a = [y for X, y in zip(augmented_images, augmented_measurements) if np.abs(y) > .3]
X_trim_a = [X for X, y in zip(augmented_images, augmented_measurements) if np.abs(y) > .3]

# absolute value less than 0.3
y_trim_b = [y for X, y in zip(augmented_images, augmented_measurements) if np.abs(y) < .3]
X_trim_b = [X for X, y in zip(augmented_images, augmented_measurements) if np.abs(y) < .3]

# produce random selection 
# idea from https://stackoverflow.com/questions/19485641/python-random-sample-of-two-arrays-but-matching-indices
random_index = random.sample(range(len(y_trim_b)), 1000)

# down sample small steering angels
X_trim_add = [X_trim_b[i] for i in random_index]
y_trim_add = [y_trim_b[i] for i in random_index]

# combine them
y_trim_c = y_trim_a + y_trim_add
X_trim_c = X_trim_a + X_trim_add

# Upsample high steering angles
# Select those steering angles with absoute value 
#    greater than 0.4. 
y_trim_d = [y for X, y in zip(X_trim_c, y_trim_c) if np.abs(y) > .4]
X_trim_d = [X for X, y in zip(X_trim_c, y_trim_c) if np.abs(y) > .4]

# produce random selection
random_index = random.sample(range(len(y_trim_d)), 1000)

# combine them
y_trim = y_trim_c + [y_trim_d[i] for i in random_index]
X_trim = X_trim_c + [X_trim_d[i] for i in random_index]
#====================================================================

#====================================================================
# Used for visualizing distribution
# commented out. 
#plt.hist(y_trim)
#pylab.show()
#====================================================================

#====================================================================
# put training data in np arrays to feed into CNN
X_train = np.array(X_trim)
y_train = np.array(y_trim)
#====================================================================

#====================================================================
# Keras !!2.0!! CNN. 5 convolutional layers and 4 fully connected with dropout. 
# additionally, data is normalized and cropped in the initial layers. 
# multiple dropout and l2 regularizations added. Additionally, only one
#   layer in the fully connected section is given an activiation function. 
#  In practice, it was found no activations in fully connceted layers 
#  produced models which drove "twitchier" and were better at recognizing recovery. 
#  However, they were prone to over correction and crashes around difficult corners. 
#  Adding one activation function sandwiched between to fully connected linear layers 
#   provided a more ideal amount of response. 
# Trained with MSE as loss function and adam optimizer from Keras. 

# begin model
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (100,320,3)))
#model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, (5, 5), strides = (2,2), activation = "relu"))
model.add(Conv2D(36, (5, 5), strides = (2,2), activation = "relu"))
model.add(Conv2D(48, (5, 5), strides = (2,2), activation = "relu"))
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(Conv2D(64, (3, 3),  activation = "relu"))
model.add(Flatten())
model.add(Dense(100, kernel_regularizer = l2(.0001), activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(25, kernel_regularizer = l2(.0001), activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(10, kernel_regularizer = l2(.0001), activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(1))
adam = Adam(lr = 0.0001)
model.compile(optimizer = adam, loss='mse')
# end model 
#====================================================================

#====================================================================
# Model training. 
# train with random validation split of 20%, shuffled. 
# trained for 15 epochs, though flexible. 
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 30)
#====================================================================

#====================================================================
# save model and exit,. 
model.save('model_crop_k2_25.h5')
exit()
#====================================================================
