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
		
# image mask function to cut off corners and distractions 
def image_mask(image):
    # make zeros
    mask = np.zeros(image.shape, dtype=np.uint8)
    # define the corners
    #cut top left and right corners, top and bottom
    corners = np.array([[(0, 80),(0, 25), (110,0), (210, 0), (320,25), (320, 80)]], dtype=np.int32)
    # get channel number
    num_channels = image.shape[2] 
    # define no color
    mask_color = (255,)*num_channels
    # get mask
    mask = cv2.fillPoly(mask, corners, mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    # Use the mask
    masked_image = cv2.bitwise_and(image, mask)
    # get correct color scheme !! for plotting and saving only 
    #out_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    return masked_image
	
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
		image_c = image[65:140,0:320,:]
		masked_image = image_mask(image_c)
		images.append(masked_image)
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
# now split into train and valid 
# sklearn data prep functions 
#from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

X_train1, X_valid, y_train1, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle = True, random_state=4998)



#====================================================================
#delete unneeded objects
del X_trim
del y_trim
del images
del augmented_images
del measurements 
del augmented_measurements
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


# make data generator for augmented training set 
from keras.preprocessing.image import ImageDataGenerator
# source[https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html]



# training generator
datagen_train = ImageDataGenerator(
        featurewise_std_normalization = False,
        samplewise_std_normalization = False, 
        rotation_range = 5,
        width_shift_range=0.05,
        channel_shift_range = 50,
        #zca_whitening = True, 
        #zca_epsilon = 1e-6,
        #height_shift_range=0.1,
        #rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        fill_mode='constant')
        
# validation generator      
datagen_valid = ImageDataGenerator(
        featurewise_std_normalization = False,
        samplewise_std_normalization = False, 
        rotation_range=5,
        width_shift_range=0.05,
        channel_shift_range = 50,
        #zca_whitening = True, 
        #zca_epsilon = 1e-6,
        #height_shift_range=0.1,
        #rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        fill_mode='constant')

# fit to image groups        
datagen_train.fit(X_train1)
datagen_valid.fit(X_valid)

# set up the flow method !! checks the method. 
X_train_b, y_train_b = next(datagen_train.flow(X_train1, y_train1, batch_size=1))
X_valid_b, y_valid_b = next(datagen_valid.flow(X_valid, y_valid, batch_size=1))
cv2.imwrite("test1.jpg", X_train_b[0])


# set up the flow method !! sets up for fit generator
train_generator = datagen_train.flow(X_train1, y_train1, batch_size=32)
valid_generator = datagen_valid.flow(X_valid, y_valid, batch_size=32)


#activation = "relu"
#model.add(Cropping2D(cropping=((70,25), (0,0))))
#kernel_regularizer = l2(.0001)

# begin model
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (75,320,3)))
model.add(Conv2D(24, (5, 5), strides = (2,2), activation = "relu"))
model.add(Conv2D(36, (5, 5), strides = (2,2), activation = "relu"))
model.add(Conv2D(48, (3, 3), activation = "relu"))
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100, kernel_regularizer = l2(.0001)))
model.add(Dropout(0.2))
model.add(Dense(25, kernel_regularizer = l2(.0001), activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))
adam = Adam(lr = 0.00001)
model.compile(optimizer = adam, loss='mse')
# end model 
#====================================================================

#====================================================================
# Model training. 
# train with random validation split of 40%, shuffled. 
# trained for 15 epochs, though flexible. 
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 3)
#model.fit_generator(X_train, y_train, shuffle = True, epochs = 3)
#====================================================================

# run model with fit generator 
model.fit_generator(
        train_generator,
        #steps_per_epoch= 200,
        samples_per_epoch = 6000, 
        epochs=20,
        validation_data = valid_generator,
        validation_steps = 800)

model.save('model_generator_k2.h5')

#fit_generator(datagen, samples_per_epoch=len(X_train), epochs = 3)



#====================================================================
# save model and exit,. 
model.save('model_generator_k2.h5')
exit()
#====================================================================
