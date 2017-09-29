# build deep net for steering angle. 

import csv 
import cv2
import numpy  as np 

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		
images = []
measurements = []
for line in lines:
	for i in range(3):
		source_ppath = line[i]
		filename = source_path.split('/')[-1]
		current_path = './data/IMG/' + filename
		
		