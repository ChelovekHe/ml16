#!/usr/bin/env python

"""
We will read through the data and create 2 file
X_train.npy		num x dimy x dimx x 3 (channel)
y_train.npy		num x 6

Where dimy and dimx are predefined dimy = 128, dimx = 128
if the images do not match that size, we will perform zero padding

6 is the number of classes:
homogeneous 		0
speckled			1
nucleolar			2
centromere			3
nuclear membrane 	4
golgi				5

"""

import pandas as pd # We will use dataframe in pandas to deal with csv
import numpy  as np # For multi-dimensional array
import cv2			# For reading image
import skimage.io  
import natsort 		# For natural sorting
import os
from Utility import *


# Prepare the label
label = ('Homogeneous','Speckled','Nucleolar','Centromere', 'NuMem', 'Golgi')
index = np.arange(6) #(0, 1, 2, 3, 4, 5)
enums = dict(zip(label,index))


trainDir =  "data/train/"
testDir  =  "data/test/"
##########################################################################
def processSubdirectory(dataDir):
	# Read the images and concatenate
	images = []
	labels = []
	for dirName, subdirList, fileList in os.walk(dataDir):
		# Sort the tif file numerically
		fileList = natsort.natsorted(fileList) 

		for f in fileList:
			filename = os.path.join(dirName, f)
			image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
			print filename
			dirname = dirName.split(os.path.sep)[-1]
			print dirname
			label =  enums[dirname]

			# Append to the images
			images.append(image)
			# Append to the labels
			labels.append(label)
	# Convert images list to numpy array
	X = np.array(images)
	y = np.array(labels)

	X = np.expand_dims(X, axis=3)
	# Get the current shape of images
	print X.shape
	print y.shape


	# 	np.save('X_train.npy', X_train)
	# 	np.save('y_train.npy', y_train)
	return X, y


if __name__ == '__main__':
	X_train, y_train = processSubdirectory(trainDir)
	X_test , y_test  = processSubdirectory(testDir)

	np.save('X_train.npy', X_train)
	np.save('y_train.npy', y_train)

	np.save('X_test.npy', X_test)
	np.save('y_test.npy', y_test)