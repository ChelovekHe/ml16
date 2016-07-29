#!/usr/bin/env python



# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 10:12:52 2016

@author: tmquan
"""

from Model 		import *
from TFlearn	import *
from Utility	import *
######################################################################################
# Prepare the label
label = ('Homogeneous','Speckled','Nucleolar','Centromere', 'NuMem', 'Golgi')
index = np.arange(6) #(0, 1, 2, 3, 4, 5)
enums = dict(zip(label,index))
######################################################################################

def predict():

	model = get_model()
	model.load("models-600")
	
	image      = cv2.imread('data/test/Homogeneous/01001.jpg', cv2.IMREAD_GRAYSCALE)
	image      = cv2.imread('data/test/Speckled/04001.jpg', cv2.IMREAD_GRAYSCALE)
	image      = cv2.imread('data/test/Nucleolar/07001.jpg', cv2.IMREAD_GRAYSCALE)
	image      = cv2.imread('data/test/Centromere/09001.jpg', cv2.IMREAD_GRAYSCALE)
	image      = cv2.imread('data/test/NuMem/12001.jpg', cv2.IMREAD_GRAYSCALE)
	image      = cv2.imread('data/test/Golgi/13572.jpg', cv2.IMREAD_GRAYSCALE)
	image 	   = image.astype(np.float32)		# From uint8 to float32
	image 	   = np.expand_dims(image, axis=0) 	# From 32x32x3 to 1x32x32x3
	image 	   = np.expand_dims(image, axis=3) 	# From 32x32x3 to 1x32x32x3

	prediction = model.predict(image)
	prediction = np.squeeze(prediction)

	np.random.seed(2016)
	plt.barh(index, prediction, color=np.random.rand ( 256,3), align='center', alpha=0.4)

	plt.ylabel('Classes')
	plt.xlabel('Probability')
	plt.xlim([0,1])
	plt.title('Scores of the prediction')
	plt.yticks(index, label, fontsize=9)
	plt.show()

if __name__ == '__main__':
	predict()
