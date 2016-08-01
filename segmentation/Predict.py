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

def predict():

	model = get_model()
	model.load("models-10000")
	
	X_test = np.load("X_test.npy")
	X_test = X_test.astype(np.float32)
	print X_test.shape
	# for k in range(0,5,20):
		# y_pred[k:k+5,:,:,:] = model.predict(X_test[k:k+5,:,:,:])
	y_pred = model.predict(X_test)
	y_pred = np.array(y_pred).astype(np.float32)
	y_pred = y_pred[:,:,:,0]
	y_pred = np.reshape(y_pred, (-1, 256, 256))
	print y_pred.shape
	skimage.io.imsave("y_pred.tif", y_pred)

if __name__ == '__main__':
	predict()
