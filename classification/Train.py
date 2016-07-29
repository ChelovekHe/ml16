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

def train():
	X = np.load('X_train.npy')
	y = np.load('y_train.npy')
	
	
	X  = X.astype(np.float32)
	y  = to_categorical(y, 6)

	
	# Get the model from Model.py
	model = get_model()
	# Shuffle the data
	print('Shuffle data...')
	X, y = shuffle(X, y)


		
	model.fit(X, y, 
		run_id="convolutional_neural_network", 
		n_epoch=30, 
		validation_set=0.2,
		shuffle=True,
		show_metric=True,
      	snapshot_epoch=True,
      	batch_size=120)
if __name__ == '__main__':
	train()
