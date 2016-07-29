from TFlearn import *

########################################################
# Real-time data preprocessing
Preprocessing = ImagePreprocessing()
Preprocessing.add_featurewise_zero_center()
Preprocessing.add_featurewise_stdnorm()

# Real-time data augmentation
Augmentation  = ImageAugmentation()
Augmentation.add_random_flip_leftright()
Augmentation.add_random_flip_updown()
Augmentation.add_random_blur()

def get_cnn():
	net = input_data(shape=[None, 32, 32, 1],
	      data_preprocessing=Preprocessing,
	      data_augmentation=Augmentation)
	net  = conv_2d(net, 48, 5, activation='relu')
	net  = max_pool_2d(net, 2, strides=2)
	net  = conv_2d(net, 96, 5, activation='relu')
	net  = max_pool_2d(net, 2, strides=2)

	net  = fully_connected(net,  512, activation='relu')
	net  = fully_connected(net,  256, activation='relu')
	net  = fully_connected(net, 6, activation='softmax')

	return net

########################################################
def get_model():
	"""
	Define the architecture of the net is here
	"""
	net = get_cnn()

	net = regression(net, 
					 optimizer='adam', 
					 learning_rate=0.001,
	                 loss='categorical_crossentropy') 
	# Training the network
	model = DNN(net, 
				checkpoint_path='models',
				tensorboard_verbose=3)
	return model

