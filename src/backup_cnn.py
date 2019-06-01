from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from skimage.transform import resize
import numpy as np

BATCH_SIZE = 128
CLASSES = 3
EPOCHS = 12
NR_STEPS = 50 #Steps per epoch
VALIDATION_STEPS = 50
IMG_H = IMG_W = 38
CHANNELS = 3
TRAIN_X_FILE = '../../data/train_letters.npy' 
TRAIN_Y_FILE = '../../data/train_labels.npy'
TEST_X_FILE = '../../data/test_letters.npy' 
TEST_Y_FILE = '../../data/test_labels.npy'
MODEL_SAVE = '../../backup_model.model'

def build_net(input_shape):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape= input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(CLASSES, activation='softmax'))
	return model

if __name__ == '__main__':
	input_shape = (IMG_H, IMG_W, CHANNELS)
	model = build_net(input_shape)

	trainX = np.load(TRAIN_X_FILE, allow_pickle=True)
	trainY = np.load(TRAIN_Y_FILE, allow_pickle=True)
	testX = np.load(TEST_X_FILE, allow_pickle=True)
	testY = np.load(TEST_Y_FILE, allow_pickle=True)

	trainX = [resize(image, (IMG_H,IMG_W)) for image in trainX]
	testX = [resize(image, (IMG_H,IMG_W))  for image in testX]
	trainX = np.array(trainX)
	testX = np.array(testX)

	print(np.shape(trainX), type(trainX))
	print(np.shape(trainY), type(trainY))

	# train_generator = ImageDataGenerator(
	# 	preprocessing_function=preprocess_input)

	# history = model.fit_generator(train_generator.flow(trainX, trainY, batch_size=BATCH_SIZE), 
	# 	epochs=EPOCHS, 
	# 	steps_per_epoch = NR_STEPS,
	# 	validation_data = (testX, testY), 
	# 	validation_steps = VALIDATION_STEPS)

	model.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=keras.optimizers.Adadelta(),
		metrics=['accuracy'])

	model.fit(trainX, trainY,
		batch_size=BATCH_SIZE,
		epochs=EPOCHS, 
		verbose=1,
		validation_data=(testX,testY))
	score = model.evaluate(testX, testY, verbose=0)
	print('Test loss: ', score[0])
	print('Test accuracy: ', score[1])
	model.save()