from __future__ import print_function

import cv2
import keras
from PIL import Image
from cv2.cv2 import copyMakeBorder
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from skimage.transform import resize
import numpy as np

np.set_printoptions(threshold=np.inf)


class CNN_network:
	def __init__(self):
		self.BATCH_SIZE = 10
		self.CLASSES = 27  # Default
		self.EPOCHS = 200  # Loops once throug all data
		self.IMG_H = 77
		self.IMG_W = 196
		self.CHANNELS = 1
		self.TRAIN_X_FILE = '../../data/train_letters.npy'
		self.TRAIN_Y_FILE = '../../data/train_labels.npy'
		self.TEST_X_FILE = '../../data/test_letters.npy'
		self.TEST_Y_FILE = '../../data/test_labels.npy'
		self.MODEL_SAVE = '../../data/backup_model.model'
		# Hyperparameters
		self.CONV_KERNELSIZES = [(3, 3), (3, 3), (5, 5)]
		self.CONV_FILTERS = [8, 16, 16]
		self.POOL_KERNELSIZE = [(2, 2)]
		self.DENSE = [128]
		self.DENSE_ACTIV = ['relu']
		self.OUTPUT_ACTIV = ['softmax']
		self.DROPOUT_1 = 0.1
		self.DROPOUT_2 = 0.2

		self.SAVE_EVERY = 1000  # steps

	def build_net(self, input_shape):
		model = Sequential()
		model.add(Conv2D(self.CONV_FILTERS[0], kernel_size=self.CONV_KERNELSIZES[0],
						 activation='relu',
						 input_shape=input_shape))
		model.add(Conv2D(self.CONV_FILTERS[1], self.CONV_KERNELSIZES[1], activation='relu'))
		model.add(MaxPooling2D(pool_size=self.POOL_KERNELSIZE[0]))
		model.add(Conv2D(self.CONV_FILTERS[2], self.CONV_KERNELSIZES[2], activation='relu'))
		model.add(Dropout(self.DROPOUT_1))
		model.add(Flatten())
		model.add(Dense(self.DENSE[0], activation=self.DENSE_ACTIV[0]))
		model.add(Dropout(self.DROPOUT_2))
		model.add(Dense(self.CLASSES, activation=self.OUTPUT_ACTIV[0]))
		return model

	def read_data(self):
		print("Loading data...")
		self.trainX = np.load(self.TRAIN_X_FILE, allow_pickle=True)
		self.trainY = np.load(self.TRAIN_Y_FILE, allow_pickle=True)
		self.CLASSES = len(self.trainY[0])  # Get real number of classes, default 27
		print("Train data loaded...")
		self.testX = np.load(self.TEST_X_FILE, allow_pickle=True)
		self.testY = np.load(self.TEST_Y_FILE, allow_pickle=True)
		print("Test data loaded...")
		print("Reshaping data...")
		max_dims = (77, 196)
		temp = []
		for image in self.trainX:
			heightPadding = max_dims[0] - image.shape[0]
			extraHeight = heightPadding % 2
			widthPadding = max_dims[1] - image.shape[1]
			extraWidth = widthPadding % 2
			newImage = copyMakeBorder(image, int(heightPadding / 2), int(heightPadding / 2) + extraHeight,
									  int(widthPadding / 2), int(widthPadding / 2) + extraWidth,
									  cv2.BORDER_CONSTANT, value=[255, 255, 255])
			temp.append(newImage)
		temp = np.expand_dims(temp, axis=3)
		print(temp.shape)
		self.trainX = temp
		temp = []
		for image in self.testX:
			heightPadding = max_dims[0] - image.shape[0]
			extraHeight = heightPadding % 2
			widthPadding = max_dims[1] - image.shape[1]
			extraWidth = widthPadding % 2
			newImage = copyMakeBorder(image, int(heightPadding / 2), int(heightPadding / 2) + extraHeight,
									  int(widthPadding / 2), int(widthPadding / 2) + extraWidth,
									  cv2.BORDER_CONSTANT, value=[255, 255, 255])
			temp.append(newImage)
		temp = np.expand_dims(temp, axis=3)
		print(temp.shape)
		self.testX = temp
		self.trainX = np.array(self.trainX)
		self.testX = np.array(self.testX)
		print(np.shape(self.trainX), type(self.trainX))
		print(np.shape(self.trainY), type(self.trainY))

	def compile_model(self, model):
		model.compile(loss=keras.losses.categorical_crossentropy,
					  optimizer=keras.optimizers.Adadelta(),
					  metrics=['accuracy'])
		return model

	def train_model(self, model):
		cb = keras.callbacks.ModelCheckpoint('../../data/weights{epoch:08d}.h5',
											 save_weights_only=True, period=self.SAVE_EVERY)
		model.fit(self.trainX, self.trainY,
				  batch_size=self.BATCH_SIZE,
				  epochs=self.EPOCHS,
				  verbose=1,
				  validation_data=(self.testX, self.testY),
				  callbacks=[cb])
		return model

	def evaluate_model(self, model):
		score = model.evaluate(self.testX, self.testY, verbose=0)
		print('Test loss: ', score[0])
		print('Test accuracy: ', score[1])

	def run_network(self):
		input_shape = (self.IMG_H, self.IMG_W, self.CHANNELS)
		self.read_data()
		model = self.build_net(input_shape)
		model = self.compile_model(model)
		model = self.train_model(model)
		self.evaluate_model(model)
		model.save(self.MODEL_SAVE)


if __name__ == '__main__':
	network = CNN_network()
	network.run_network()
