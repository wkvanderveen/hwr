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



class CNN_network:
	def __init__(self):
		self.BATCH_SIZE = 128
		self.CLASSES = 3
		self.EPOCHS = 12
		self.IMG_H = self.IMG_W = 38
		self.CHANNELS = 1
		self.TRAIN_X_FILE = '../../data/train_letters.npy' 
		self.TRAIN_Y_FILE = '../../data/train_labels.npy'
		self.TEST_X_FILE = '../../data/test_letters.npy' 
		self.TEST_Y_FILE = '../../data/test_labels.npy'
		self.MODEL_SAVE = '../../backup_model.model'

	def build_net(self, input_shape):
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
		model.add(Dense(self.CLASSES, activation='softmax'))
		return model

	def read_data(self):
		self.trainX = np.load(self.TRAIN_X_FILE, allow_pickle=True)
		self.trainY = np.load(self.TRAIN_Y_FILE, allow_pickle=True)
		self.testX = np.load(self.TEST_X_FILE, allow_pickle=True)
		self.testY = np.load(self.TEST_Y_FILE, allow_pickle=True)
		self.trainX = [resize(image, (self.IMG_H,self.IMG_W, self.CHANNELS)) for image in self.trainX]
		self.testX = [resize(image, (self.IMG_H,self.IMG_W, self.CHANNELS))  for image in self.testX]
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
		model.fit(self.trainX, self.trainY,
			batch_size=self.BATCH_SIZE,
			epochs=self.EPOCHS, 
			verbose=1,
			validation_data=(self.testX,self.testY))
		return model

	def evaluate_model(self, model):
		score = model.evaluate(self.testX, self.testY, verbose=0)
		print('Test loss: ', score[0])
		print('Test accuracy: ', score[1])

	def run_network(self):
		input_shape = (self.IMG_H, self.IMG_W, self.CHANNELS)
		model = self.build_net(input_shape)
		self.read_data()
		model = self.compile_model(model)
		model = self.train_model(model)
		self.evaluate_model(model)
		model.save(self.MODEL_SAVE)

if __name__ == '__main__':
	network = CNN_network()
	network.run_network()

	