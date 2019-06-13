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
np.set_printoptions(threshold=np.inf)

class CNN_network:
	def __init__(self):
		self.BATCH_SIZE = 1
		self.CLASSES = 27 #Default
		self.EPOCHS = 1 #Loops once throug all data
		self.IMG_H = self.IMG_W = 38
		self.CHANNELS = 1
		self.TRAIN_X_FILE = '../../data/train_letters.npy' 
		self.TRAIN_Y_FILE = '../../data/train_labels.npy'
		self.TEST_X_FILE = '../../data/test_letters.npy' 
		self.TEST_Y_FILE = '../../data/test_labels.npy'
		self.MODEL_SAVE = '../../data/backup_model.model'
		#Hyperparameters
		self.CONV_KERNELSIZES = [(3,3), (3,3), (5,5)]
		self.CONV_FILTERS = [8, 16, 16]
		self.POOL_KERNELSIZE = [(2,2)]
		self.DENSE = [128]
		self.DENSE_ACTIV = ['relu']
		self.OUTPUT_ACTIV = ['softmax']
		self.DROPOUT_1 = 0.1
		self.DROPOUT_2 = 0.2

		self.SAVE_EVERY = 1000 #steps


	def build_net(self, input_shape):
		model = Sequential()
		model.add(Conv2D(self.CONV_FILTERS[0], kernel_size=self.CONV_KERNELSIZES[0],
		                 activation='relu',
		                 input_shape= input_shape))
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
		self.CLASSES = len(self.trainY[0]) #Get real number of classes, default 27
		self.trainX = self.trainX[1:20]
		self.trainY = self.trainY[1:20]
		print(self.trainX[0])
		print("Train data loaded...")
		self.testX = np.load(self.TEST_X_FILE, allow_pickle=True)
		self.testY = np.load(self.TEST_Y_FILE, allow_pickle=True)
		print("Test data loaded...")
		print("Reshaping data...")
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
		cb = keras.callbacks.ModelCheckpoint('../../data/weights{epoch:08d}.h5', 
                                     save_weights_only=True, period=self.SAVE_EVERY)
		model.fit(self.trainX, self.trainY,
			batch_size=self.BATCH_SIZE,
			epochs=self.EPOCHS, 
			verbose=1,
			validation_data=(self.testX,self.testY),
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

	