from __future__ import print_function
import cv2
import keras
from PIL import Image
from cv2.cv2 import copyMakeBorder
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from skimage.transform import resize
from skimage.util import pad
import numpy as np

np.set_printoptions(threshold=np.inf)


class CNN_network:
	def __init__(self):
		self.BATCH_SIZE = 10
		self.CLASSES = 27  # Default
		self.EPOCHS = 3  # Loops once throug all data
		self.IMG_H = 39
		self.IMG_W = 39
		self.CHANNELS = 1
		self.TRAIN_X_FILE = '../../data/train_letters.npy'
		self.TRAIN_Y_FILE = '../../data/train_labels.npy'
		self.TRAIN_X_FILE_AUG = '../../data/train_letters_aug.npy' 
		self.TRAIN_Y_FILE_AUG = '../../data/train_labels_aug.npy'
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
		self.TRAIN_ON_AUGMENT = True

		self.SAVE_EVERY = 1000  # steps
		self.EARLY_STOPPING_PATIENCE = 5 # Epochs

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

	def read_data(self, X_path, Y_path, mode):
		print("Loading data...")

		if mode == 'train':
			self.trainX = np.load(X_path, allow_pickle=True)
			self.trainY = np.load(Y_path, allow_pickle=True)
			self.CLASSES = len(self.trainY[0])  # Get real number of classes, default 27
			print("Train data loaded...")
		if mode =='augmented':
			self.trainX_AUG = np.load(X_path, allow_pickle=True)
			self.trainY_AUG = np.load(Y_path, allow_pickle=True)
			self.CLASSES = len(self.trainY_AUG[0])  # Get real number of classes, default 27
		if mode == 'test':
			self.testX = np.load(X_path, allow_pickle=True)
			self.testY = np.load(Y_path, allow_pickle=True)
			print("Test data loaded...")

		print("Reshaping data...")
		max_dims = (77, 77)
		temp = []
		if mode == 'train':
			for image in self.trainX:
				heightPadding = max_dims[0] - image.shape[0]
				extraHeight = heightPadding % 2
				widthPadding = max_dims[1] - image.shape[1]
				extraWidth = widthPadding % 2
				newImage = copyMakeBorder(image, int(heightPadding / 2), int(heightPadding / 2) + extraHeight,
										  int(widthPadding / 2), int(widthPadding / 2) + extraWidth,
										  cv2.BORDER_CONSTANT, value=[1, 1, 1])
				newImage = cv2.resize(newImage, (39,39))
				# if counter % 500 == 0:
				# 	cv2.imshow('train',newImage)
				# 	cv2.waitKey(0)
				# 	cv2.destroyAllWindows()
				# counter += 1
				temp.append(newImage)
			temp = np.expand_dims(temp, axis=3)
			self.trainX = temp
			self.trainX = np.array(self.trainX)
			print(np.shape(self.trainX), type(self.trainX))
		if mode == 'augmented':
			for image in self.trainX_AUG:
				heightPadding = max_dims[0] - image.shape[0]
				extraHeight = heightPadding % 2
				widthPadding = max_dims[1] - image.shape[1]
				extraWidth = widthPadding % 2
				newImage = copyMakeBorder(image, int(heightPadding / 2), int(heightPadding / 2) + extraHeight,
										  int(widthPadding / 2), int(widthPadding / 2) + extraWidth,
										  cv2.BORDER_CONSTANT, value=[1, 1, 1])
				newImage = cv2.resize(newImage, (39,39))
				# if counter % 500 == 0:
				# 	cv2.imshow('train',newImage)
				# 	cv2.waitKey(0)
				# 	cv2.destroyAllWindows()
				# counter += 1
				temp.append(newImage)
			temp = np.expand_dims(temp, axis=3)
			self.trainX_AUG = temp
			self.trainX_AUG = np.array(self.trainX_AUG)
			print(np.shape(self.trainX_AUG), type(self.trainX_AUG))

		if mode == 'test':
			for image in self.testX:
				heightPadding = max_dims[0] - image.shape[0]
				extraHeight = heightPadding % 2
				widthPadding = max_dims[1] - image.shape[1]
				extraWidth = widthPadding % 2
				newImage = copyMakeBorder(image, int(heightPadding / 2), int(heightPadding / 2) + extraHeight,
										  int(widthPadding / 2), int(widthPadding / 2) + extraWidth,
										  cv2.BORDER_CONSTANT, value=[1, 1, 1])
				newImage = cv2.resize(newImage, (39,39))
				# print(self.testY[count])
				# cv2.imshow('train',newImage)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				temp.append(newImage)
			temp = np.expand_dims(temp, axis=3)
			print(temp.shape)
			self.testX = temp
			self.testX = np.array(self.testX)
			print(np.shape(self.testX), type(self.testX))
		

	def compile_model(self, model):
		model.compile(loss=keras.losses.categorical_crossentropy,
					  optimizer=keras.optimizers.Adam(),#keras.optimizers.Adadelta(),
					  metrics=['accuracy'])
		return model

	def train_model(self, model, trainX, trainY):
		cb = keras.callbacks.ModelCheckpoint('../../data/weights{epoch:08d}.h5',
											 save_weights_only=True, period=self.SAVE_EVERY)
		es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.EARLY_STOPPING_PATIENCE, restore_best_weights=True)
		model.fit(trainX, trainY,
				  batch_size=self.BATCH_SIZE,
				  epochs=self.EPOCHS,
				  verbose=1,
				  validation_data=(self.testX, self.testY))#,
				  #callbacks=[cb, es])
		return model

	def evaluate_model(self, model):
		score = model.evaluate(self.testX, self.testY, verbose=0)
		print('Test loss: ', score[0])
		print('Test accuracy: ', score[1])

	def run_network(self):
		input_shape = (self.IMG_H, self.IMG_W, self.CHANNELS)
		#self.read_data()
		self.read_data(self.TRAIN_X_FILE, self.TRAIN_Y_FILE, 'train')
		self.read_data(self.TEST_X_FILE, self.TEST_Y_FILE, 'test')
		model = self.build_net(input_shape)
		model = self.compile_model(model)
		model = self.train_model(model, self.trainX, self.trainY)
		if self.TRAIN_ON_AUGMENT:
			# EMPTY PREVIOUS TRAIN DATA, FREEING MEMORY
			self.trainX = []
			self.trainY = []
			self.read_data(self.TRAIN_X_FILE_AUG, self.TRAIN_Y_FILE_AUG, 'augmented')
			model = self.train_model(model, self.trainX_AUG, self.trainY_AUG)
		self.evaluate_model(model)
		model.save(self.MODEL_SAVE)


if __name__ == '__main__':
	network = CNN_network()
	network.run_network()
