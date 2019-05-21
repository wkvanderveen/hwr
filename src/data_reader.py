import numpy as np 
import matplotlib.pylab as plt
import os
import cv2
import sys
np.set_printoptions(threshold=sys.maxsize)


class DataReader:
	def __init__(self):
		self.data = []
		self.path = '../data/letters/' #define path of example letter images
		self.save_path = '../data/'
		self.save_file = self.save_path + "letters"
		self.threshold = 200

	def read_letters(self):
		letters = []
		print("Reading letters from dataset")
		for filepath in os.listdir(self.path):
			sub_path = self.path+filepath
			for sub_file in os.listdir(sub_path):
				img_path = sub_path+'/'+sub_file
				#print(img_path)
				letters.append(plt.imread(img_path))
		letters = np.array(letters)
		print("Number of letters found in dataset:", letters.shape)
		np.save(self.save_file, letters)
		print("Letters saved as npy file: ", self.save_file)

	def binarize_images(self, data):
		binarized_images = []
		for image in data:
			binary_image = np.where(image>self.threshold, 1, 0)
			binarized_images.append(binary_image)
		return np.array(binarized_images)

	def read_test_data(self):
		# Read the test data and call the preprocessing function here
		pass


reader = DataReader()
reader.read_letters()
#reader.binarize_images()