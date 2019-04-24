import numpy as np 
import matplotlib.pylab as plt
import os
import cv2


class DataReader:
	def __init__(self):
		self.data = []
		self.path = '../../data/letters/' #define path of example letter images
		self.save_path = '../../data/'
		self.save_file = self.save_path + "letters"

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

	def read_test_data(self):
		# Read the test data and call the preprocessing function here
		pass


reader = DataReader()
reader.read_letters()