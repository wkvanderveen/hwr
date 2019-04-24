'''
binarizer.py

This file contains the code used for binarizing the input image


'''
import cv2
import mahotas
from os.path import join, abspath
import os
import numpy as np

class Binarizer:
	def __init__(self):
		pass

	def binarize_otsu(self, img):
		thres = mahotas.thresholding.otsu(img) #create otsu threshold
		(y_max, x_max) = np.shape(img)

		for y in range(y_max):
			for x in range(x_max):
				img[y][x] = 255 if img[y][x] > thres else 0
		return img




if __name__ == '__main__':
	#look for files on path
	path = join(abspath('..'), 'data')
	print("looking for files in " + path)
	imgs = [cv2.imread(join(path, file)) for file in os.listdir(path) if file.endswith('.png')]

	b = Binarizer()

	#binarize images
	for img in imgs:
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to grayscale
		#print(np.shape(img))
		bin_img = b.binarize_otsu(img)

		cv2.imshow('img', bin_img)
		cv2.waitKey(0)
