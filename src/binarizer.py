'''
binarizer.py

This file contains the code used for binarizing the input image


'''
import cv2
import mahotas
from os.path import join, abspath
import os
import numpy as np
import skimage

class Binarizer:
	def __init__(self):
		pass

	def get_negative(self, img):
		(y_max, x_max) = np.shape(img)

		for y in range(y_max):
			for x in range(x_max):
				img[y][x] = 255 - img[y][x]
		return img

	def binarize_otsu(self, img):
		thres = mahotas.thresholding.otsu(img) #create otsu threshold
		(y_max, x_max) = np.shape(img)

		for y in range(y_max):
			for x in range(x_max):
				img[y][x] = 255 if img[y][x] > thres else 0
		return img

	def binarize_simple(self, img, thres = 100):
		(y_max, x_max) = np.shape(img)

		for y in range(y_max):
			for x in range(x_max):
				img[y][x] = 255 if img[y][x] > thres else 0
		return img

	def open(self, img, se_size = 5):
		se = np.ones((se_size, se_size), np.uint8)
		return cv2.morphologyEx(img, cv2.MORPH_OPEN, se, iterations = 1)

	def close(self, img, se_size = 5):
		se = np.ones((se_size, se_size), np.uint8)
		return cv2.morphologyEx(img, cv2.MORPH_CLOSE, se, iterations = 1)


	def erode(self, img, se_size = 5):
		se = np.ones((se_size, se_size), np.uint8)
		return cv2.erode(img, se, iterations = 1)

	def dilate(self, img, se_size = 5):
		se = np.ones((se_size, se_size), np.uint8)
		return cv2.dilate(img, se, iterations = 1)





if __name__ == '__main__':
	#look for files on path
	

	b = Binarizer()

	# #binarize images. This code is just used to test if everything executes
	# try:
	# 	path = join(join(join(abspath('..'), 'data'), 'letters'), 'Alef')
	# 	print("looking for files in " + path)
	# 	imgs = [cv2.imread(join(path, file)) for file in os.listdir(path) if (file.endswith('.png') or file.endswith('.jpg'))]
	# 	for img in imgs:
	# 		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to grayscale

	# 		#manipulate image
	# 		alt_img = b.open_img(img)
	# 		alt_img = b.close_img(alt_img)

	# 		img_besides = np.concatenate((img, alt_img), axis=1)

	# 		cv2.imshow('img', img_besides)
	# 		cv2.waitKey(0)
	# except KeyboardInterrupt:
	# 	quit()


	# Test on actual dead sea scroll image
	path = join(abspath('..'), 'data')
	img = cv2.imread(join(path, 'test_img.jpg'))

	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to grayscale
	#print(np.shape(img))

	alt_img = b.open(img, 2)
	alt_img = b.erode(alt_img, 5)
	alt_img = b.binarize_simple(alt_img, 180)

	img_besides = np.concatenate((img, alt_img), axis=1)

	cv2.imshow('img', alt_img)
	cv2.waitKey(0)