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

	def apply_mask(self, img):
		mask = b.dilate(img, 20)
		mask = b.erode(mask, 20)
		(y_max, x_max) = np.shape(img)
		img_thres = mahotas.thresholding.otsu(img) #create otsu threshold
		mask_thres = mahotas.thresholding.otsu(mask) #create otsu threshold
		img_out = np.zeros((y_max, x_max)) #alloc memory

		for y in range(y_max):
			for x in range(x_max):
				img_out[y][x] = 255 if (img[y][x] < img_thres and mask[y][x] > mask_thres) else 0
		
		# for y in range(y_max):
		# 	for x in range(x_max):
		# 		img[y][x] = 255 if (img[y][x] < img_thres and img[y][x] > mask_thres) else 0

		return img_out

	def compare_imgs(self, col_img, bw_img):
		img = np.zeros(np.shape(bw_img)) #alloc memory
		(y_max, x_max) = np.shape(bw_img)
		for y in range(y_max):
			for x in range(x_max):
				if bw_img[y][x] == col_img[y][x][0] and \
					bw_img[y][x] == col_img[y][x][1] and \
					bw_img[y][x] == col_img[y][x][2]:
					img[y][x] = 255
				else:
					img[y][x] = bw_img[y][x]
		return img

	def get_hist_bin_img(self, img, dim = 0):
		'''
		Bins the amount of 255 pixels in rows and columns, then returns the histograms
		This function expects a binarized image
		'''
		(y_max, x_max) = np.shape(img)
		y_hist = np.zeros(y_max)
		x_hist = np.zeros(x_max)

		for idx in range(y_max):
			y_hist[idx] = np.sum(img[idx, :]) / 255

		for idx in range(x_max):
			x_hist[idx] = np.sum(img[idx, :]) / 255

		return (y_hist, x_hist)






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
	col_img = cv2.imread(join(path, 'test_img.jpg'))
	# col_img = cv2.imread(join(join(path, 'image-data'), 'P21-Fg006-R-C01-R01.jpg'))
	# bw_img =  cv2.imread(join(join(path, 'image-data'), 'P21-Fg006-R-C01-R01-fused.jpg'))
	# print(np.shape(bw_img))
	# print(np.shape(col_img))
	

	bw_img = cv2.cvtColor(col_img,cv2.COLOR_BGR2GRAY) #convert to grayscale

	# # img = b.compare_imgs(col_img, bw_img)

	# bw_img = b.binarize_otsu(bw_img)


	# img = b.binarize_simple(bw_img, 100)
	img = b.apply_mask(bw_img)
	img = b.dilate(img, 2)
	img = b.dilate(img, 2)
	img = b.erode(img, 2)

	img = b.erode(img, 4)
	img = b.dilate(img, 4)




	# img_besides = np.concatenate((img, alt_img), axis=1)

	cv2.imshow('img', img)
	cv2.waitKey(0)
	cv2.imwrite(join(path, 'img_out.png'), img)