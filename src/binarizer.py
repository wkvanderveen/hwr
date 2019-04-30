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
import matplotlib.pyplot as plt

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
		'''
		Creates a mask from the image to segement out the backgroud. 
		Uses Otsu to create thresholds to use when the mask is applied.
		Outputs a binarized image.
		Works best with bw images (not binarized)
		'''

		mask = b.dilate(img, 20)
		mask = b.erode(mask, 20)
		(y_max, x_max) = np.shape(img)
		img_thres = mahotas.thresholding.otsu(img) #create otsu threshold
		mask_thres = mahotas.thresholding.otsu(mask) #create otsu threshold
		img_out = np.zeros((y_max, x_max)) #alloc memory

		for y in range(y_max):
			for x in range(x_max):
				img_out[y][x] = 255 if (img[y][x] < img_thres and mask[y][x] > mask_thres) else 0
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

	def get_hist_bin_img(self, img):
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
			x_hist[idx] = np.sum(img[:, idx]) / 255

		return (y_hist, x_hist)

	def crop_img_hist(self, img, y_hist, x_hist):
		'''
		Uses histograms to crop the image
		Assumes the scroll is centered (it often is)
		'''
		(y_max, x_max) = np.shape(img)
		(y_hist, x_hist) = self.get_hist_bin_img(img)
		center_y = uint8(y_max/2)
		center_x = uint8(x_max/2)
		left = 0
		right = x_max
		top = 0
		bot = y_max

		#calculate bounds

		for idx in range(center_x, x_max - 1, 1): #loop from center to right of image, step size = 1
			if x_hist[idx] < x_hist[idx-1] and x_hist[idx] < x_hist[idx+1]: #found local minima
				right = idx
				break

		for idx in range(center_x, 0 + 1, -1): #loop from center to left of image,step size = -1
			if x_hist[idx] < x_hist[idx-1] and x_hist[idx] < x_hist[idx+1]: #found local minima
				left = idx
				break

		for idx in range(center_y, y_max + 1, -1): #loop from center to left of image,step size = -1
			if x_hist[idx] < x_hist[idx-1] and x_hist[idx] < x_hist[idx+1]: #found local minima
				top = idx
				break

		for idx in range(center_y, 0 + 1, -1): #loop from center to left of image,step size = -1
			if x_hist[idx] < x_hist[idx-1] and x_hist[idx] < x_hist[idx+1]: #found local minima
				bot = idx
				break

		print('l, r, t, b, centerx, centery', left, right, top, bot, center_x, center_y)
		return img[bot:top, left:right]






				






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

	# ### working example
	# img = b.apply_mask(bw_img)
	# img = b.dilate(img, 2)
	# img = b.dilate(img, 2)
	# img = b.erode(img, 2)

	# img = b.erode(img, 4)
	# img = b.dilate(img, 4)
	# ### end example

	img = b.binarize_otsu(bw_img)
	(y_hist, x_hist) = b.get_hist_bin_img(img)


	##convolve histograms with gaussian difference filter
	s, m = 2, 5
	denom = np.sqrt(2*np.pi*s*s)
	gauss = [np.exp(-z*z/(2*s*s))/denom for z in range(-m, m+1)] 
	window = np.convolve(gauss, [1, -1])
	y_hist_conv = np.convolve(y_hist, window)

	plt.figure(1)
	plt.subplot(211)
	plt.plot(y_hist)
	plt.subplot(212)
	plt.plot(y_hist_conv)
	plt.show()

	print(y_hist_conv, max(y_hist_conv))

	print(sum(y_hist_conv != 0))

	print(len(y_hist), len(y_hist_conv))
	# print(x_hist, max(x_hist))



	# img_besides = np.concatenate((img, alt_img), axis=1)

	# cv2.imshow('img', img)
	# cv2.waitKey(0)
	# cv2.imwrite(join(path, 'img_out.png'), img)