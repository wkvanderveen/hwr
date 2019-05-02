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

	def binarize_image(self, img):
		'''
		This function contains the (currently) optimal binarization pipeline for the images
		It expects a bw image 
		'''
		img = b.apply_mask(img)

		#remove noise
		img = b.dilate(img, 2)
		img = b.dilate(img, 2)
		img = b.erode(img, 2)

		#restore characters
		img = b.erode(img, 4)
		img = b.dilate(img, 4)

		img = b.erode(img, 4)
		img = b.dilate(img, 4)

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

		mask = self.dilate(img, 20)
		mask = self.erode(mask, 20)

		(y_max, x_max) = np.shape(img)
		img_thres = mahotas.thresholding.otsu(img) #create otsu threshold
		mask_thres = mahotas.thresholding.otsu(mask) #create otsu threshold
		img_out = np.zeros((y_max, x_max)) #alloc memory

		for y in range(y_max):
			for x in range(x_max):
				img_out[y][x] = 0 if (img[y][x] < img_thres and mask[y][x] > mask_thres) else 255
		return img_out

	def compare_imgs(self, col_img, bw_img):
		img = np.zeros(np.shape(bw_img)) #alloc memory
		(y_max, x_max) = np.shape(bw_img)
		for y in range(y_max):
			for x in range(x_max):
				if self.are_equal(bw_img[y][x], col_img[y][x][0], 10) and \
					self.are_equal(bw_img[y][x], col_img[y][x][1], 10) and \
					self.are_equal(bw_img[y][x], col_img[y][x][2], 10):
					img[y][x] = 255
				else:
					img[y][x] = bw_img[y][x]
		return img

	def compare_imgs_2(self, col_img, bw_img):
		img = np.zeros(np.shape(bw_img)) #alloc memory
		(y_max, x_max) = np.shape(bw_img)

		low = 47
		high = 55

		for y in range(y_max):
			for x in range(x_max):
				if col_img[y][x][0] >= low and col_img[y][x][0] <= high  and \
					col_img[y][x][1] >= low and col_img[y][x][1] <= high and \
					col_img[y][x][2] >= low and col_img[y][x][2] <= high and \
					bw_img[y][x] <= 5:
					img[y][x] = 255
				else:
					img[y][x] = 0#bw_img[y][x]
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

	def are_equal(self, x, y, range_):
		return x + range_ >= y and x - range_ <= y

	def crop_img_hist(self, img):
		'''
		Uses histograms to crop the image
		Assumes the scroll is centered (it often is)
		Assumes the image is ot yet binarized
		'''
		img = self.binarize_otsu(img)
		(y_max, x_max) = np.shape(img)
		(y_hist, x_hist) = self.get_hist_bin_img(img)
		center_y = np.uint(y_max/2)
		center_x = np.uint(x_max/2)
		left = 0
		right = x_max
		top = 0
		bot = y_max

		#calculate bounds
		ran = 20 #values should be roughly similar for ran consecutive values to crop the image there, should be an even int

		count = 0
		val = x_hist[center_x]
		for idx in range(center_x - 1, 0, -1): #loop from center to left of image, step size = 1
			if self.are_equal(x_hist[idx], val, 10):
				count += 1
			else:
				count = 0
				val = x_hist[idx]

			if count == ran:
				left = np.uint(idx + ran/2)
				break

		count = 0
		val = x_hist[center_x]
		for idx in range(center_x + 1, x_max, 1): #loop from center to right of image, step size = 1
			if self.are_equal(x_hist[idx], val, 10):
				count += 1
			else:
				count = 0
				val = x_hist[idx]

			if count == ran:
				right = np.uint(idx - ran/2)
				break


		count = 0
		val = y_hist[center_y]
		for idx in range(center_y - 1, 0, -1): #loop from center to right of image, step size = 1
			if self.are_equal(y_hist[idx], val, 10):
				count += 1
			else:
				count = 0
				val = y_hist[idx]

			if count == ran:
				top = np.uint8(idx + ran/2)
				break

		count = 0
		val = y_hist[center_y]
		for idx in range(center_y + 1, y_max, 1): #loop from center to right of image, step size = 1
			if self.are_equal(y_hist[idx], val, 10):
				count += 1
			else:
				count = 0
				val = y_hist[idx]

			if count == ran:
				bot = np.uint8(idx - ran/2)
				break

		print('l, r, t, b, centerx, centery', left, right, top, bot, center_x, center_y)
		return (left, right, bot, top)


	def create_and_show_histogram(self, img):
		img = b.binarize_otsu(bw_img.copy())
		(y_hist, x_hist) = b.get_hist_bin_img(img)

		s, m = 3, 10
		denom = np.sqrt(2*np.pi*s*s)
		gauss = [np.exp(-z*z/(2*s*s))/denom for z in range(-m, m+1)] 
		window = np.convolve(gauss, [1, -1])
		y_hist_conv = np.convolve(y_hist, window)

		x_hist_conv = np.convolve(x_hist, window)

		# size = 200
		# window = np.repeat(1.0/np.float(size), size)
		# y_hist_conv = np.convolve(y_hist, window)
		# x_hist_conv = np.convolve(x_hist, window)

		plt.figure(1)
		plt.title(img_name)
		plt.subplot(221)
		plt.plot(y_hist)
		plt.title('y_hist, '+ img_name)
		plt.subplot(222)
		plt.plot(y_hist_conv)
		plt.title('y_hist_conv, '+ img_name)

		plt.subplot(223)
		plt.plot(x_hist)
		plt.title('x_hist, '+ img_name)
		plt.subplot(224)
		plt.plot(x_hist_conv)
		plt.title('x_hist_conv, '+ img_name)

		plt.show()


if __name__ == '__main__':

	b = Binarizer()

	# Test on actual dead sea scroll image
	path = join(abspath('..'), 'data')
	img_name = 'P22-Fg008-R-C01-R01' #'P513-Fg001-R-C01-R01' 'P106-Fg002-R-C01-R01' 'P21-Fg006-R-C01-R01.jpg';
	col_img = cv2.imread(join(join(path, 'image-data'), img_name + '.jpg'))
	bw_img =  cv2.imread(join(join(path, 'image-data'), img_name + '-fused.jpg'))
	print("converting image: " + img_name)

	bw_img = cv2.cvtColor(bw_img,cv2.COLOR_BGR2GRAY) #convert to grayscale

	img = b.binarize_image(bw_img)

	cv2.imshow('img', img)
	cv2.waitKey(0)
	cv2.imwrite(join(path, 'img_out.png'), img)
	print("saved converted image \"" + img_name + "\" to \"" + join(path, 'img_out.png') + "\"")