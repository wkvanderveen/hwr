'''
line_segmenter.py

This file includes code for the line segmentation part of the pipeline
'''
import cv2
import mahotas
from os.path import join, abspath
import os
import numpy as np
import skimage
import matplotlib.pyplot as plt
from binarizer import Binarizer

LINE_WIDTH = 20 #line width per stroke in image

class Line_segmenter:
	def __init__(self):
		pass

	def create_v_histogram(self, img):
		(y_max, x_max) = np.shape(img)
		hist = np.zeros((y_max))

		for idx in range(y_max):
			hist[idx] = np.sum(img[idx, :]) / 255

		return hist

	def smooth_hist(self, hist, smooth_len):
		#applies an averager over the histogram to smooth itq
		window = np.repeat(1.0/np.float(smooth_len), smooth_len)
		return np.convolve(hist, smooth_len)


	def histogram(self, img):
		(y_max, x_max) = np.shape(img)

		numlines = y_max // LINE_WIDTH

		for idx in range(numlines):
			stroke = img[:, idx*LINE_WIDTH:(idx+1)*LINE_WIDTH]
			hist = self.smooth_hist(self.create_v_histogram(img), 201)
			plt.plot(hist)
			plt.show()
			print(len(hist), y_max)







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

	l = Line_segmenter()
	l.histogram(img)












	cv2.imshow('img', img)
	cv2.waitKey(0)
	cv2.imwrite(join(path, 'img_out.png'), img)
	print("saved converted image \"" + img_name + "\" to \"" + join(path, 'img_out.png') + "\"")
