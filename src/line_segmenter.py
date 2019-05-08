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

STROKE_WIDTH = 20 #line width per stroke in image
LINE_WIDTH = 50 #expected line width


class Line_segmenter:
	def __init__(self):
		pass

	def create_v_histogram(self, img):
		(y_max, x_max) = np.shape(img)
		hist = np.zeros((y_max))

		for idx in range(y_max):
			# hist[idx] = np.sum(img[idx, :]) / 255
			hist[idx] = (img[idx, :] == 0).sum()

		return hist

	def smooth_hist(self, hist, smooth_len):
		#applies an averager over the histogram to smooth it
		window = np.repeat(1.0/np.float(smooth_len), smooth_len)
		return np.convolve(hist, smooth_len)


	def histogram(self, img):
		(y_max, x_max) = np.shape(img)

		numlines = y_max // STROKE_WIDTH

		peaks_arr = []

		for idx in range(numlines):
			stroke = img[:, idx*STROKE_WIDTH:(idx+1)*STROKE_WIDTH]
			hist = self.create_v_histogram(img)#self.smooth_hist(self.create_v_histogram(img), 1001)
			peaks = []

			#get peaks (very simplistic approach)
			for idx2 in range(y_max // LINE_WIDTH):
				hist_section = hist[idx2*LINE_WIDTH:(idx2+1)*LINE_WIDTH]
				#max_val = np.max(hist_section)
				#print(np.where(hist_section == hist_section.max()))
				maxes = np.where(hist_section == hist_section.max())[0]
				if(len(maxes) < 10):
					peak = maxes[len(maxes)//2] #get center element
					peak += idx2*LINE_WIDTH		#add offset
					peaks.append(peak)			#append to set

				
				#visualize
				hist = np.array(hist, dtype=np.float)
				hist = hist - hist.min()
				hist = hist / hist.max()
				hist * 1000.0 #scale to [0,n]

				plt.plot(hist)
				plt.show()
 
				fig, ax = plt.subplots()
				ax.imshow(img, extent=[0, x_max, 0, y_max])
				# ax.plot(hist)
				ax.barh(range(y_max), hist)#, height=height)
				# plt.plot(hist)
				# peaks = np.array(peaks)
				# print(peaks)
				# plt.plot(np.array(peaks), np.repeat(1000, len(peaks)))
				plt.show()
				# peaks_arr.append(peaks)


		# #visualize
		# img2 = img #np.uint8(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR))
		# for idx in range(numlines-1):
		# 	for peak in peaks_arr[idx]:
		# 		for pixel in range(idx*LINE_WIDTH, (idx+1)*LINE_WIDTH - 1, 1):
		# 			img2[peak, pixel] = 155#[0, 255, 0] #make green

		# cv2.imshow(img2)
		# cv2.waitKey(0)





if __name__ == '__main__':
	b = Binarizer()

	# Test on actual dead sea scroll image
	path = join(abspath('..'), 'data')
	img_name = 'P21-Fg006-R-C01-R01'#'P22-Fg008-R-C01-R01' #'P513-Fg001-R-C01-R01' 'P106-Fg002-R-C01-R01' 'P21-Fg006-R-C01-R01';
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
