'''
line_segmenter.pyx

This file includes code for the line segmentation part of the pipeline, rewritten in Cython
'''
import cython
import cv2
from os.path import join, abspath
import os
import numpy as np
import skimage
import matplotlib.pyplot as plt
from binarizer import Binarizer

cimport numpy as np

cdef int STROKE_WIDTH = 30 #line width per vertical stroke in image
cdef double THRESH_RATIO = 0.1#Every histogram peak above this ratio is considered a line
cdef int LINE_WIDTH = 5 #expected line width


class Line_segmenter:
	def __init__(self):
		pass

	def create_v_histogram(self, np.ndarray[np.uint8_t, ndim=2] img):
		cdef int idx, y_max, x_max
		(y_max, x_max) = np.shape(img)
		cdef np.ndarray[np.uint64_t, ndim=1] hist = np.zeros((y_max), dtype=np.uint64)

		for idx in range(y_max):
			hist[idx] = (img[idx, :] == 0).sum()

		return hist

	def smooth_hist(self, np.ndarray[np.uint64_t, ndim=1] hist, unsigned int smooth_len):
		#applies an averager over the histogram to smooth it
		cdef np.ndarray[np.float64_t, ndim=1] window = np.repeat(1.0/np.float64(smooth_len), smooth_len)
		return  np.uint64(np.convolve(hist, smooth_len)) #cast back to uints, conv returns floats


	def histogram(self, img):
		(y_max, x_max) = np.shape(img)
		STROKE_WIDTH = x_max
		numlines = x_max // STROKE_WIDTH
		peaks_arr = []
		segmented_images = []

		for idx in range(numlines):
			stroke = img[:, idx*STROKE_WIDTH:(idx*STROKE_WIDTH)+STROKE_WIDTH]
			hist = self.create_v_histogram(stroke)
			line_begin_coords = []
			line_end_coords = []

			hist[hist < int(THRESH_RATIO*hist.max())] = 0
			seek_line_end = False #vlak to for seeking dip in histogram
			for idx, val in enumerate(hist):
				if not seek_line_end:
					if val != 0:
						line_begin_coords.append((0, idx))
						line_end_coords.append((STROKE_WIDTH, idx))
						seek_line_end = True
				if seek_line_end:
						
					if val == 0:
						if idx > int(line_begin_coords[-1][1]+LINE_WIDTH): #get previous index
							line_begin_coords.append((0, idx))
							line_end_coords.append((STROKE_WIDTH, idx))
						else: #distance not large enough to be considered a line
							line_begin_coords.pop()
							line_end_coords.pop()
						seek_line_end = False
			stroke = cv2.cvtColor(stroke.astype(np.uint8),cv2.COLOR_GRAY2RGB)
			color = (0,255,0)
			for idx in range(0, len(line_begin_coords)):
				(x_start, y_start) = line_begin_coords[idx]
				(x_end, y_end) = line_end_coords[idx]
				
				if color == (0,255,0):
					color = (0,0,255)
				else:
					color = (0,255,0)
					cv2.line(stroke, (x_start,y_start), (x_end,y_end),color,2)

			#visualize
			hist = np.array(hist, dtype=np.float)
			hist = hist - hist.min()
			hist = hist / hist.max()
			hist * 1000.0 #scale to [0,n]
			
			segmented_images.append(stroke)
		return segmented_images, hist

	def show_segm_img(self, img_arr):
		full_image = np.array(img_arr[0])
		img_arr = np.array(img_arr)

		for image in img_arr[1:]:
			full_image = np.concatenate((full_image, image), axis=1)
		print(full_image.shape)
		cv2.imshow('stroke',full_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def get_segm_img(self, img_arr):
		full_image = np.array(img_arr[0])
		img_arr = np.array(img_arr)

		for image in img_arr[1:]:
			full_image = np.concatenate((full_image, image), axis=1)
		return full_image

	def get_minima(self, np.ndarray[np.uint64_t, ndim=1] hist, unsigned int min_dist = 30, unsigned int smooth=101, unsigned int stride=21):
		cdef int idx
		cdef list temp_minima = []
		cdef list minima = []
		hist = self.smooth_hist(hist, smooth)
		cdef double mean = np.mean(hist)
		#add all minima to a list
		for idx in range(stride, len(hist) - stride, 1):
			if hist[idx] < hist[idx-stride] and hist[idx] < hist[idx+stride] and hist[idx] < mean:
				temp_minima.append(idx)

		#refine list
		for idx in range(len(temp_minima) - 1):
			if np.abs(temp_minima[idx] - temp_minima[idx+1]) > min_dist: #there is enough distance between the two minima
				minima.append(temp_minima[idx])

		if len(temp_minima) > 0:
			if len(minima) == 0: #append last minima
				minima.append(temp_minima[-1])
			elif temp_minima[-1] - minima[-1] > min_dist:
				minima.append(temp_minima[-1])
		return minima
		

if __name__ == '__main__':
	b = Binarizer()
	image_arr = ['124-Fg004', 'P123-Fg002-R-C01-R01', 'P21-Fg006-R-C01-R01', 'P22-Fg008-R-C01-R01', 'P513-Fg001-R-C01-R01', 'P106-Fg002-R-C01-R01', 'P21-Fg006-R-C01-R01']
	# Test on actual dead sea scroll image
	for img_name in image_arr:
		path = join(abspath('..'), 'data')
		col_img = cv2.imread(join(join(path, 'image-data'), img_name + '.jpg'))
		bw_img =  cv2.imread(join(join(path, 'image-data'), img_name + '-fused.jpg'))
		print("converting image: " + img_name)

		bw_img = cv2.cvtColor(bw_img,cv2.COLOR_BGR2GRAY) #convert to grayscale

		img = b.binarize_image(bw_img)
		l = Line_segmenter()
		seg_images = l.histogram(img)
		l.show_segm_img(seg_images)

		cv2.imwrite(join(path, 'img_out.png'), img)
		print("saved converted image \"" + img_name + "\" to \"" + join(path, 'img_out.png') + "\"")
