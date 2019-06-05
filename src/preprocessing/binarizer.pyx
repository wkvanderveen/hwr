'''
binarizer.py

This file contains the code used for binarizing the input image
'''
import cython
import cv2
from os.path import join, abspath
import numpy as np
from skimage.filters import threshold_otsu, threshold_sauvola
cimport numpy as np

class Binarizer:
	def __init__(self):
		pass

	def get_negative(self, np.ndarray[np.uint8_t, ndim=2] img):
		cdef int x, y, x_max, y_max
		(y_max, x_max) = np.shape(img)

		for y in range(y_max):
			for x in range(x_max):
				img[y, x] = 255 - img[y, x]
		return img

	def binarize_otsu(self, np.ndarray[np.uint8_t, ndim=2] img):
		cdef int x, y, x_max, y_max
		cdef double thres = threshold_otsu(img) #create otsu threshold
		(y_max, x_max) = np.shape(img)

		for y in range(y_max):
			for x in range(x_max):
				img[y, x] = 255 if img[y, x] > thres else 0
		return img

	def binarize_simple(self, np.ndarray[np.uint8_t, ndim=2] img, thres = 100):
		cdef int x, y, x_max, y_max
		(y_max, x_max) = np.shape(img)

		for y in range(y_max):
			for x in range(x_max):
				img[y, x] = 255 if img[y, x] > thres else 0
		return img

	def binarize_image(self,  imgin):
		cdef int x, y, w, h
		cdef np.ndarray[np.uint8_t, ndim=2] img
		'''
		This function contains the (currently) optimal binarization pipeline for the images
		Can handle both bw and color images
		'''
		cdef tuple s = np.shape(imgin)
		if len(s) > 2:
			if s[2] > 1: #check if the image is in RGB or grayscale
				img = cv2.cvtColor(imgin,cv2.COLOR_BGR2GRAY) #convert to grayscale
		else:
			img = imgin



		 

		cdef np.ndarray[np.uint8_t, ndim=2] mask = self.get_mask(img)

		#give the mask here, as the mask over the text is the biggest connected component.
		cdef list rects = self.get_connected_components(mask)
		(x, y, w, h) = rects[0]

		img = img[y:y+h, x:x+w]
		mask = mask[y:y+h, x:x+w]

		#mask image
		# cv2.imwrite('img_in.png', img)
		img = self.apply_mask(img, mask)
		# cv2.imwrite('masked_new.png', img)
		# cv2.imwrite('mask.png', mask)

		#remove noise
		# img = np.array(img, dtype=np.uint8)
		img = cv2.medianBlur(img, 7)

		# cv2.imwrite('cleaned_new.png', img)
		# img = np.array(img, dtype=np.uint8)
		return img

	def erode(self, img, se_size = 5):
		se = np.ones((se_size, se_size), np.uint8)
		return cv2.erode(img, se, iterations = 1)

	def dilate(self, img, se_size = 5):
		se = np.ones((se_size, se_size), np.uint8)
		return cv2.dilate(img, se, iterations = 1)


	def get_connected_components(self, np.ndarray[np.uint8_t, ndim=2] img):
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		sorted_ctrs = sorted(contours, key = cv2.contourArea, reverse = True)

		return [cv2.boundingRect(ctr) for ctr in sorted_ctrs]


	def get_mask(self, np.ndarray[np.uint8_t, ndim=2] img):
		cdef np.ndarray[np.uint8_t, ndim=2] mask = self.dilate(img, 20)
		mask = self.erode(mask, 20)

		mask = self.binarize_otsu(mask)
		return mask

	def apply_mask(self, np.ndarray[np.uint8_t, ndim=2] img, np.ndarray[np.uint8_t, ndim=2] mask):
		cdef int x, y, x_max, y_max, mask_thres, img_thres
		'''
		Creates a mask from the image to segement out the backgroud. 
		Uses Otsu to create thresholds to use when the mask is applied.
		Outputs a binarized image.
		Works best with bw images (not binarized)
		'''

		(y_max, x_max) = np.shape(img)
		
		mask_thres = 100 # (mask is already binarized, so an arbitrary number works as a threshold)
		img_out = np.zeros((y_max, x_max), dtype=np.uint8) #alloc memory

		img_thres = threshold_otsu(img) #create global otsu threshold (sauvola was tried, but gave worse results)
		for y in range(y_max):
			for x in range(x_max):
				img_out[y, x] = 0 if (img[y, x] < img_thres and mask[y, x] > mask_thres) else 255

		return img_out

if __name__ == '__main__':

	b = Binarizer()

	# Test on actual dead sea scroll image
	path = join(abspath('..'), 'data')
	img_name = 'P21-Fg006-R-C01-R01'#'P583-Fg006-R-C01-R01'#'P22-Fg008-R-C01-R01' #'P513-Fg001-R-C01-R01' 'P106-Fg002-R-C01-R01' 'P21-Fg006-R-C01-R01.jpg';
	col_img = cv2.imread(join(join(path, 'image-data'), img_name + '.jpg'))
	bw_img =  cv2.imread(join(join(path, 'image-data'), img_name + '-fused.jpg'))
	print("converting image: " + img_name)

	bw_img = cv2.cvtColor(bw_img,cv2.COLOR_BGR2GRAY) #convert to grayscale

	img = b.binarize_image(bw_img)

	cv2.imshow('img', img)
	cv2.waitKey(0)
	cv2.imwrite(join(path, 'img_out.png'), img)
	print("saved converted image \"" + img_name + "\" to \"" + join(path, 'img_out.png') + "\"")