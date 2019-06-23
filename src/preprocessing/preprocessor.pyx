from binarizer import Binarizer
from smear_test import Smear
from line_segmenter import Line_segmenter
from acid_drop import Acid_drop

import cython
from copy import deepcopy
import cv2
import numpy as np
cimport numpy as np

import matplotlib.pyplot as plt

cdef float MAX_CROPPING_HEIGHT = 130 #in px
cdef int MIN_BLACK_PIXELS = 1200 #minimum number of black pixels in cropping of line to be saved
cdef int MIN_BLACK_PIXELS_CHAR = 400 #minimum number of black pixels in cropping of char to be saved
cdef char APPLY_CHARACTER_SEGMENTATION = 1 ## apply char segmentation after line segmentation

## obtained from https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py
def rotate_bound(np.ndarray[np.uint8_t, ndim=2] image, float angle):
	# grab the dimensions of the image and then determine the
	# center
	(h, w) = image.shape[:2]
	(cX, cY) = (w / 2, h / 2)

	# grab the rotation matrix (applying the negative of the
	# angle to rotate clockwise), then grab the sine and cosine
	# (i.e., the rotation components of the matrix)
	M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	# compute the new bounding dimensions of the image
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))

	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	# perform the actual rotation and return the image
	return cv2.warpAffine(image, M, (nW, nH))


def sum_black_pixels(np.ndarray[np.uint8_t, ndim=2] img):
	# return np.sum(img[img == 0])
	return img[img == 0].size


def preprocess_image(np.ndarray[np.uint8_t, ndim=3] imgin):
	'''
	Takes an image as its input and returns a list of croppings using the full preprocessing pipeline
	'''

	#type declarations for cython
	cdef int x, y, xmax, ymax
	cdef np.ndarray[np.uint8_t, ndim=2] img, c, c2, s, out
	cdef np.ndarray[np.uint64_t, ndim=1] hist
	cdef list croppings, smear_croppings, final_croppings
	cdef dict linedict, linedict_old

	cdef list char_croppings = [], char_croppings_final = []

	#instantiate used classes
	b = Binarizer()
	sm = Smear()
	l = Line_segmenter()
	a = Acid_drop()


	#binarize
	img = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY) #convert to grayscale
	img = b.binarize_image(img)

	#smear
	(_, smear, croppings, smear_croppings) = sm.split_into_lines_and_contour(img)

	croppings.reverse() #order croppings from top to bot
	smear_croppings.reverse()

	final_croppings = []
	# for (c, s) in zip(croppings, smear_croppings):
	c = img
	s = smear
	if True:
		(height, width) = np.shape(s)
		if height > MAX_CROPPING_HEIGHT: 
			#split further using acid drop
			hist = l.create_v_histogram(s)

			minima = l.get_minima(hist)

			if len(minima) == 0: 
				#cropping can't be cropped any further
				if sum_black_pixels(c) > MIN_BLACK_PIXELS:
					final_croppings.append(c)
			else: 				
				#crop further using acid drop

				#initialize linedict with zeros (index of top row of image)
				linedict = {}
				(ymax, xmax) = np.shape(c)
				for x in range(xmax):
					linedict[x] = 0

				for m in minima:
					linedict_old = deepcopy(linedict)
					line = a.acid_drop(c, 0, m, xmax-1, m, 9000)

					linedict = {}

					for (x, y) in line:
						linedict[x] = y

					out = np.full_like(c, 255, dtype=np.uint8) 
					

					for y in range(ymax):
						for x in range(xmax):
							if y <= linedict[x] and y >= linedict_old[x]:
								out[y, x] = c[y, x] #copy the pixel from the original croppings

					out = out[min(linedict_old.values()):max(linedict.values()), :] #crop vertically
					if sum_black_pixels(out) > MIN_BLACK_PIXELS:
						final_croppings.append(out)

				#add final cropping (last line to bot of image)
				out = np.full_like(c, 255, dtype=np.uint8) 
				for y in range(ymax):
					for x in range(xmax):
						if y >= linedict[x]:
							out[y, x] = c[y, x] #copy the pixel from the original croppings
				out = out[min(linedict.values()): , :] #crop vertically
				if sum_black_pixels(out) > MIN_BLACK_PIXELS:
					final_croppings.append(out)


		else:
			#the cropping is properly made by the smearer
			if sum_black_pixels(c) > MIN_BLACK_PIXELS:
				final_croppings.append(c)

	## EXPERIMENTAL CHARACTER SEGMENTATION
	if APPLY_CHARACTER_SEGMENTATION:
		for c2 in final_croppings:
			c2 = rotate_bound(c2, 270) # turn sideways
			(_, _, croppings, smear_croppings) = sm.split_into_lines_and_contour(c2)
			for (c, s) in zip(croppings, smear_croppings):
				hist = l.create_v_histogram(s)

				minima = l.get_minima(hist)

				if len(minima) == 0 and c.shape[0] > 0 and c.shape[1] > 0:
						out = rotate_bound(c, 90) ## rotate back into place
						if sum_black_pixels(out) > MIN_BLACK_PIXELS_CHAR:
							char_croppings.append(out)
				else: 				
					#crop further using acid drop

					#initialize linedict with zeros (index of top row of image)
					linedict = {}
					(ymax, xmax) = np.shape(c)
					for x in range(xmax):
						linedict[x] = 0

					for m in minima:
						linedict_old = deepcopy(linedict)
						linedict = {}
						line = a.acid_drop(c, 0, m, xmax-1, m, 9000)

						for (x, y) in line:
							linedict[x] = y
						# for x in range(xmax):
						# 	linedict[x] = m

						out = np.full_like(c, 255, dtype=np.uint8) 
						

						for y in range(ymax):
							for x in range(xmax):
								if y <= linedict[x] and y >= linedict_old[x]:
									out[y, x] = c[y, x] #copy the pixel from the original croppings

						out = out[min(linedict_old.values()):max(linedict.values()), :] #crop vertically
						if out.shape[0] > 0 and out.shape[1] > 0:
							out = rotate_bound(out, 90) ## rotate back into place
							if sum_black_pixels(out) > MIN_BLACK_PIXELS_CHAR:
								char_croppings.append(out)

					#add final cropping (last line to bot of image)
					out = np.full_like(c, 255, dtype=np.uint8)
					for y in range(ymax):
						for x in range(xmax):
							if y >= linedict[x]:
								out[y, x] = c[y, x] #copy the pixel from the original croppings
					out = out[min(linedict.values()): , :] #crop vertically
					if out.shape[0] > 0 and out.shape[1] > 0:
						out = rotate_bound(out, 90) ## rotate back into place
						if sum_black_pixels(out) > MIN_BLACK_PIXELS_CHAR:
							char_croppings.append(out)
			char_croppings_final.append(char_croppings)
			char_croppings = []

		final_croppings = char_croppings_final




	return final_croppings