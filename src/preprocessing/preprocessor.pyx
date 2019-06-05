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

cdef float MAX_CROPPING_HEIGHT = 150 #in px

def preprocess_image(np.ndarray[np.uint8_t, ndim=3] imgin):
	'''
	Takes an image as its input and returns a list of croppings using the full preprocessing pipeline
	'''

	#type declarations
	cdef int x, y, xmax, ymax
	cdef np.ndarray[np.uint8_t, ndim=2] img, c, s, out
	cdef np.ndarray[np.uint64_t, ndim=1] hist
	cdef list croppings, smear_croppings, final_croppings
	cdef dict linedict, linedict_old

	b = Binarizer()
	sm = Smear()
	l = Line_segmenter()
	a = Acid_drop()


	#binarize
	img = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY) #convert to grayscale
	img = b.binarize_image(img)

	#smear
	(_, _, croppings, smear_croppings) = sm.split_into_lines_and_contour(img)

	croppings.reverse() #order croppings from top to bot
	smear_croppings.reverse()

	final_croppings = []
	for (c, s) in zip(croppings, smear_croppings):
		(height, width) = np.shape(c)
		if height > MAX_CROPPING_HEIGHT: 
			#split further using acid drop
			hist = l.create_v_histogram(s)

			# plt.plot(hist)
			# plt.show()

			# cv2.imshow("c", contoured)
			# cv2.waitKey(0)

			minima = l.get_minima(hist)

			if len(minima) == 0: 
				#cropping can't be cropped any further
				final_croppings.append(c)
			else: 				
				#crop further using acid drop
				linedict = {}
				(_, xmax) = np.shape(c)
				for x in range(xmax):
					linedict[x] = 0

				(ymax, xmax) = np.shape(c)

				for m in minima:
					linedict_old = deepcopy(linedict)
					line = a.acid_drop(c, 0, m, xmax-1, m, 9000)

					#put line in dict for easier accessing in next loop
					linedict = {}

					for (x, y) in line:
						linedict[x] = y

					out = np.full_like(c, 255, dtype=np.uint8) 
					

					for y in range(ymax):
						for x in range(xmax):
							if y <= linedict[x] and y >= linedict_old[x]:
								out[y, x] = c[y, x] #copy the pixel from the original croppings

					print(min(linedict_old.values()), max(linedict.values()))

					out = out[min(linedict_old.values()):max(linedict.values()), :] #crop vertically

					final_croppings.append(out)

				#add final cropping (last line to bot of image)
				out = np.full_like(c, 255, dtype=np.uint8) 
				for y in range(ymax):
					for x in range(xmax):
						if y >= linedict[x]:
							out[y, x] = c[y, x] #copy the pixel from the original croppings
				out = out[min(linedict.values()): , :] #crop vertically
				final_croppings.append(out)


		else:
			#the cropping is probably good
			final_croppings.append(c)

	return (img, final_croppings) ##returns image for testing