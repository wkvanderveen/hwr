'''
The only function from this file that is needed in the complete pipeline is the function crop_image. 
All other functions are helper functions for this function and don't need to be called explicitly in the pipeline. 

'''

import cython
import cv2
from os.path import join, abspath
import numpy as np
from binarizer import Binarizer
from copy import deepcopy

cimport numpy as np

cdef float SMEAR_DENSITY = 0.90
cdef int NR_SMEARS = 4
cdef int SIZE_THRESHOLD = 900 #in pixels
cdef float AREA_THRESHOLD = 0.01 #area the blob has to be at least in order to get added to the set lines (percentage of the biggest blob)
cdef int AREA_THRESHOLD_2 = 8000 #pixel threshold
cdef double EPSILON_DELTA = 0.0001 #modifier for the epsilon value used for determining how tightly the contour must fit 
cdef int PADDING = 100 #padding around the smeared image in pixels


cdef np.ndarray[np.uint8_t, ndim=2] smear(np.ndarray[np.uint8_t, ndim=2] img):
	cdef int idx, y_max, x_max, x, y, val
	(y_max, x_max) = np.shape(img)


	
	for idx in range(NR_SMEARS):
		if idx % 2 == 0: #smear from left to right
			for y in range(y_max):
				for x in range(5, x_max, 1):
					val = img[y, x] + <int>(SMEAR_DENSITY * img[y, x-1])
					img[y, x] = min(255, val)
					
		else: #smear from right to left
			for y in range(y_max):
				for x in range(x_max-1, 0, -1):
					val = img[y, x-1] + <int>(SMEAR_DENSITY * img[y, x])
					img[y, x-1] = min(255, val)



	return img




class Smear:
	def __init__(self):
		self.b = Binarizer()

	def get_contoured_image(self, np.ndarray[np.uint8_t, ndim=2] img_smear, np.ndarray[np.uint8_t, ndim=2] img_original):
		'''
		Used for extracting the contoured image.. Mostly for testing
		'''
		# typedefs
		cdef list approx, rects
		cdef np.ndarray a
		cdef np.ndarray[np.uint8_t, ndim=2] img_copy

		img_original = self.padd_image(img_original, PADDING)
		img_smear = self.padd_image(img_smear, PADDING)

		img_smear = self.b.binarize_simple(img_smear, 10)

		(approx, rects) = self.get_contour_approximations(img_smear)
		img_copy = deepcopy(img_original)

		for a in approx:
			cv2.drawContours(img_copy,[a],0,(90,0,255),2)

		return img_copy

	def get_contour_approximations(self, np.ndarray[np.uint8_t, ndim=2] img):
		img = np.array(img, dtype=np.uint8)
		contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		areas = [cv2.contourArea(cnt) for cnt in contours]
		areas.remove(max(areas)) #remove biggest blob; the entire image is always parsed as a blob
		thres =  AREA_THRESHOLD * max(areas)

		contours = [cnt for (cnt, area) in zip(contours, areas) if area > thres]

		removed = len(areas) - len(contours)
		# print('removed ' + str(removed) + ' areas')

		approx = []
		for cnt in contours:
			epsilon = EPSILON_DELTA * cv2.arcLength(cnt,True)
			approx.append(cv2.approxPolyDP(cnt,epsilon,True))

		rects = [cv2.boundingRect(ctr) for ctr in contours]

		return (approx, rects)

	def padd_image(self, np.ndarray[np.uint8_t, ndim=2] img, int padding):
		return cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255,255,255])

	def get_croppings(self, np.ndarray[np.uint8_t, ndim=2] img_original, np.ndarray[np.uint8_t, ndim=2] img_smear):
		cdef int x, y, w, h, idx, h_img, w_img
		img_original = self.padd_image(img_original, PADDING)
		img_smear = self.padd_image(img_smear, PADDING)

		# cv2.imwrite("dilated.png", img_smear)

		(approx, rects) = self.get_contour_approximations(img_smear)

		cdef list croppings = []
		cdef list smear_croppings = []
		for idx in range(len(approx)):
			mask = np.zeros_like(img_smear) # Create mask where white is what we want, black otherwise
			cv2.drawContours(mask, approx, idx, 255, -1) # Draw filled contour in mask
			out = np.full_like(img_smear, 255, dtype=np.uint8) #create a white canvas with the same size as the input image
			out[mask == 255] = img_original[mask == 255]

			#create a smeared cropping for acid drop segmentation
			cout = np.full_like(img_smear, 255, dtype=np.uint8) 
			cout[mask == 255] = img_smear[mask == 255]


			(x, y, w, h) = rects[idx]
			out = out[y:(y+h), x:(x+w)]
			cout = cout[y:(y+h), x:(x+w)]
			# if w * h > AREA_THRESHOLD_2:
			# 	(h_img, w_img) = np.shape(img_original)
			# 	if w * h < h_img * w_img:
			croppings.append(out)
			smear_croppings.append(cout)



		return (croppings, smear_croppings)

	def split_into_lines(self, np.ndarray[np.uint8_t, ndim=2] img):
		img_smear = np.array(img, dtype=np.uint8) #copy image
		img_smear = self.b.get_negative(img_smear)
		# print('start smearing')
		img_smear = smear(img_smear)
		# print('finished smearing')
		img_smear = self.b.get_negative(img_smear)

		(croppings, _) = self.get_croppings(img, img_smear)

		return croppings

	def split_into_lines_and_contour(self, np.ndarray[np.uint8_t, ndim=2] img):
		'''
		Extended split_into_lines.
		Also returns the contoured image
		For testing
		'''

		cdef np.ndarray[np.uint8_t, ndim=2] img_contoured, img_smear
		cdef list croppings

		img_smear = np.array(img, dtype=np.uint8) #copy image
		img_smear = self.b.get_negative(img_smear)
		# print('start smearing')
		img_smear = smear(img_smear)
		# print('finished smearing')
		img_smear = self.b.get_negative(img_smear)

		(croppings, smear_croppings) = self.get_croppings(img, img_smear)
		img_contoured = self.get_contoured_image(img_smear, img)

		return (img_contoured, img_smear, croppings, smear_croppings)




if __name__ == '__main__':
	'''
	The main of this file is solely used for unit testing the Smear class above
	
	'''

	b = Binarizer()
	s = Smear()

	# Test on actual dead sea scroll image
	path = join(abspath('..'), 'data')
	nice_img_name = 'P632-Fg002-R-C01-R01'#'P583-Fg006-R-C01-R01' 
	bad_img_name = 'P21-Fg006-R-C01-R01'
	ground_truth_test_name = '124-Fg004'
	img_name = ground_truth_test_name

	bw_img =  cv2.imread(join(join(path, 'image-data'), img_name + '-fused.jpg'))
	print("converting image: " + img_name)

	bw_img = cv2.cvtColor(bw_img,cv2.COLOR_BGR2GRAY) #convert to grayscale

	img = b.binarize_image(bw_img)
	
	(contoured_img, smear_img, croppings) = s.split_into_lines_and_contour(img)
	cv2.imwrite("contoured.jpg", contoured_img)
	cv2.imwrite("smeared.jpg", smear_img)

	for idx, c in enumerate(croppings):
		cv2.imwrite("cropping_%d.png" % (idx), c)

	print("saved %d croppings" % (len(croppings)))
