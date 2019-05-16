'''
The only function from this file that is needed in the complete pipeline is the function crop_image. 
All other functions are helper functions for this function and don't need to be called explicitly in the pipeline. 

'''


import cv2
from os.path import join, abspath
import numpy as np
from binarizer import Binarizer

from copy import deepcopy

SMEAR_DENSITY = 0.90
NR_SMEARS = 4
SIZE_THRESHOLD = 900 #in pixels
AREA_THRESHOLD = 0.01 #area the blob has to be at least in order to get added to the set lines (percentage of the biggest blob)
EPSILON_DELTA = 0.0001 #modifier for the epsilon value used for determining how tightly the contour must fit 
AREA_THRESHOLD_2 = 8000 #pixel threshold
PADDING = 100 #padding around the smeared image in pixels

class Smear:
	def __init__(self):
		self.b = Binarizer()

	def smear(self, img):
		(y_max, x_max) = np.shape(img)

		for idx in range(NR_SMEARS):
			if idx % 2 == 0: #smear from left to right
				for y in range(y_max):
					for x in range(5, x_max, 1):
						val = img[y, x] + SMEAR_DENSITY * img[y, x-1]
						img[y, x] = min(255, val)
				# img = [img[y, x] + SMEAR_DENSITY * img[y, x-1] for x in range(1, x_max, 1) for y in range(y_max)]

			if idx % 2 == 1: #smear from right to left
				for y in range(y_max):
					for x in range(x_max-1, 0, -1):
						val = img[y, x-1] + SMEAR_DENSITY * img[y, x]
						img[y, x-1] = min(255, val)


		return img

	def get_contoured_image(self, img_smear, img_original):
		'''
		Used for extracting the contoured image.. Mostly for testing
		'''
		img_original = self.padd_image(img_original, PADDING)
		img_smear = self.padd_image(img_smear, PADDING)

		# img_smear = b.dilate(img_smear, 25)
		# img_smear = b.erode(img_smear, 20)

		img_smear = self.b.binarize_simple(img_smear, 10)

		# cv2.imwrite("dilated.png", img_smear)

		(approx, rects) = self.get_contour_approximations(img_smear)
		img_copy = deepcopy(img_original)

		for a in approx:
			cv2.drawContours(img_copy,[a],0,(90,0,255),2)

		return img_copy

	def get_contour_approximations(self, img):
		img = np.array(img, dtype=np.uint8)
		contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		areas = [cv2.contourArea(cnt) for cnt in contours]
		areas.remove(max(areas)) #remove biggest blob; the entire image is always parsed as a blob
		thres =  AREA_THRESHOLD * max(areas)

		contours = [cnt for (cnt, area) in zip(contours, areas) if area > thres]

		removed = len(areas) - len(contours)
		print('removed ' + str(removed) + ' areas')

		approx = []
		for cnt in contours:
			epsilon = EPSILON_DELTA * cv2.arcLength(cnt,True)
			approx.append(cv2.approxPolyDP(cnt,epsilon,True))

		rects = [cv2.boundingRect(ctr) for ctr in contours]

		return (approx, rects)

	def padd_image(self, img, padding):
		return cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255,255,255])

	def get_croppings(self, img_original, img_smear):
		img_original = self.padd_image(img_original, PADDING)
		img_smear = self.padd_image(img_smear, PADDING)

		# img_smear = self.b.dilate(img_smear, 25)
		# img_smear = self.b.erode(img_smear, 20)

		# img_smear = self.b.binarize_simple(img_smear, 10)

		cv2.imwrite("dilated.png", img_smear)

		(approx, rects) = self.get_contour_approximations(img_smear)

		if True: ### USED FOR TESTING
			img_copy = deepcopy(img_original)

			for a in approx:
				cv2.drawContours(img_copy,[a],0,(90,0,255),2)

			# cv2.imwrite("contoured.png", img_copy)

		croppings = []
		for idx in range(len(approx)):
			mask = np.zeros_like(img_smear) # Create mask where white is what we want, black otherwise
			cv2.drawContours(mask, approx, idx, 255, -1) # Draw filled contour in mask
			out = np.full_like(img_smear, 255, dtype=np.uint8) #create a white canvas with the same size as the input image
			out[mask == 255] = img_original[mask == 255]

			(x, y, w, h) = rects[idx]
			out = out[y:(y+h), x:(x+w)]
			if w * h > AREA_THRESHOLD_2:
				(h_img, w_img) = np.shape(img_original)
				if w * h < h_img * w_img:
					croppings.append(out)
					# print("keeping item of size " + str(w*h) + "h: " + str(h))
					# cv2.imshow('kept', out)
					# cv2.waitKey(0)
			# 	else:
			# 		print("cropping was of size equal to image.. skipping")
			# else:
			# 	print("removing item of size " + str(w*h))
			# 	cv2.imshow('removed', out)
			# 	cv2.waitKey(0)

		return croppings

	def split_into_lines(self, img):
		img_smear = np.array(img, dtype=np.uint8) #copy image
		img_smear = self.b.get_negative(img_smear)
		print('start smearing')
		img_smear = self.smear(img_smear)
		print('finished smearing')
		img_smear = self.b.get_negative(img_smear)

		croppings = self.get_croppings(img, img_smear)

		return croppings

	def split_into_lines_and_contour(self, img):
		'''
		Extended split_into_lines.
		Also returns the contoured image
		For testing
		'''
		img_smear = np.array(img, dtype=np.uint8) #copy image
		img_smear = self.b.get_negative(img_smear)
		print('start smearing')
		img_smear = self.smear(img_smear)
		print('finished smearing')
		img_smear = self.b.get_negative(img_smear)

		croppings = self.get_croppings(img, img_smear)
		img_contoured = self.get_contoured_image(img_smear, img)

		return (img_contoured, img_smear, croppings)




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
	img_name = nice_img_name

	bw_img =  cv2.imread(join(join(path, 'image-data'), img_name + '-fused.jpg'))
	print("converting image: " + img_name)

	bw_img = cv2.cvtColor(bw_img,cv2.COLOR_BGR2GRAY) #convert to grayscale

	img = b.binarize_image(bw_img)
	# img2 = deepcopy(img)
	# img = b.get_negative(img)
	# smeared = s.smear(img)
	# smeared = b.get_negative(smeared)
	# cv2.imwrite("smeared.png", smeared)
	# cv2.imshow('smeared', smeared)
	# cv2.waitKey(0)

	# print("done binarizing")
	
	croppings = s.split_into_lines(img)

	for idx, c in enumerate(croppings):
		# cv2.imshow('crop', c)
		cv2.imwrite("cropping_%d.png" % (idx), c)
		# cv2.waitKey(0)
		
