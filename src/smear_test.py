import cv2
from os.path import join, abspath
import numpy as np
from binarizer import Binarizer

SMEAR_DENSITY = 0.90
NR_SMEARS = 4
SIZE_THRESHOLD = 900 #in pixels
AREA_THRESHOLD = 0.01 #area the blob has to be at least in order to get added to the set lines (percentage of the biggest blob)
EPSILON_DELTA = 0.001 #modifier for the epsilon value used for determining how tightly the contour must fit 
AREA_THRESHOLD_2 = 10000 #pixel threshold

class Smear:
	def __init__(self):
		pass

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

	def get_contour_approximations(self, img):
		img = np.array(img, dtype=np.uint8)
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

	def get_croppings(self, img_original, img_smear):
		padding  = 40 #pad n pixels in all directions
		img_original = cv2.copyMakeBorder(img_original, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255,255,255])
		img_smear = cv2.copyMakeBorder(img_smear, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255,255,255])
		(approx, rects) = self.get_contour_approximations(img_smear)

		croppings = []
		for idx in range(len(approx)):
			mask = np.zeros_like(img_smear) # Create mask where white is what we want, black otherwise
			cv2.drawContours(mask, approx, idx, 255, -1) # Draw filled contour in mask
			out = np.full_like(img_smear, 255, dtype=np.uint8) #create a white canvas with the same size as the input image
			out[mask == 255] = img_original[mask == 255]

			(x, y, w, h) = rects[idx]
			out = out[y:(y+h), x:(x+w)]
			if w * h > AREA_THRESHOLD_2:
				#quick hack for double lines, improve later edit:doesnt work
				# if h > 180:
				# 	print('splitting cropping')

				# 	cut = h // 2
				# 	croppings.append(out[:cut, :])
				# 	croppings.append(out[cut:, :])
				# 	cv2.imshow('kept', out[:cut, :])
				# 	cv2.waitKey(0)
				# 	cv2.imshow('kept', out[cut:, :])
				# 	cv2.waitKey(0)
				# else:
				croppings.append(out)
				print("keeping item of size " + str(w*h) + "h: " + str(h))
				cv2.imshow('kept', out)
				cv2.waitKey(0)
			else:
				print("removing item of size " + str(w*h))
				# cv2.imshow('removed', out)
				# cv2.waitKey(0)

		return croppings


if __name__ == '__main__':

	b = Binarizer()
	s = Smear()

	# Test on actual dead sea scroll image
	path = join(abspath('..'), 'data')
	nice_img_name = 'P583-Fg006-R-C01-R01'
	bad_img_name = 'P21-Fg006-R-C01-R01'
	img_name = nice_img_name

	col_img = cv2.imread(join(join(path, 'image-data'), img_name + '.jpg'))
	bw_img =  cv2.imread(join(join(path, 'image-data'), img_name + '-fused.jpg'))
	print("converting image: " + img_name)

	bw_img = cv2.cvtColor(bw_img,cv2.COLOR_BGR2GRAY) #convert to grayscale

	img = b.binarize_image(bw_img)

	print("done binarizing")

	# #remove noise
	# img = np.array(img, dtype=np.uint8)
	# rects = b.get_connected_components(img)
	# for (x, y, w, h) in rects:
	# 	if w * h < SIZE_THRESHOLD:
	# 		print('removing item of size ' + str(w * h))
	# 		# cv2.imshow('item ' +  str(w*h),  img[y:(y+h), x:(x+w)])
	# 		cv2.waitKey(0)
	# 		for yidx in range(y, y+h, 1):
	# 			for xidx in range(x, x+w, 1):
	# 				img[y][x] = 255
	# print("done removing noise")


	#image has to be flipped in order for the smearing to work
	img_smear = np.array(img, dtype=np.uint8)
	img_original = np.array(img, dtype=np.uint8)
	img_smear = b.get_negative(img_smear)
	print('start smearing')
	img_smear = s.smear(img_smear)
	print('finished smearing')
	img_smear = b.get_negative(img_smear)

	croppings = s.get_croppings(img_original, img_smear)

	for c in croppings:
		cv2.imshow('crop', c)
		cv2.waitKey(0)
		
	cv2.imwrite(join(path, 'img_smear2.png'), img)
	print("saved converted image \"" + img_name + "\" to \"" + join(path, 'img_smear2.png') + "\"")
