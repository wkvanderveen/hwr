import cv2
from os.path import join, abspath
import numpy as np
from binarizer import Binarizer

SMEAR_DENSITY = 0.90
NR_SMEARS = 4
SIZE_THRESHOLD = 900 #in pixels

def smear(img):
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



if __name__ == '__main__':

	b = Binarizer()

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
	img = np.array(img, dtype=np.uint8)
	rects = b.get_connected_components(img)
	for (x, y, w, h) in rects:
		if w * h < SIZE_THRESHOLD:
			print('removing item of size ' + str(w * h))
			# cv2.imshow('item ' +  str(w*h),  img[y:(y+h), x:(x+w)])
			cv2.waitKey(0)
			for yidx in range(y, y+h, 1):
				for xidx in range(x, x+w, 1):
					img[y][x] = 255
	print("done connected comps")

	
	# cv2.imshow('img before smear', img)
	# cv2.waitKey(0)
	img = np.array(img, dtype=np.uint8)
	img = b.get_negative(img)
	print('start smearing')
	img = smear(img)
	print('finished smearing')
	img = b.get_negative(img)
	# cv2.imshow('img after smear', img)
	# cv2.waitKey(0)
	cv2.imwrite(join(path, 'img_smear.png'), img)
	print("saved converted image \"" + img_name + "\" to \"" + join(path, 'img_smear.png') + "\"")


	img = np.array(img, dtype=np.uint8)
	rects = b.get_connected_components(img)

	contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	areas = [cv2.contourArea(cnt) for cnt in contours]
	thres =  0.25 * max(areas)

	contours2 = []
	for cnt, area in zip(contours, areas):
		if area > thres:
			contours2.append(cnt)

	contours = contours2

	# contours = [cnt for (cnt, area) in zip(contours, areas) if area > thres]
	removed = len(areas) - len(contours)

	print('removed ' + str(removed) + ' areas')

	for cnt in contours:
		epsilon = 0.001*cv2.arcLength(cnt,True)
		approx = cv2.approxPolyDP(cnt,epsilon,True)
		# if (cv2.isContourConvex(cnt)):
		cv2.drawContours(img,[approx],0,(90,0,255),2)

	# for (x, y, w, h) in rects:
	# 	cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
		
	cv2.imwrite(join(path, 'img_smear2.png'), img)
	print("saved converted image \"" + img_name + "\" to \"" + join(path, 'img_smear2.png') + "\"")

	print("done connected comps 2")
	cv2.imshow('final img', img)
	cv2.waitKey(0)
