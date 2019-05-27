'''
Applies acid drop to a binarized image with a histogram
'''

import cv2
from os.path import join, abspath
import os
import numpy as np
import skimage
import matplotlib.pyplot as plt
from heapq import heappush, heappop

from binarizer import Binarizer
from smear_test import Smear
from line_segmenter import Line_segmenter

PATH = join(abspath('..'), 'data')
nice_img_name = 'P632-Fg002-R-C01-R01'#'P583-Fg006-R-C01-R01' 
bad_img_name = 'P21-Fg006-R-C01-R01'
ground_truth_img_name = '124-Fg004'

def acid_drop(img, x, y, tx, ty, max_len):
	(maxy, maxx) = np.shape(img)
	heap = []
	visitedmap = np.zeros((maxy, maxx))
	navmap = np.zeros((maxy, maxx, 2), dtype=np.uint16)
	heappush(heap, (0, x, y))
	target_found = False
	sx = x #startvalues for x and y
	sy = y

	targetv = np.array((tx, ty)) #used for dist calc
	startv = np.array((sx, sy)) #used for dist calc

	while not target_found and len(heap) > 0:
		dist, x, y = heappop(heap)
		
		# print("(%d, %d)" % (x, y))
		if x == tx and y == ty:
			target_found = True
		elif dist == max_len:
			print("max path length (%d) reached.." % max_len)
			return []
		else:
			#enqueue all directions
			if x+1 < maxx and img[y, x+1] == 255 and visitedmap[y, x+1] == 0:
				dist = np.linalg.norm(np.array((x+1, y)) - startv)
				heappush(heap, (dist, x+1, y))
				visitedmap[y, x+1] = 1
				navmap[y, x+1, 0] = x
				navmap[y, x+1, 1] = y
			if x-1 > -1 and img[y, x-1] == 255 and visitedmap[y, x-1] == 0:
				dist = np.linalg.norm(np.array((x-1, y)) - startv)
				heappush(heap, (dist, x-1, y))
				visitedmap[y,x-1] = 1
				navmap[y, x-1, 0] = x
				navmap[y, x-1, 1] = y
			if y+1 < maxy and img[y+1, x] == 255 and visitedmap[y+1, x] == 0:
				dist = np.linalg.norm(np.array((x, y+1)) - startv)
				heappush(heap, (dist, x, y+1))
				visitedmap[y+1, x] = 1
				navmap[y+1, x, 0] = x
				navmap[y+1, x, 1] = y

			if y-1 > -1 and img[y-1, x] == 255 and visitedmap[y-1, x] == 0:
				dist = np.linalg.norm(np.array((x, y-1)) - startv)
				heappush(heap, (dist, x, y-1))
				visitedmap[y-1, x] = 1
				navmap[y-1, x, 0] = x
				navmap[y-1, x, 1] = y

	if target_found:
		print("target found!")
		# print(navmap)
		# print(visitedmap)
		line = []
		print(np.shape(navmap))
		while x != sx or y != sy:
			# print("adding to line", x, y)
			line.append((x, y))
			oldx = x
			oldy = y
			x = navmap[oldy, oldx, 0]
			y = navmap[oldy, oldx, 1]
		line.append((sx, sy))
		# print("returning line")
		return line#np.array(reversed(line))

	print("could not construct a path..")
	return []



if __name__ == '__main__':
	b = Binarizer()
	s = Smear()
	l = Line_segmenter()

	img_name = ground_truth_img_name

	bw_img =  cv2.imread(join(join(PATH, 'image-data'), img_name + '-fused.jpg'))
	print("converting image: " + img_name)

	bw_img = cv2.cvtColor(bw_img,cv2.COLOR_BGR2GRAY) #convert to grayscale

	img = b.binarize_image(bw_img)

	(_, smear, _) = s.split_into_lines_and_contour(img)
	segments, hist = l.histogram(smear)

	minima = l.get_minima(hist)

	## actually apply acid drop
	(_, maxx) = np.shape(img)

	for m in minima:
		print("working on minima.. %d" % (m))
		# line = find_path(img, m)
		line = acid_drop(img, 0, m, maxx-1, m, 9000)
		for (x, y) in line:
			print("(%d,%d)" % (x, y))
		line = np.array(line)
		# pts = line.reshape((-1,1,2))
		img = cv2.polylines(img,[line],False,(125))


	cv2.imwrite('a-star.png', img)