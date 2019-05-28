'''
Applies acid drop to a binarized image with a histogram
'''

import cv2
from os.path import join, abspath
import os
import sys
import numpy as np
import skimage
import matplotlib.pyplot as plt
from heapq import heappush, heappop

import time

from binarizer import Binarizer
from smear_test import Smear
from line_segmenter import Line_segmenter

PATH = join(abspath('..'), 'data')
nice_img_name = 'P632-Fg002-R-C01-R01'#'P583-Fg006-R-C01-R01' 
bad_img_name = 'P21-Fg006-R-C01-R01'
ground_truth_img_name = '124-Fg004'

BURNCOST = 4.5 		#penalty to burn through a single black pixel (this is added to other possible penalties)
VCOST1 = 2.0 		#penalty to move in the vertical direction away from the target y
VCOST2 = 0.5		#penalty to move in the vertical direction toward the target y
LCOST = 1.5			#penalty to move to the left (back)
RCOST = 1.0			#penalty to move right (forward; this is the preferred action)

# VCOST = 1.0 		#multiplication penalty (test)


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
		
		if x == tx and y == ty:
			target_found = True
		elif dist == max_len:
			print("max path length (%d) reached.." % max_len)
			return []
		else:
			#enqueue all directions
			if x+1 < maxx and visitedmap[y, x+1] == 0:
				# dist = np.linalg.norm(np.array((x+1, y)) - startv)
				rdist = dist + RCOST
				if img[y, x+1] != 255:
					rdist += BURNCOST
				heappush(heap, (rdist, x+1, y))
				visitedmap[y, x+1] = 1
				navmap[y, x+1, 0] = x
				navmap[y, x+1, 1] = y
			if x-1 > -1 and visitedmap[y, x-1] == 0:
				# dist = np.linalg.norm(np.array((x-1, y)) - startv)
				ldist = dist + LCOST
				if img[y, x-1] != 255:
					ldist += BURNCOST
				heappush(heap, (ldist, x-1, y))
				visitedmap[y,x-1] = 1
				navmap[y, x-1, 0] = x
				navmap[y, x-1, 1] = y
			if y+1 < maxy and visitedmap[y+1, x] == 0:
				# dist = np.linalg.norm(np.array((x, y+1)) - startv)
				if ((y+1) - ty)*((y+1) - ty) > (y - ty)*(y - ty): #diverging from the target y
					udist = dist + VCOST1
				else:	#coming towards the target y
					udist = dist + VCOST2
				# udist = dist + VCOST * np.abs((y+1) - ty) / 20.0
				if img[y+1, x] != 255:
					udist += BURNCOST
				heappush(heap, (udist, x, y+1))
				visitedmap[y+1, x] = 1
				navmap[y+1, x, 0] = x
				navmap[y+1, x, 1] = y

			if y-1 > -1 and visitedmap[y-1, x] == 0:
				# dist = np.linalg.norm(np.array((x, y-1)) - startv)
				if ((y-1) - ty)*((y-1) - ty) > (y - ty)*(y - ty): #diverging from the target y
					ddist = dist + VCOST1
				else:	#coming towards the target y
					ddist = dist + VCOST2
				# ddist = dist + VCOST * np.abs((y-1) - ty) / 20.0
				if img[y-1, x] != 255:
					ddist += BURNCOST
				heappush(heap, (ddist, x, y-1))
				visitedmap[y-1, x] = 1
				navmap[y-1, x, 0] = x
				navmap[y-1, x, 1] = y

	if target_found:
		print("target found!")
		line = []
		while x != sx or y != sy:
			line.append((x, y))
			oldx = x
			oldy = y
			x = navmap[oldy, oldx, 0]
			y = navmap[oldy, oldx, 1]
		line.append((sx, sy))
		return line

	print("could not construct a path.. (heap empty")
	return []



if __name__ == '__main__':
	b = Binarizer()
	s = Smear()
	l = Line_segmenter()



	img_name = ground_truth_img_name

	files =  [join(join(PATH, 'image-data'), f) for f in os.listdir(join(PATH, 'image-data')) if os.path.isfile(join(join(PATH, 'image-data'),  f)) and f.endswith('-fused.jpg')]

	idx = 0
	# bw_img =  cv2.imread(join(join(PATH, 'image-data'), img_name + '-fused.jpg'))
	for f in files:
		bw_img =  cv2.imread(f)
		print("converting image: " + f)

		bw_img = cv2.cvtColor(bw_img,cv2.COLOR_BGR2GRAY) #convert to grayscale

		t = time.time()

		img = b.binarize_image(bw_img)

		print("time passed in binarizer: ", time.time() - t)
		t2 = time.time()

		(_, smear, _) = s.split_into_lines_and_contour(img)
		segments, hist = l.histogram(smear)
		print("time passed in smear: ", time.time() - t2)
		print("time passed in prepro: ", time.time() - t)

		minima = l.get_minima(hist)

		## actually apply acid drop
		(_, maxx) = np.shape(img)

		for m in minima:
			print("working on minima.. %d" % (m))
			sys.stdout.flush()
			# line = find_path(img, m)
			line = acid_drop(img, 0, m, maxx-1, m, 9000)
			# for (x, y) in line:
			# 	print("(%d,%d)" % (x, y))
			line = np.array(line)
			# pts = line.reshape((-1,1,2))
			img = cv2.polylines(img,[line],False,(125))


		cv2.imwrite('a-star%d.png' % (idx), img)
		print("saved image to a-star.png")
		idx += 1