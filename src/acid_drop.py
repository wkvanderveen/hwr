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

def get_straight_path(x, y, tx):
	line = []
	while x != tx+1:
		line.append((x, y))
		x += 1
	return line

def find_target(img, x, y):
	x += 1
	(maxx, _) = np.shape(img)
	while x != maxx and img[x, y] != 255: #find first white pixel
		x += 1

	return (x, y)

def acid_drop(img, x, y, max_len):
	(tx, ty) = find_target(img, x, y)
	(maxx, maxy) = np.shape(img)
	heap = []
	visitedmap = np.zeros((maxx, maxy))
	navmap = np.zeros((maxx, maxy, 2))
	heappush(heap, (x, y, 0))
	target_found = False
	sx = x #startvalues for x and y
	sy = y

	while not target_found and len(heap) > 0:
		x, y, dist = heappop(heap)

		if x == tx and y == ty:
			target_found = True
		elif dist == max_len:
			print("max path length (%d) reached.." % max_len)
			return(get_straight_path(sx, sy, tx))
		else:
			#enqueue all directions
			if x+1 < maxx and img[x+1, y] == 255 and visitedmap[x+1, y] == 0:
				heappush(heap, (x+1, y, dist+1))
				visitedmap[x+1, y] = 1
				navmap[x+1, y] = [x, y]
			if x-1 > -1 and img[x-1, y] == 255 and visitedmap[x-1, y] == 0:
				heappush(heap, (x-1, y, dist+1))
				visitedmap[x-1, y] = 1
				navmap[x-1, y] = [x, y]
			if y+1 < maxy and img[x, y+1] == 255 and visitedmap[x, y+1] == 0:
				heappush(heap, (x, y+1, dist+1))
				visitedmap[x, y+1] = 1
				navmap[x, y+1] = [x, y]
			if y-1 > -1 and img[x, y-1] == 255 and visitedmap[x, y-1] == 0:
				heappush(heap, (x, y-1, dist+1))
				visitedmap[x, y-1] = 1
				navmap[x, y-1] = [x, y]

	if target_found:
		line = []
		while x != sx and y != sy:
			line.append((x, y))
			(x, y) = navmap[x, y]
		line.append((sx, sy))
		return list(reversed(line))

	print("could not construct a path..")
	return get_straight_path(sx, sy, tx)




	

def find_path(img, m, max_len = 200):
	line = []

	(maxx, _) = np.shape(img)
	print(np.shape(img))

	line.append((0, m))
	x = 1
	y = m
	while x < (maxx - 2): #search until the end of the image is reached
		print("x: ", x)
		print("y: ", y)
		running = True

		while (x < (maxx - 2) and running):
			running = False
			if img[x, y] != 0: # path is clear (white) and end hasn't been reached
				line.append((x, y))
				x += 1
				running = True #horrible workdarpund for no lazy eval

		#apply dijkstra based acid drop
		[line.append(item) for item in acid_drop(img, x, y, max_len)]
		(x, y) = line[-1]

	return line



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

	for m in minima:
		print("working on minima.. %d" % (m))
		line = find_path(img, m)
		print("path found")
		for (x, y) in line:
			print("(%d,%d)" % (x, y))
		line = np.array(line)
		pts = line.reshape((-1,1,2))
		img = cv2.polylines(img,[pts],True,(125))