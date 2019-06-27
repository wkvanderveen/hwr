'''
Applies acid drop to a binarized image with a histogram
'''

import cython
import numpy as np
from heapq import heappush, heappop

from binarizer import Binarizer
from smear_test import Smear
from line_segmenter import Line_segmenter

cimport numpy as np


cdef float BURNCOST = 3.5 		#penalty to burn through a single black pixel (this is added to other possible penalties)
cdef float VCOST1 = 2.0 		#penalty to move in the vertical direction away from the target y
cdef float VCOST2 = -0.5		#penalty to move in the vertical direction toward the target y
cdef float LCOST = 1.5			#penalty to move to the left (back)
cdef float RCOST = 1.0			#penalty to move right (forward; this is the preferred action)

cdef list acid_drop_c(np.ndarray[np.uint8_t, ndim=2] img, int x, int y, int tx, int ty, float max_len):
	#type declarations
	cdef int maxx, maxy, sx, sy
	cdef float dist, rdist, ldist, udist, ddist
	cdef list heap, line

	cdef np.ndarray[np.uint32_t, ndim=3] navmap #used for keeping track of the A* part of acid drop
	cdef np.ndarray[np.uint8_t, ndim=2] visitedmap #will only store bools
	cdef np.ndarray[np.uint32_t, ndim=1] targetv, startv
	cdef char target_found #is used as a bool

	#function start
	(maxy, maxx) = np.shape(img)
	heap = []
	visitedmap = np.zeros((maxy, maxx), dtype=np.uint8)
	navmap = np.zeros((maxy, maxx, 2), dtype=np.uint32)
	heappush(heap, (0, x, y))
	target_found = 0
	sx = x #startvalues for x and y
	sy = y

	targetv = np.array((tx, ty), dtype=np.uint32) #used for dist calc
	startv = np.array((sx, sy), dtype=np.uint32) #used for dist calc

	while not target_found and len(heap) > 0:
		dist, x, y = heappop(heap)
		
		if x == tx and y == ty:
			target_found = 1
		elif dist > max_len:
			print("max path length (%f) reached.." % max_len)
			return []
		else:
			#enqueue all directions
			if x+1 < maxx and visitedmap[y, x+1] == 0:
				rdist = dist + RCOST
				if img[y, x+1] != 255:
					rdist += BURNCOST
				heappush(heap, (rdist, x+1, y))
				visitedmap[y, x+1] = 1
				navmap[y, x+1, 0] = x
				navmap[y, x+1, 1] = y
			if x-1 > -1 and visitedmap[y, x-1] == 0:
				ldist = dist + LCOST
				if img[y, x-1] != 255:
					ldist += BURNCOST
				heappush(heap, (ldist, x-1, y))
				visitedmap[y,x-1] = 1
				navmap[y, x-1, 0] = x
				navmap[y, x-1, 1] = y
			if y+1 < maxy and visitedmap[y+1, x] == 0:
				if ((y+1) - ty)*((y+1) - ty) > (y - ty)*(y - ty): #diverging from the target y
					udist = dist + VCOST1
				else:	#coming towards the target y
					udist = dist + VCOST2
				if img[y+1, x] != 255:
					udist += BURNCOST
				heappush(heap, (udist, x, y+1))
				visitedmap[y+1, x] = 1
				navmap[y+1, x, 0] = x
				navmap[y+1, x, 1] = y

			if y-1 > -1 and visitedmap[y-1, x] == 0:
				if ((y-1) - ty)*((y-1) - ty) > (y - ty)*(y - ty): #diverging from the target y
					ddist = dist + VCOST1
				else:	#coming towards the target y
					ddist = dist + VCOST2
				if img[y-1, x] != 255:
					ddist += BURNCOST
				heappush(heap, (ddist, x, y-1))
				visitedmap[y-1, x] = 1
				navmap[y-1, x, 0] = x
				navmap[y-1, x, 1] = y

	if target_found:
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



class Acid_drop: #only wrapped in a class for uniformity.
	def acid_drop(self, np.ndarray[np.uint8_t, ndim=2] img, int x, int y, int tx, int ty, float max_len):
		return acid_drop_c(img, x, y, tx, ty, max_len)
	