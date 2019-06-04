'''
Example part of the preprocessing pipeline including all needed imports
'''


import sys
import time #used for timing the app

sys.path.append('preprocessing/')

import cython
import cv2
import os
from os.path import join, abspath
import sys
import numpy as np
from heapq import heappush, heappop

from binarizer import Binarizer
from smear_test import Smear
from line_segmenter import Line_segmenter
from acid_drop import Acid_drop
from preprocessor import preprocess_image


PATH = join(abspath('..'), 'data')
nice_img_name = 'P632-Fg002-R-C01-R01'#'P583-Fg006-R-C01-R01' 
bad_img_name = 'P21-Fg006-R-C01-R01'
ground_truth_img_name = '124-Fg004'


if __name__ == '__main__':
	b = Binarizer()
	s = Smear()
	l = Line_segmenter()
	a = Acid_drop()



	img_name = ground_truth_img_name

	files =  [join(join(PATH, 'image-data'), f) for f in os.listdir(join(PATH, 'image-data')) if os.path.isfile(join(join(PATH, 'image-data'),  f)) and f.endswith('-fused.jpg')]

	idx = 0
	# bw_img =  cv2.imread(join(join(PATH, 'image-data'), img_name + '-fused.jpg'))
	for f in files:
		bw_img =  cv2.imread(f)
		print("converting image: " + f)


		preprocess_image(bw_img)



		# bw_img = cv2.cvtColor(bw_img,cv2.COLOR_BGR2GRAY) #convert to grayscale

		# t = time.time()

		# img = b.binarize_image(bw_img)

		# print("time passed in binarizer: ", time.time() - t)
		# t2 = time.time()

		# (_, smear, _) = s.split_into_lines_and_contour(img)
		# hist = l.create_v_histogram(smear)
		# print("time passed in smear: ", time.time() - t2)
		# print("time passed in prepro: ", time.time() - t)

		# minima = l.get_minima(hist)

		# ## actually apply acid drop
		# (_, maxx) = np.shape(img)

		# for m in minima:
		# 	print("working on minima.. %d" % (m))
		# 	sys.stdout.flush()
		# 	# line = find_path(img, m)
		# 	line = a.acid_drop(img, 0, m, maxx-1, m, 9000)
		# 	# for (x, y) in line:
		# 	# 	print("(%d,%d)" % (x, y))
		# 	line = np.array(line)
		# 	# pts = line.reshape((-1,1,2))
		# 	img = cv2.polylines(img,[line],False,(125))


		# cv2.imwrite('a-star%d.png' % (idx), img)
		# print("saved image to a-star.png")
		# idx += 1