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

if __name__ == '__main__':
	files =  [join(join(PATH, 'image-data'), f) for f in os.listdir(join(PATH, 'image-data')) if os.path.isfile(join(join(PATH, 'image-data'),  f)) and f.endswith('-fused.jpg')]

	for f in files:
		bw_img =  cv2.imread(f)
		print("converting image: " + f)

		(preprocessed_img, croppings) = preprocess_image(bw_img)
		print("seperated into %d croppings" % (len(croppings)))
		cv2.imshow("original", preprocessed_img)
		cv2.waitKey(0)

		for idx, c in enumerate(croppings):
			cv2.imshow("cropping%d" % idx, c)
			cv2.waitKey(0)
		cv2.destroyAllWindows()

