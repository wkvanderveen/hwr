'''
Example part of the preprocessing pipeline including all needed imports
'''


import sys

sys.path.append('preprocessing/')

import cython
import cv2
import os
from os.path import join, abspath
import numpy as np

#this is the file needed for preprocessing
from preprocessor import preprocess_image


PATH = join(abspath('..'), 'data')
OUTPATH = join(join(abspath('..'), 'data'), 'lines')

if __name__ == '__main__':
	f = join(join(PATH, 'image-data'), 'P632-Fg002-R-C01-R01-fused.jpg')
	if not os.path.isdir(OUTPATH):
		print("path " + OUTPATH + "not found.. creating new dir")
		os.mkdir(OUTPATH)

	bw_img =  cv2.imread(f)
	print("converting image: " + f)

	croppings = preprocess_image(bw_img)
	print("seperated into %d croppings" % (len(croppings)))
	print(np.shape(croppings))

	for line_idx, chars in enumerate(croppings):
		cv2.imwrite(join(OUTPATH, "%d_.png" % (line_idx)), chars)
	print("saved %d croppings!" % (len(croppings)))

