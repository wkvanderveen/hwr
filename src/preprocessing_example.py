'''
Example part of the preprocessing pipeline including all needed imports
'''


import sys

sys.path.append('preprocessing/')

import cython
import cv2
import os
from os.path import join, abspath

#this is the file needed for preprocessing
from preprocessor import preprocess_image


PATH = join(abspath('..'), 'data')
OUTPATH = join(join(abspath('..'), 'data'), 'lines')

if __name__ == '__main__':
	files =  [join(join(PATH, 'image-data'), f) for f in os.listdir(join(PATH, 'image-data')) if os.path.isfile(join(join(PATH, 'image-data'),  f)) and f.endswith('-fused.jpg')]

	if not os.path.isdir(OUTPATH):
		print("path " + OUTPATH + "not found.. creating new dir")
		os.mkdir(OUTPATH)

	for (fidx, f) in enumerate(files):
		bw_img =  cv2.imread(f)
		print("converting image: " + f)

		croppings = preprocess_image(bw_img)
		print("seperated into %d croppings" % (len(croppings)))
		# cv2.imshow("original", preprocessed_img)
		# cv2.waitKey(0)

		for idx, c in enumerate(croppings):
			cv2.imwrite(join(OUTPATH, "%d_%d.png" % (fidx, idx)), c)
		# 	cv2.imshow("cropping%d" % idx, c)
		# 	cv2.waitKey(0)
		# cv2.destroyAllWindows()
		print("saved %d croppings!" % (len(croppings)))

