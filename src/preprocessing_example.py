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
	# files =  [join(join(PATH, 'image-data'), f) for f in os.listdir(join(PATH, 'image-data')) if os.path.isfile(join(join(PATH, 'image-data'),  f)) and f.endswith('-fused.jpg')]
	f = join(join(PATH, 'image-data'), 'P632-Fg002-R-C01-R01-fused.jpg')
	if not os.path.isdir(OUTPATH):
		print("path " + OUTPATH + "not found.. creating new dir")
		os.mkdir(OUTPATH)

# <<<<<<< Updated upstream
# 	for (fidx, f) in enumerate(files):
# 		bw_img =  cv2.imread(f)
# 		print("converting image: " + f)

# 		croppings = preprocess_image(bw_img)
# 		print("seperated into %d croppings" % (len(croppings)))
# 		# cv2.imshow("original", preprocessed_img)
# 		# cv2.waitKey(0)

# 		for idx, c in enumerate(croppings):
# 			cv2.imwrite(join(OUTPATH, "%d_%d_%d.png" % (fidx, idx, sum_black_pixels(c))), c)
# 		# 	cv2.imshow("cropping%d" % idx, c)
# 		# 	cv2.waitKey(0)
# 		# cv2.destroyAllWindows()
# 		print("saved %d croppings!" % (len(croppings)))
# =======
	# for (fidx, f) in enumerate(files):
	bw_img =  cv2.imread(f)
	print("converting image: " + f)

	croppings = preprocess_image(bw_img)
	print("seperated into %d croppings" % (len(croppings)))
	# cv2.imshow("original", preprocessed_img)
	# cv2.waitKey(0)

	print(np.shape(croppings))

	for line_idx, chars in enumerate(croppings):
		for idx, c in enumerate(chars):
			cv2.imwrite(join(OUTPATH, "%d_%d.png" % (line_idx, idx)), c)
			# 	cv2.imshow("cropping%d" % idx, c)
			# 	cv2.waitKey(0)
			# cv2.destroyAllWindows()
			print("saved %d croppings!" % (len(chars)))
# >>>>>>> Stashed changes

