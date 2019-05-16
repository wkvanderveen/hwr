import os
import sys
from os.path import isfile, join, abspath
import cv2
import numpy as np

sys.path.append(abspath("../src")) #for importing Binarizer and Smear

from smear_test import Smear
from binarizer import Binarizer

PATH = "../data/image-data"
OUTPATH = "../data/lines"

if __name__ == '__main__':
	s = Smear()
	b = Binarizer()

	path = abspath(PATH) #get absolute path
	print("Getting data from " + path)

	outpath = abspath(OUTPATH)
	if not os.path.isdir(outpath):
		inp = input("path " + outpath + "not found, press y to create file\n")
		if inp =='y':
			os.mkdir(outpath)
		else:
			quit()


	files =  [f for f in os.listdir(path) if (isfile(join(path, f)) and f.endswith("-fused.jpg") )]

	for fidx, f in enumerate(files):
		fn_parent = f.split('.')[0]
		print("working on %s, (%d/%d) %f %%" % (fn_parent, fidx + 1, len(files), np.float((fidx+1) / len(files) * 100.0)))

		img = cv2.imread(join(path, f))
		img = b.binarize_image(img)
		(contoured_img, smear_img, lines) = s.split_into_lines_and_contour(img)
		cv2.imwrite(join(outpath, fn_parent + "_contoured.jpg"), contoured_img)
		cv2.imwrite(join(outpath, fn_parent + "_smeared.jpg"), smear_img)

		for lidx, line in enumerate(lines):
			fn_out = "%s_line%d.png" % (fn_parent, lidx)

			cv2.imwrite(join(outpath, fn_out), line)

			print("- saved " + fn_out)

