import os
import sys
from os.path import isfile, join, abspath
import cv2

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


	files =  [join(path, f) for f in os.listdir(path) if (isfile(join(path, f)) and f.endswith("-fused.jpg") )]

	for fidx, f in enumerate(files):
		print("working on " + f)
		img = cv2.imread(f)
		img = b.binarize_image(img)
		lines = s.split_into_lines(img)

		fn_parent = f.split('/')[-1]

		for lidx, line in enumerate(lines):
			fn = "file_%d_line%d.png" % (fidx, lidx)

			cv2.imwrite(join(outpath, fn), line)

			print("- saved " + fn)

