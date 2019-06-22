'''
final_classifier.py

This is the file that will be submitted for this project. 

It should take a path as its input (through command line) and write the transcription of a file to a seperate .txt file

'''
import sys #for argv
from os.path import isfile, isdir, join, abspath, isabs # for path manipulations
import os 	# for listdir
import cv2 	# for reading in the image

## import own code
sys.path.append('preprocessing/')
from preprocessor import preprocess_image
from bayesian_postp import Bayesian_processor
from write_to_file import write_to_file
from sliding_window import SlidingWindow

if __name__ == '__main__':

	## assert correct usage
	if len(sys.argv) != 2:
		print("usage: python final_classifier.py <path_to_test_folder>")
		quit()

	## build and check absolute path
	if isabs(sys.argv[1]):
		path = sys.argv[1]
	else:
		path = abspath(sys.argv[1])

	if not isdir(path):
		print("usage: python final_classifier.py <path_to_test_folder>")
		print("incorrect path given. Path did not lead to a folder")
		print("path: ", path)
		quit()

	## process all files

	files = files =  [f for f in os.listdir(path) if isfile(join(path,  f))]


	for file in files:
		print("Transcribing \"%s\"." % (file) )

		## load in image
		img = cv2.imread(join(path, file))

		## preprocess image
		preprocessed_lines = preprocess_image(img)

		## get root filename for writing the transcribed lines
		outfile = file.split('.')[0] 

		## classify lines
		for line in preprocessed_lines:
			sentence = ''
			for letter in line:
			
				## neural network call here
				sw = SlidingWindow()
				sw.WRITE_WINDOWS = False # If True, the input images of the cnn will be written to a file
				sw.load_image_final_pipeline(letter)
				transcribed_lines = sw.get_letters()

				## apply postprocessing
				postp = Bayesian_processor()
				final_letter = postp.apply_postprocessing(transcribed_lines)
				print(final_letter)
				sentence = sentence+str(final_letter)

				## write croppings to file
			write_to_file(sentence, path, outfile)

		print("Succesfully transcribed \"%s\" to \"%s\"." % (file, outfile))

	print("Finished transcribing.")