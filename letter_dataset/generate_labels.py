import sys
import cv2
import os
from os.path import isfile, join
import numpy as np
import matplotlib.pylab as plt
import shutil
np.set_printoptions(threshold=sys.maxsize)

path_train = '../imagesTrain/data/letters/'
path_test = '../imagesTest/data/letters/'
output_path = 'images'

letters = []
paths = []
write_lines = []
names = []
for class_, filepath in enumerate(os.listdir(path_train)):
	sub_path = path_train+filepath
	names.append(filepath)
	for sub_file in os.listdir(sub_path):
		img_path = sub_path+'/'+sub_file
		paths.append(img_path)
		letter = plt.imread(img_path)
		height = str(letter.shape[0])
		width = str(letter.shape[1])
		write_lines.append(str(img_path)+' '+'0 '+'0 '+str(width)+' '+str(height)+' '+str(class_))

		#names.append()
with open('labels.txt', mode='wt', encoding='utf-8') as file:
	file.write('\n'.join(write_lines))
with open('letters.names', mode='wt', encoding='utf-8') as file:
	file.write('\n'.join(names))

'''
temp = [f for f in os.listdir(path_train) if not isfile(join(path_train, f))]
for path in temp:
	new_path = join(path_train, path)
	files = [f for f in os.listdir(new_path) if isfile(join(new_path, f))]
	for file in files:
		#print(output_path, '\n', new_path, '\n', file)
		os.rename(join(new_path, file), join(output_path, file))
'''	
