import cv2
import numpy as np
import os
from tqdm import tqdm
import random as rand


channels = 3
n_lines = 100
line_length_bounds = (20,40) # characters in line (inclusive)
overlap = 15  # Maximum character overlap in pixels

# initialize blank lines
# lines = 255*np.ones((n_lines, height, width, channels), np.uint8)

dataset = "../../data/train/"
classes     = os.listdir(dataset)
num_classes = len(classes)

target_folder = "../../data/lines/"
if not os.path.isdir(target_folder):
    os.mkdir(target_folder)


def see(img):
    cv2.imshow("", img)
    cv2.waitKey(0)

for line_idx in range(n_lines):

    # Loop over random letters and put them in order in this image.
    line_length = rand.randint(line_length_bounds[0], line_length_bounds[1])

    linepath = os.path.join(target_folder, str(line_idx) + ".jpeg")
    with open("../../data/linedata.txt", "a") as file:
        label = f"{linepath} "
        file.write(str(label).rstrip('\n'))

    chars = []
    positions = {
        'x1': [],
        'x2': [],
        'y2': []
    }

    labels = []

    ## GENERATE LINE IMAGE ##

    # Make list of characters and labels
    for character_idx in range(line_length):
        char_name = rand.choice(classes)
        pathname = os.path.join(dataset, char_name)
        pathname = os.path.join(pathname, rand.choice(os.listdir(pathname)))

        img = cv2.imread(pathname)
        chars.append(img)
        labels.append(char_name)
        positions['y2'].append(chars[character_idx].shape[0])


    max_height = max([c.shape[0] for c in chars])

    positions['x1'].append(0)
    positions['x2'].append(chars[0].shape[1])

    for char_idx, char in enumerate(chars):
        chars[char_idx] = cv2.copyMakeBorder(char, max_height-char.shape[0], 0,0,0,cv2.BORDER_REPLICATE)

    line = chars[0]

    if not overlap:
        for char_idx in range(1,line_length):
            positions['x1'].append(line.shape[0])
            line = np.concatenate((line, chars[char_idx]), axis=1)
            positions['x2'].append(line.shape[0])
    else:
        for char_idx in range(1,line_length):
            overlap_pix = min(rand.randint(1,overlap), chars[char_idx].shape[1])
            positions['x1'].append(line.shape[0]-overlap_pix)


            overlap_from_line = line[:, -overlap_pix:]
            overlap_from_char = chars[char_idx][:, :overlap_pix]
            char_remainder = chars[char_idx][:, overlap_pix:]


            line[:, -overlap_pix:] = np.minimum(overlap_from_line, overlap_from_char)
            line = np.concatenate((line, char_remainder), axis=1)
            positions['x2'].append(line.shape[0])

    for char_idx, char in enumerate(chars):
        with open("../../data/linedata.txt", "a") as file:
            label = (f"{positions['x1'][char_idx]} 0 " +
                     f"{positions['x2'][char_idx]} " +
                     f"{positions['y2'][char_idx]} {classes.index(labels[char_idx])}")
            file.write(str(label).rstrip('\n'))

    with open("../../data/linedata.txt", "a") as file:
        file.write('\n')
    cv2.imwrite(linepath, line)
