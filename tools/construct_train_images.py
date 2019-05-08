import cv2
import numpy as np
import os
from tqdm import tqdm
import random as rand
from math import ceil

def see(img):
    cv2.imshow("", img)
    cv2.waitKey(0)

def make_lines(set_type, label_dir):

    channels = 3
    n_lines = 100
    line_length_bounds = (20,40) # characters in line (inclusive)
    overlap = 0  # Maximum character overlap in pixels

    dataset     = f"../../data/letters-{set_type}/"
    classes     = os.listdir(dataset)
    num_classes = len(classes)

    max_h, max_w = 0, 0

    target_folder = f"../../data/lines-{set_type}/"
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    print(f"Making lines from the {set_type}ing letters...")
    for line_idx in tqdm(range(n_lines)):

        # Loop over random letters and put them in order in this image.
        line_length = rand.randint(line_length_bounds[0], line_length_bounds[1])

        linepath = os.path.join(target_folder, str(line_idx) + ".jpeg")
        with open(label_dir, "a") as file:
            label = f"{linepath} "
            file.write(str(label).rstrip('\n'))

        chars = []
        positions = {
            'x1': [],  # left edge of character (pixels from left)
            'y1': [],  # top edge of character (pixels from top)
            'x2': [],  # right edge of character (pixels from left)
            'y2': []   # bottom edge of character (pixels from top)
        }

        labels = []

        # Make list of characters and labels
        for character_idx in range(line_length):
            char_name = rand.choice(classes)
            pathname = os.path.join(dataset, char_name)
            pathname = os.path.join(pathname, rand.choice(os.listdir(pathname)))

            img = cv2.imread(pathname)
            chars.append(img)
            labels.append(char_name)

        max_height = max([c.shape[0] for c in chars])

        positions['x1'].append(0)
        positions['x2'].append(chars[0].shape[1])

        for char_idx, char in enumerate(chars):
            border_width = max_height-char.shape[0]

            positions['y1'].append(border_width)
            positions['y2'].append(border_width + char.shape[0])

            chars[char_idx] = cv2.copyMakeBorder(char, max_height-char.shape[0], 0,0,0,cv2.BORDER_REPLICATE)

        line = chars[0]

        if not overlap:
            for char_idx in range(1,line_length):
                positions['x1'].append(line.shape[1])
                line = np.concatenate((line, chars[char_idx]), axis=1)
                positions['x2'].append(line.shape[1])
        else:
            for char_idx in range(1,line_length):
                overlap_pix = min(rand.randint(1,overlap), chars[char_idx].shape[1])
                positions['x1'].append(line.shape[1]-overlap_pix)


                overlap_from_line = line[:, -overlap_pix:]
                overlap_from_char = chars[char_idx][:, :overlap_pix]
                char_remainder = chars[char_idx][:, overlap_pix:]


                line[:, -overlap_pix:] = np.minimum(overlap_from_line, overlap_from_char)
                line = np.concatenate((line, char_remainder), axis=1)
                positions['x2'].append(line.shape[1])

        for char_idx, char in enumerate(chars):
            with open(label_dir, "a") as file:
                label = (f"{positions['x1'][char_idx]} " +
                         f"{positions['y1'][char_idx]} " +
                         f"{positions['x2'][char_idx]} " +
                         f"{positions['y2'][char_idx]} " +
                         f"{classes.index(labels[char_idx])} ")
                file.write(str(label).rstrip('\n'))

        with open(label_dir, "a") as file:
            file.write('\n')
        cv2.imwrite(linepath, line)

        this_h, this_w = line.shape[0], line.shape[1]
        max_h = max(this_h, max_h)
        max_w = max(this_w, max_w)

    return max_h, max_w


max_h1, max_w1 = make_lines(set_type="train", label_dir="../../data/labels-train.txt")
max_h2, max_w2 = make_lines(set_type="test", label_dir="../../data/labels-test.txt")

max_h = ceil(max(max_h1, max_h2)/32.0)*32
max_w = ceil(max(max_w1, max_w2)/32.0)*32

with open("../../data/max_wh.txt", "w+") as filename:
    print(f"{max_w} {max_h}", file=filename)
