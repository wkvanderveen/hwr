import cv2
import numpy as np
import os
from tqdm import tqdm
import random as rand
from math import ceil

class Linemaker(object):
    """docstring for Linemaker"""
    def __init__(self, set_type, source_dir, target_dir, label_dir, line_length_bounds, n_lines, max_overlap):
        super(Linemaker, self).__init__()
        self.set_type = set_type
        self.label_dir = label_dir
        self.target_dir = target_dir
        self.source_dir = source_dir
        self.line_length_bounds = line_length_bounds
        self.n_lines = n_lines
        self.max_overlap = max_overlap

    def make_lines(self):

        channels = 3
        n_lines = 100

        classes     = os.listdir(self.source_dir)
        num_classes = len(classes)

        max_h, max_w = 0, 0

        if not os.path.isdir(self.target_dir):
            os.mkdir(self.target_dir)

        print("Making lines from the {}ing letters...".format(self.set_type))
        for line_idx in tqdm(range(self.n_lines)):

            # Loop over random letters and put them in order in this image.
            line_length = rand.randint(self.line_length_bounds[0],
                                       self.line_length_bounds[1])

            linepath = os.path.join(self.target_dir, str(line_idx) + ".jpeg")
            with open(self.label_dir, "a") as file:
                label = "{} ".format(linepath)
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
                pathname = os.path.join(self.source_dir, char_name)
                x = os.listdir(pathname)
                pathname = os.path.join(pathname, rand.choice(x))

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

            if not self.max_overlap:
                for char_idx in range(1,line_length):
                    positions['x1'].append(line.shape[1])
                    line = np.concatenate((line, chars[char_idx]), axis=1)
                    positions['x2'].append(line.shape[1])
            else:
                for char_idx in range(1,line_length):
                    overlap_pix = min(rand.randint(1,self.max_overlap), chars[char_idx].shape[1])
                    positions['x1'].append(line.shape[1]-overlap_pix)


                    overlap_from_line = line[:, -overlap_pix:]
                    overlap_from_char = chars[char_idx][:, :overlap_pix]
                    char_remainder = chars[char_idx][:, overlap_pix:]


                    line[:, -overlap_pix:] = np.minimum(overlap_from_line, overlap_from_char)
                    line = np.concatenate((line, char_remainder), axis=1)
                    positions['x2'].append(line.shape[1])

            for char_idx, char in enumerate(chars):
                with open(self.label_dir, "a") as file:
                    label = ("{} ".format(positions['x1'][char_idx]) +
                             "{} ".format(positions['y1'][char_idx]) +
                             "{} ".format(positions['x2'][char_idx]) +
                             "{} ".format(positions['y2'][char_idx]) +
                             "{} ".format(classes.index(labels[char_idx])))
                    file.write(str(label).rstrip('\n'))

            with open(self.label_dir, "a") as file:
                file.write('\n')
            cv2.imwrite(linepath, line)

            this_h, this_w = line.shape[0], line.shape[1]
            max_h = max(this_h, max_h)
            max_w = max(this_w, max_w)

        return max_h, max_w

if __name__ == "__main__":
    linemaker = Linemaker(set_type="train",
                          source_dir="../data/letters-train",
                          target_dir="../data/lines-train",
                          label_dir="../data/labels-train.txt",
                          line_length_bounds=(10,20),
                          n_lines=20,
                          max_overlap=10)
    max_h1, max_w1 = linemaker.make_lines()

    linemaker = Linemaker(set_type="test",
                          source_dir="../data/letters-test",
                          target_dir="../data/lines-test",
                          label_dir="../data/labels-test.txt",
                          line_length_bounds=(10,20),
                          n_lines=20,
                          max_overlap=10)
    max_h2, max_w2 = linemaker.make_lines()

    max_h = ceil(max(max_h1, max_h2)/32.0)*32
    max_w = ceil(max(max_w1, max_w2)/32.0)*32

    with open(dimensions_dir, "w+") as filename:
        print("{} {}".format(max_w,max_h))
