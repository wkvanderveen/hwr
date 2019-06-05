import cv2
import numpy as np
import os
from tqdm import tqdm
import random as rand
from math import ceil

class Linemaker(object):
    """docstring for Linemaker"""
    def __init__(self, set_type, source_dir, target_dir, cell_size, label_dir, line_length_bounds, n_lines, max_overlap):
        super(Linemaker, self).__init__()
        self.set_type = set_type
        self.label_dir = label_dir
        self.target_dir = target_dir
        self.source_dir = source_dir
        self.line_length_bounds = line_length_bounds
        self.n_lines = n_lines
        self.max_overlap = max_overlap
        self.cell_size = cell_size

    def make_lines(self):

        channels = 3
        n_lines = 100

        classes     = os.listdir(self.source_dir)
        num_classes = len(classes)

        max_h, max_w = 0, 0
        max_paddings = [10, 10, 10, 10] # L,T,R,B  (min 5)


        if not os.path.isdir(self.target_dir):
            os.mkdir(self.target_dir)

        print(f"Making lines from the {self.set_type}ing letters...")
        for line_idx in tqdm(range(self.n_lines)):

            # Loop over random letters and put them in order in this image.
            line_length = rand.randint(self.line_length_bounds[0],
                                       self.line_length_bounds[1])

            linepath = os.path.join(self.target_dir, str(line_idx) + ".jpeg")
            with open(self.label_dir, "a") as file:
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
                pathname = os.path.join(self.source_dir, char_name)
                x = os.listdir(pathname)
                pathname = os.path.join(pathname, rand.choice(x))

                img = cv2.imread(pathname)
                chars.append(img)
                labels.append(char_name)

            max_height = max_paddings[1] + max_paddings[3] + max([c.shape[0] for c in chars])

            positions['x1'].append(0)
            positions['x2'].append(chars[0].shape[1])

            for char_idx in range(len(chars)):

                top_offset = rand.randint(5, max_paddings[1])
                bot_offset = rand.randint(5, max_paddings[3])

                # add random top and bottom offset
                chars[char_idx] = cv2.copyMakeBorder(
                    chars[char_idx],
                    top_offset,
                    bot_offset,
                    0,
                    0,
                    cv2.BORDER_REPLICATE)

                border_width = max_height-chars[char_idx].shape[0]

                positions['y1'].append(border_width+top_offset)
                positions['y2'].append(border_width+chars[char_idx].shape[0]-bot_offset)

                # top up the char to the max image size
                chars[char_idx] = cv2.copyMakeBorder(
                    chars[char_idx],
                    max_height-chars[char_idx].shape[0],
                    0,
                    0,
                    0,
                    cv2.BORDER_REPLICATE)

            # Start making the line from the characters

            line = chars[0]

            # left pad the line
            left_offset = rand.randint(5,max_paddings[0])

            positions['x1'][0] += left_offset
            positions['x2'][0] += left_offset

            line = cv2.copyMakeBorder(
                    line,
                    0,
                    0,
                    left_offset,
                    0,
                    cv2.BORDER_REPLICATE)

            if not self.max_overlap:
                for char_idx in range(1,line_length):
                    positions['x1'].append(line.shape[1])
                    line = np.concatenate((line, chars[char_idx]), axis=1)
                    positions['x2'].append(line.shape[1])

                    # Add spaces randomly
                    if rand.random() < 0.2:
                        line = np.concatenate((line, 255*np.ones(shape=chars[char_idx].shape)), axis=1)
            else:
                for char_idx in range(1,line_length):
                    # Take a random overlap distance
                    overlap_pix = min(rand.randint(
                        1,self.max_overlap), min(chars[char_idx].shape[1], line.shape[1]))

                    # Store bounding box coordinate
                    positions['x1'].append(line.shape[1]-overlap_pix)

                    overlap_from_line = line[:, -overlap_pix:]

                    overlap_from_char = chars[char_idx][:, :overlap_pix]

                    char_remainder = chars[char_idx][:, overlap_pix:]

                    # Overlap the parts by taking the minimum values
                    line[:, -overlap_pix:] = np.minimum(
                        overlap_from_line, overlap_from_char)

                    line = np.concatenate((line, char_remainder), axis=1)
                    positions['x2'].append(line.shape[1])

                    # Add spaces randomly
                    if rand.random() < 0.2:
                        line = np.concatenate((line, 255*np.ones(shape=chars[char_idx].shape)), axis=1)

            line = cv2.copyMakeBorder(
                    line,
                    0,
                    0,
                    0,
                    rand.randint(5, max_paddings[3]),
                    cv2.BORDER_REPLICATE)

            for char_idx, char in enumerate(chars):
                with open(self.label_dir, "a") as file:
                    label = (f"{positions['x1'][char_idx]} " +
                             f"{positions['y1'][char_idx]} " +
                             f"{positions['x2'][char_idx]} " +
                             f"{positions['y2'][char_idx]} " +
                             f"{classes.index(labels[char_idx])} ")
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
                          source_dir="../../data/letters-train",
                          target_dir="../../data/lines-train",
                          label_dir="../../data/labels-train.txt",
                          line_length_bounds=(10,20),
                          n_lines=20,
                          max_overlap=10)
    max_h1, max_w1 = linemaker.make_lines()

    linemaker = Linemaker(set_type="test",
                          source_dir="../../data/letters-test",
                          target_dir="../../data/lines-test",
                          label_dir="../../data/labels-test.txt",
                          line_length_bounds=(10,20),
                          n_lines=20,
                          max_overlap=10)
    max_h2, max_w2 = linemaker.make_lines()

    max_h = ceil(max(max_h1, max_h2)/float(self.cell_size))*self.cell_size
    max_w = ceil(max(max_w1, max_w2)/float(self.cell_size))*self.cell_size

    with open(dimensions_dir, "w+") as filename:
        print(f"{max_w} {max_h}", file=filename)
