import os
from split import Splitter
from data_augmenter import Augmenter
from construct_train_images import Linemaker
from math import ceil

# PARAMETERS
orig_letters_dir = "../../data/letters/"
letters_train_dir = "../../data/letters-train/"
letters_test_dir = "../../data/letters-test/"
lines_train_dir = "../../data/lines-train/"
lines_test_dir = "../../data/lines-test/"
label_train_dir = "../../data/labels-train.txt"
label_test_dir = "../../data/labels-test.txt"
dimensions_dir = "../../data/max_wh.txt"

split_percentage = 20
line_length_bounds = (10,40)
n_training_lines = 20
n_testing_lines = 20
max_overlap_train = 10
max_overlap_test = 10

# PREPARE NETWORK IF NOT READY
 # SPLIT
 # AUGMENT
 # MAKE LINES
 # PREPARE NETWORK

# READ INPUT

# BINARIZE INPUT

# SEGMENT LINES FROM INPUT

# FOR LINE IN LINES:
    # FEED LINE TO NETWORK, GET RESULT

    # POSTPROCESS CHARACTER LIKELIHOODS

    # PRINT PREDICTION


if not os.path.exists(letters_train_dir):
    splitter = Splitter(source_dir=orig_letters_dir,
                        train_dir=letters_train_dir,
                        test_dir=letters_test_dir,
                        percentage=split_percentage)
    print(f"Splitting {splitter.percentage} of the data found in {splitter.source_dir}...")
    splitter.split()

    print(f"Augmenting training letters...")
    augmenter = Augmenter(source_dir=letters_train_dir, shear=True, coarse_dropout=(0.02, 0.5))
    augmenter.augment()
else:
    print("Training dataset detected! Skipping splitting & augmenting.")


if not os.path.exists(lines_train_dir):
    linemaker = Linemaker(set_type="train",
                          source_dir=letters_train_dir,
                          target_dir=lines_train_dir,
                          label_dir=label_train_dir,
                          line_length_bounds=line_length_bounds,
                          n_lines=n_training_lines,
                          max_overlap=max_overlap_train)
    max_h1, max_w1 = linemaker.make_lines()

    linemaker = Linemaker(set_type="test",
                          source_dir=letters_test_dir,
                          target_dir=lines_test_dir,
                          label_dir=label_test_dir,
                          line_length_bounds=line_length_bounds,
                          n_lines=n_testing_lines,
                          max_overlap=max_overlap_test)
    max_h2, max_w2 = linemaker.make_lines()

    max_h = ceil(max(max_h1, max_h2)/32.0)*32
    max_w = ceil(max(max_w1, max_w2)/32.0)*32

    with open(dimensions_dir, "w+") as filename:
        print(f"{max_w} {max_h}", file=filename)
else:
    print("Line data detected! Skipping linemaking.")
