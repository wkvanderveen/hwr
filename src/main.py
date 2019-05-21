import os
from time import sleep
from split import Splitter
from data_augmenter import Augmenter
from construct_train_images import Linemaker
from math import ceil
from make_tfrecords import TfRecordMaker
from kmeans import AnchorMaker
from convert_weight import WeightConverter
from quick_train import Trainer
from show_input_image import ExampleDisplayer
from quick_test import Tester

## PARAMETERS ##

# File structure parameters
orig_letters_dir = "../data/original_letters/"
letters_train_dir = "../data/letters-train/"
letters_test_dir = "../data/letters-test/"
lines_train_dir = "../data/lines-train/"
lines_test_dir = "../data/lines-test/"
label_train_path = "../data/labels-train.txt"
label_test_dir = "../data/labels-test.txt"
checkpoint_dir = "../data/checkpoint/"
dimensions_file = "../data/dimensions.txt"
weights_dir = "../data/weights/"
anchor_file = "../data/anchors.txt"

# Data parameters
num_classes = 5  # not too few
split_percentage = 20
line_length_bounds = (20,50)
n_training_lines = 50
n_testing_lines = 50
max_overlap_train = 20
max_overlap_test = 20

# Network parameters
cluster_num = 9
iou_threshold = 0.1
score_threshold = 0.1
batch_size = 8
steps = 100
learning_rate = 1e-3
decay_steps = 100
decay_rate = 0.9
shuffle_size = 200
eval_internal = 10
save_internal = 50

# Other parameters
retrain = False
show_tfrecord_example = True
test_example = False
evaluate_network = False # TBA


# READ INPUT

# BINARIZE INPUT

# SEGMENT LINES FROM INPUT


# PREPARE NETWORK IF NOT READY
network_exists = bool(os.path.isfile("../data/checkpoint/checkpoint"))

if not network_exists or retrain:

    if not os.path.exists(letters_train_dir):
        splitter = Splitter(source_dir=orig_letters_dir,
                            num_classes=num_classes,
                            train_dir=letters_train_dir,
                            test_dir=letters_test_dir,
                            percentage=split_percentage)
        print("Splitting {}% of the data found in {}...".format(splitter.percentage,splitter.source_dir))
        sleep(1)
        num_classes = splitter.split()

        print("Augmenting training letters...")
        augmenter = Augmenter(source_dir=letters_train_dir, shear=True, coarse_dropout=(0.02, 0.5))
        augmenter.augment()
    else:
        print("Training dataset detected! Skipping splitting & augmenting.")
        sleep(1)
        num_classes = len(os.listdir(letters_train_dir))


    if not os.path.exists(lines_train_dir):
        linemaker = Linemaker(set_type="train",
                              source_dir=letters_train_dir,
                              target_dir=lines_train_dir,
                              label_dir=label_train_path,
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
        img_dims = (max_h, max_w)

        with open(dimensions_file, "w+") as filename:
            print("{} {}".format(max_h,max_w), file=filename)
        img_dims = (max_h, max_w)


    else:
        print("Line data detected! Skipping linemaking.")
        sleep(1)

        with open(dimensions_file, "r") as max_dimensions:
            img_h, img_w = [int(x) for x in max_dimensions.read().split()]
        img_dims = (img_h, img_w)


    if not os.path.isfile(os.path.normpath(lines_train_dir) + ".tfrecords"):
        print("Making tfrecords...")
        recordmaker = TfRecordMaker(imgs_dir=lines_train_dir, label_path=label_train_path)
        recordmaker.make_records()
        recordmaker = TfRecordMaker(imgs_dir=lines_test_dir, label_path=label_test_dir)
        recordmaker.make_records()
    else:
        print("Not creating TfRecords files because they already exist!")
        sleep(1)

    if not os.path.isfile(anchor_file):
        print("Making anchors...")
        anchormaker = AnchorMaker(target_file=anchor_file,
                                  label_path=label_train_path,
                                  cluster_num=cluster_num)
        anchormaker.make_anchors()

    else:
        print("Not creating anchors file because it already exists!")
        sleep(1)

    if not os.path.exists(weights_dir):
        print("Error: no weights detected! You need the pretrained " +
              "weights in the {} directory.".format(weights_dir))

    weightconverter = WeightConverter(freeze=False,
                                      convert=True,
                                      num_classes=num_classes,
                                      img_dims=img_dims,
                                      checkpoint_dir=checkpoint_dir,
                                      weights_dir=weights_dir,
                                      anchors_path=anchor_file,
                                      score_threshold=score_threshold,
                                      iou_threshold=iou_threshold)
    print("Converting weights...")
    weightconverter.convert_weights()

    trainer = Trainer(num_classes=num_classes,
                      batch_size=batch_size,
                      steps=steps,
                      learning_rate=learning_rate,
                      decay_steps=decay_steps,
                      decay_rate=decay_rate,
                      shuffle_size=shuffle_size,
                      eval_internal=eval_internal,
                      save_internal=save_internal,
                      img_dims=img_dims,
                      anchors_path=anchor_file,
                      train_records=os.path.normpath(lines_train_dir) + ".tfrecords",
                      test_records=os.path.normpath(lines_test_dir) + ".tfrecords",
                      checkpoint_path=checkpoint_dir)
    print("Training network...")
    trainer.train()

    network_exists = True

else: # if network already exists and not retraining
    print("Network already trained!")
    sleep(1)

    with open(dimensions_file, "r") as max_dimensions:
        img_h, img_w = [int(x) for x in max_dimensions.read().split()]
    img_dims = (img_h, img_w)
    num_classes = len(os.listdir(letters_train_dir))


if network_exists and show_tfrecord_example:
    example_displayer = ExampleDisplayer(source_dir=os.path.normpath(lines_train_dir) + ".tfrecords",
                                         img_dims=img_dims,
                                         anchor_dir=anchor_file,
                                         num_classes=num_classes)
    example_displayer.show_example()

if network_exists and test_example:

    weightconverter = WeightConverter(freeze=True,
                                      num_classes=num_classes,
                                      img_dims=img_dims,
                                      checkpoint_dir=checkpoint_dir,
                                      weights_dir=weights_dir,
                                      anchors_path=anchor_file,
                                      score_threshold=score_threshold,
                                      iou_threshold=iou_threshold,
                                      convert=False,
                                      checkpoint_step=steps-(steps%save_internal))
    weightconverter.convert_weights()

    tester = Tester(source_dir=lines_test_dir,
                    num_classes=num_classes,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                    img_dims=img_dims,
                    checkpoint_dir=checkpoint_dir,
                    letters_test_dir=letters_test_dir)
    tester.test()

if network_exists and evaluate_network:
    pass
    # evaluater = Evaluater()
    # evaluater.eval()


# FOR LINE IN LINES:
    # FEED LINE TO NETWORK, GET RESULT

    # POSTPROCESS CHARACTER LIKELIHOODS

    # PRINT PREDICTION


"""
TODO:
* Put data augmentation stuff in parameters here
* Actually train on the data to see if YOLO works in the first place


"""