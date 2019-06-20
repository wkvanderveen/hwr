import os
from os.path import abspath, join
import sys
import cv2
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

sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}' +
                f'/preprocessing/')

from preprocessor import preprocess_image

# # PARAMETERS # #

# tensorboard --logdir train:./train_summary,eval:./test_summary

# File structure parameters
core_data_path = join(join(abspath(".."), ".."), "data")

orig_letters_dir = join(core_data_path, "original_letters")
letters_train_dir = join(core_data_path, "letters-train")
letters_test_dir = join(core_data_path, "letters-test")
lines_train_dir = join(core_data_path, "lines-train")
lines_test_dir = join(core_data_path, "lines-test")
label_train_path = join(core_data_path, "labels-train.txt")
label_test_dir = join(core_data_path, "labels-test.txt")
checkpoint_dir = join(core_data_path, "checkpoint")
dimensions_file = join(core_data_path, "dimensions.txt")
weights_dir = join(core_data_path, "weights/")
anchor_file = join(core_data_path, "anchors.txt")
image_dir = join(core_data_path, "image-data")
processed_image_dir = join(core_data_path, "new-lines") #change later


# Data parameters
num_classes = 4
split_percentage = 20
augment = False
line_length_bounds = (10, 30)
n_training_lines = 2000
n_testing_lines = 500
max_overlap_train = 10
max_overlap_test = 10
max_boxes = 12
test_on_train = False
test_on_scrolls = True

# Network parameters (darknet)
n_filters_dn = (1024*1024,)  # More is better
n_strides_dn = (1, )  # Only increase beyond 1 if needed for memory savings
n_ksizes_dn = (30, )  # Optimally: half of the usual letter width and height

# Number of anchors (the final conv layer in yolo.detection_layer()
#   has cluster_num+5 filters)
cluster_num = num_classes  # This is a heuristic, not sure if best

# Thresholds and filters
filters = True
iou_threshold = 0.0
score_threshold = 0.0
ignore_threshold = 0.0  # doesn't do anything
size_threshold = (10, 10)  # in pixels
# Also remember the min_dist parameter in utils and iou_threshold in quick_train!

batch_size = 3
steps = 200
learning_rate = 1e-5
decay_steps = 50
decay_rate = 0.95
shuffle_size = 20
eval_internal = 100
save_internal = 100
print_every_n = 10
cell_size = 1


# Other parameters
retrain = False
show_tfrecord_example = False
test_example = True


# [preprocessing here]
if not os.path.isdir(processed_image_dir):
    print("Preprocessing images")
    files = [join(image_dir, fn) for fn in os.listdir(image_dir)
             if os.path.isfile(join(image_dir, fn))]
    os.mkdir(processed_image_dir)
    idx = 0
    for file in files:
        extracted_lines = preprocess_image(cv2.imread(file))
        for line in extracted_lines:
            cv2.imwrite(join(processed_image_dir, "%d.png" % (idx)), line)
            idx += 1

else:
    print("Preprocessed images detected!\nSkipping preprocessing.")


network_exists = bool(os.path.isfile("../../data/checkpoint/checkpoint"))

if not network_exists or retrain:

    if not os.path.exists(letters_train_dir):
        splitter = Splitter(source_dir=orig_letters_dir,
                            num_classes=num_classes,
                            train_dir=letters_train_dir,
                            test_dir=letters_test_dir,
                            percentage=split_percentage)
        print(f"Splitting {splitter.percentage}% of the data " +
              f"found in {splitter.source_dir}...")

        sleep(0.2)
        num_classes = splitter.split()
        if augment:
            print(f"Augmenting training letters...")
            augmenter = Augmenter(source_dir=letters_train_dir,
                                  shear=True,
                                  coarse_dropout=(0.10, 0.5))
            augmenter.augment()

    else:  # if training letters exist
        print("Training dataset detected! " +
              "Skipping splitting & augmenting.")
        sleep(0.2)
        num_classes = len(os.listdir(letters_train_dir))

    if not os.path.exists(lines_train_dir):
        linemaker = Linemaker(set_type="train",
                              source_dir=letters_train_dir,
                              target_dir=lines_train_dir,
                              label_dir=label_train_path,
                              line_length_bounds=line_length_bounds,
                              n_lines=n_training_lines,
                              max_overlap=max_overlap_train,
                              cell_size=cell_size)

        max_h1, max_w1 = linemaker.make_lines()

        linemaker = Linemaker(set_type="test",
                              source_dir=letters_test_dir,
                              target_dir=lines_test_dir,
                              label_dir=label_test_dir,
                              line_length_bounds=line_length_bounds,
                              n_lines=n_testing_lines,
                              max_overlap=max_overlap_test,
                              cell_size=cell_size)

        max_h2, max_w2 = linemaker.make_lines()

        max_h = ceil(max(max_h1, max_h2)/float(cell_size))*cell_size
        max_w = ceil(max(max_w1, max_w2)/float(cell_size))*cell_size

        img_dims = (max_h, max_w)

        with open(dimensions_file, "w+") as filename:
            print(f"{max_h} {max_w}", file=filename)
        img_dims = (max_h, max_w)

    else:
        print("Line data detected! Skipping linemaking.")
        sleep(0.2)

        with open(dimensions_file, "r") as max_dimensions:
            img_h, img_w = [int(x) for x in max_dimensions.read().split()]
        img_dims = (img_h, img_w)

    if not os.path.isfile(os.path.normpath(lines_train_dir) + ".tfrecords"):
        print("Making tfrecords...")

        recordmaker = TfRecordMaker(imgs_dir=lines_train_dir,
                                    label_path=label_train_path)
        recordmaker.make_records()

        recordmaker = TfRecordMaker(imgs_dir=lines_test_dir,
                                    label_path=label_test_dir)
        recordmaker.make_records()

    else:
        print("Not creating TfRecords files because they already exist!")
        sleep(0.2)

    if not os.path.isfile(anchor_file):
        print("Making anchors...")
        anchormaker = AnchorMaker(target_file=anchor_file,
                                  label_path=label_train_path,
                                  cluster_num=cluster_num)
        anchormaker.make_anchors()

    else:
        print("Not creating anchors file because it already exists!")
        sleep(0.2)

    if not os.path.exists(weights_dir):
        print("Error: no weights detected! You need the pretrained " +
              f"weights in the {weights_dir} directory.")

    trainer = Trainer(num_classes=num_classes,
                      batch_size=batch_size,
                      steps=steps,
                      learning_rate=learning_rate,
                      decay_steps=decay_steps,
                      decay_rate=decay_rate,
                      n_filters_dn=n_filters_dn,
                      n_strides_dn=n_strides_dn,
                      n_ksizes_dn=n_ksizes_dn,
                      size_threshold=size_threshold,
                      ignore_threshold=ignore_threshold,
                      shuffle_size=shuffle_size,
                      eval_internal=eval_internal,
                      save_internal=save_internal,
                      print_every_n=print_every_n,
                      img_dims=img_dims,
                      cell_size=cell_size,
                      anchors_path=anchor_file,
                      train_records={os.path.normpath(lines_train_dir) +
                                     ".tfrecords"},
                      test_records={os.path.normpath(lines_test_dir) +
                                    ".tfrecords"},
                      checkpoint_path=checkpoint_dir)

    print("Training network...")
    trainer.train()

    network_exists = True

else:  # if network already exists and not retraining
    print("Network already trained!")
    sleep(0.2)

    with open(dimensions_file, "r") as max_dimensions:
        img_h, img_w = [int(x) for x in max_dimensions.read().split()]
    img_dims = (img_h, img_w)
    num_classes = len(os.listdir(letters_train_dir))


if network_exists and show_tfrecord_example:
    example_displayer = ExampleDisplayer(
        source_dir=os.path.normpath(lines_train_dir) + ".tfrecords",
        img_dims=img_dims,
        anchor_dir=anchor_file,
        num_classes=num_classes,
        cell_size=cell_size)

    example_displayer.show_example()

if network_exists and test_example:

    weightconverter = WeightConverter(
        freeze=True,
        num_classes=num_classes,
        checkpoint_dir=checkpoint_dir,
        img_dims=img_dims,
        weights_dir=weights_dir,
        n_filters_dn=n_filters_dn,
        n_strides_dn=n_strides_dn,
        n_ksizes_dn=n_ksizes_dn,
        anchors_path=anchor_file,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        convert=False,
        checkpoint_step=steps - (steps % save_internal))

    weightconverter.convert_weights()

    rescale_test = False
    if test_on_scrolls:
        source_dir = processed_image_dir
        rescale_test = True
    elif test_on_train:
        source_dir = lines_train_dir
    else:
        source_dir = lines_test_dir

    tester = Tester(source_dir=source_dir,
                    num_classes=num_classes,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                    size_threshold=size_threshold,
                    img_dims=img_dims,
                    checkpoint_dir=checkpoint_dir,
                    letters_test_dir=letters_train_dir,
                    max_boxes=max_boxes,
                    filters=filters,
                    rescale_test=rescale_test)
    results = tester.test()

    print('\n'*5)

    if not results:
        print("No characters were detected!")
    else:
        [print(f"x={int(x)}:\t{c}\t(p = {p:.3f})") for (x, c, p) in results]

    print('\n'*5)

    def convert_to_uni(name):
        hebrew = {
            'Alef':         u'\u05D0',
            'Ayin':         u'\u05E2',
            'Bet':          u'\u05D1',
            'Dalet':        u'\u05D3',
            'Gimel':        u'\u05D2',
            'He':           u'\u05D4',
            'Het':          u'\u05D7',
            'Kaf':          u'\u05DB',
            'Kaf-final':    u'\u05DA',
            'Lamed':        u'\u05DC',
            'Mem':          u'\u05DD',
            'Mem-medial':   u'\u05DE',
            'Nun-final':    u'\u05DF',
            'Nun-medial':   u'\u05E0',
            'Pe':           u'\u05E4',
            'Pe-final':     u'\u05E3',
            'Qof':          u'\u05E7',
            'Resh':         u'\u05E8',
            'Samekh':       u'\u05E1',
            'Shin':         u'\u05E9',
            'Taw':          u'\u05EA',
            'Tet':          u'\u05D8',
            'Tsadi-final':  u'\u05E5',
            'Tsadi-medial': u'\u05E6',
            'Waw':          u'\u05D5',
            'Yod':          u'\u05D9',
            'Zayin':        u'\u05D6'

        }
        if name in hebrew:
            return hebrew[name]
        else:
            return '?'

    print(''.join([convert_to_uni(c) for (_, c, _) in results]))

    print('\n'*5)
