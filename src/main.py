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
from numpy import prod, asarray

## PARAMETERS ##

# File structure parameters
orig_letters_dir = "../../data/original_letters/"
letters_train_dir = "../../data/letters-train/"
letters_test_dir = "../../data/letters-test/"
lines_train_dir = "../../data/lines-train/"
lines_test_dir = "../../data/lines-test/"
label_train_path = "../../data/labels-train.txt"
label_test_dir = "../../data/labels-test.txt"
checkpoint_dir = "../../data/checkpoint/"
dimensions_file = "../../data/dimensions.txt"
weights_dir = "../../data/weights/"
anchor_file = "../../data/anchors.txt"

# Data parameters
num_classes = 10
split_percentage = 20
line_length_bounds = (15,20)
n_training_lines = 500
n_testing_lines = 50
max_overlap_train = 5
max_overlap_test = 5
max_boxes = 20
test_on_train = False


# Network parameters
n_filters_dn = (32,)
n_strides_dn = (3,)
n_ksizes_dn = (5,)
cell_size = 1

n_filt_yolo = (8, 16, 32)
n_ksizes_yolo = (1,1,1)
cluster_num = 8
iou_threshold = 0.0
score_threshold = 0.0
ignore_threshold = 0.0
size_threshold = (1,1)  # in pixels
remove_overlap_half = True
remove_overlap_full = True  # redundant if `remove_overlap_half == True'

batch_size = 1
steps = 220
learning_rate = 1e-2
decay_steps = 50
decay_rate = 0.5
shuffle_size = 10
eval_internal = 20
save_internal = 20
print_every_n = 10


# Other parameters
retrain = True
show_tfrecord_example = False
test_example = True


# [preprocessing here]

# PREPARE NETWORK IF NOT READY
network_exists = bool(os.path.isfile("../../data/checkpoint/checkpoint"))

if not network_exists or retrain:

    if not os.path.exists(letters_train_dir):
        splitter = Splitter(source_dir=orig_letters_dir,
                            num_classes=num_classes,
                            train_dir=letters_train_dir,
                            test_dir=letters_test_dir,
                            percentage=split_percentage)
        print(f"Splitting {splitter.percentage}% of the data "+
              f"found in {splitter.source_dir}...")

        sleep(0.2)
        num_classes = splitter.split()

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

    weightconverter = WeightConverter(freeze=False,
                                      convert=False,
                                      num_classes=num_classes,
                                      img_dims=img_dims,
                                      checkpoint_dir=checkpoint_dir,
                                      weights_dir=weights_dir,
                                      anchors_path=anchor_file,
                                      score_threshold=score_threshold,
                                      iou_threshold=iou_threshold,
                                      n_filt_yolo=n_filt_yolo,
                                      ksizes_yolo=n_ksizes_yolo,
                                      n_filters_dn=n_filters_dn,
                                      n_strides_dn=n_strides_dn,
                                      n_ksizes_dn=n_ksizes_dn)
    print("Converting weights...")
    weightconverter.convert_weights()

    trainer = Trainer(num_classes=num_classes,
                      batch_size=batch_size,
                      steps=steps,
                      learning_rate=learning_rate,
                      decay_steps=decay_steps,
                      decay_rate=decay_rate,
                      n_filters_dn=n_filters_dn,
                      n_strides_dn=n_strides_dn,
                      n_ksizes_dn=n_ksizes_dn,
                      n_filt_yolo=n_filt_yolo,
                      ksizes_yolo=n_ksizes_yolo,
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

else: # if network already exists and not retraining
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
        n_filt_yolo=n_filt_yolo,
        ksizes_yolo=n_ksizes_yolo,
        anchors_path=anchor_file,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        convert=False,
        checkpoint_step=steps-(steps%save_internal))

    weightconverter.convert_weights()

    tester = Tester(source_dir=(lines_train_dir if test_on_train else lines_test_dir),
                    num_classes=num_classes,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                    size_threshold=size_threshold,
                    img_dims=img_dims,
                    checkpoint_dir=checkpoint_dir,
                    letters_test_dir=(letters_train_dir if test_on_train else letters_test_dir),
                    max_boxes=max_boxes,
                    remove_overlap_half=remove_overlap_half,
                    remove_overlap_full=remove_overlap_full)
    results = tester.test()

    print('\n'*5)

    if not results:
        print("No characters were detected!")
    else:
        [print(f"x={int(x)}:\t{c}\t(p = {p:.3f})") for (x,c,p) in results]

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

    print(''.join([convert_to_uni(c) for (_,c,_) in results]))

    print('\n'*5)



# [postprocessing here]


"""
TODO:
* Train on the data to see if YOLO works
* Add preprocessing
* Add postprocessing & writer
"""
