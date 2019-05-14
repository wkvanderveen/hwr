#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : debug.py
#   Author      : YunYang1994
#   Created date: 2019-01-21 15:02:05
#   Description :
#
#================================================================

import cv2
import os
import numpy as np
import tensorflow as tf
from core import utils
from PIL import Image
from core.dataset import Parser, dataset

class ExampleDisplayer(object):
    """docstring for ExampleDisplayer"""
    def __init__(self, source_dir, img_dims, anchor_dir, num_classes):
        super(ExampleDisplayer, self).__init__()
        self.source_dir = source_dir
        self.anchor_dir = anchor_dir
        self.num_classes = num_classes
        self.img_h = img_dims[0]
        self.img_w = img_dims[1]

    def show_example(self):
        sess = tf.Session()
        classes = os.listdir(self.source_dir[:-len(".tfrecords") or None])

        train_tfrecord = self.source_dir
        anchors        = utils.get_anchors(self.anchor_dir, self.img_h, self.img_w)

        parser   = Parser(self.img_h, self.img_w, anchors, self.num_classes, debug=True)
        trainset = dataset(parser, train_tfrecord, 1, shuffle=1)

        is_training = tf.placeholder(tf.bool)
        example = trainset.get_next()

        image, boxes = sess.run(example)
        image, boxes = image[0], boxes[0]

        n_box = len(boxes)
        for i in range(n_box):
            image = cv2.rectangle(image,(int(float(boxes[i][0])),
                                         int(float(boxes[i][1]))),
                                        (int(float(boxes[i][2])),
                                         int(float(boxes[i][3]))), (255,0,0), 1)
            label = classes[int(float(boxes[i][4]))]
            image = cv2.putText(image, label, (int(float(boxes[i][0])),int(float(boxes[i][1]))),
                                cv2.FONT_HERSHEY_SIMPLEX,  .6, (0, 255, 0), 1, 2)

        image = Image.fromarray(np.uint8(image))
        image.show()


if __name__ == "__main__":
    displayer = ExampleDisplayer(source_dir="../../data/lines-train.tfrecords",
                                 dims_dir="../../data/max_wh.txt")
    displayer.show_example()
