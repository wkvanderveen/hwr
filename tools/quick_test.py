#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : fuck.py
#   Author      : YunYang1994
#   Created date: 2019-01-23 10:21:50
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils
import os


classes = os.listdir('../../data/letters-test/')
num_classes = len(classes)
with open("../../data/max_wh.txt", "r") as max_dimensions:
    dimensions_string = max_dimensions.read()
IMAGE_W, IMAGE_H = [int(x) for x in dimensions_string.split()]

image_path = "../../data/lines-test/0.jpeg"
img = Image.open(image_path)
img = img.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)

cpu_nms_graph = tf.Graph()

input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, "../../data/checkpoint/yolov3_cpu_nms.pb",
                                           ["Placeholder:0", "concat_9:0", "mul_6:0"])
with tf.Session(graph=cpu_nms_graph) as sess:
    boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img, axis=0)})
    boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.01, iou_thresh=0.01)
    image = utils.draw_boxes(img, boxes, scores, labels, classes, [IMAGE_H, IMAGE_W], show=True)
