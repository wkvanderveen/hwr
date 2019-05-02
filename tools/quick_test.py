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


classes = os.listdir('../../data/test/')
num_classes = len(classes)

image_path = "../../data/train/Alef/navis-QIrug-Qumran_extr09_0709-line-006-y1=602-y2=818-zone-HUMAN-x=0527-y=0119-w=0037-h=0050-ybas=0148-nink=855-segm=COCOS5cocos.jpg"
img = Image.open(image_path)
cpu_nms_graph = tf.Graph()

input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, "../../data/checkpoint/yolov3_cpu_nms.pb",
                                           ["Placeholder:0", "concat_9:0", "mul_6:0"])
with tf.Session(graph=cpu_nms_graph) as sess:
    boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img, axis=0)})
    boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.3, iou_thresh=0.5)
    image = utils.draw_boxes(img, boxes, scores, labels, classes, [IMAGE_H, IMAGE_W], show=True)
