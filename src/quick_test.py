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
import cv2
from core import utils
import os
from numpy import expand_dims
from random import choice

class Tester(object):
    """docstring for Tester"""
    def __init__(self, img_dims, num_classes, source_dir, score_threshold,
        iou_threshold, size_threshold, checkpoint_dir, letters_test_dir,
        max_boxes, remove_overlap_half, remove_overlap_full):

        super(Tester, self).__init__()
        self.img_h = img_dims[0]
        self.img_w = img_dims[1]
        self.num_classes = num_classes
        self.source_dir = source_dir
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.size_threshold = size_threshold
        self.checkpoint_dir = checkpoint_dir
        self.letters_test_dir = letters_test_dir
        self.max_boxes = max_boxes
        self.remove_overlap_half = remove_overlap_half
        self.remove_overlap_full = remove_overlap_full

    def test(self):
        image_name = choice(os.listdir(self.source_dir))
        image_path = os.path.join(self.source_dir, image_name)

        img = cv2.imread(image_path, 0)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = expand_dims(img, axis=2)

        classes = os.listdir(self.letters_test_dir)

        cpu_nms_graph = tf.Graph()

        input_tensor, output_tensors = utils.read_pb_return_tensors(
            cpu_nms_graph,
            os.path.join(self.checkpoint_dir, "yolov3_cpu_nms.pb"),
            ["Placeholder:0", "concat_5:0", "mul_2:0"])

        with tf.Session(graph=cpu_nms_graph) as sess:
            boxes, scores = sess.run(
                output_tensors,
                feed_dict={input_tensor: np.expand_dims(img, axis=0)})
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1, self.num_classes)

            mask = np.logical_and(boxes[:,0] >= 0, boxes[:,0] <= img.shape[1])
            mask = np.logical_and(mask, boxes[:,1] >= 0)
            mask = np.logical_and(mask, boxes[:,2] >= 0)
            mask = np.logical_and(mask, boxes[:,3] >= 0)
            mask = np.logical_and(mask, boxes[:,1] <= img.shape[0])
            mask = np.logical_and(mask, boxes[:,2] <= img.shape[1])
            mask = np.logical_and(mask, boxes[:,3] <= img.shape[0])
            mask = np.logical_and(mask, boxes[:,0] < boxes[:,2])
            mask = np.logical_and(mask, boxes[:,1] < boxes[:,3])
            mask = np.logical_and(mask, abs(boxes[:,2]-boxes[:,0]) >= self.size_threshold[0])
            mask = np.logical_and(mask, abs(boxes[:,3]-boxes[:,1]) >= self.size_threshold[1])

            boxes = boxes[mask]
            scores = scores[mask]

            boxes, scores, labels = utils.cpu_nms(
                boxes=boxes,
                scores=scores,
                num_classes=self.num_classes,
                score_thresh=self.score_threshold,
                iou_thresh=self.iou_threshold,
                max_boxes=self.max_boxes)

            (image, results) = utils.draw_boxes(
                img,
                boxes,
                scores,
                labels,
                classes,
                [self.img_h, self.img_w],
                show=True,
                size_threshold=self.size_threshold,
                remove_overlap_half=self.remove_overlap_half,
                remove_overlap_full=self.remove_overlap_full)

        return results
