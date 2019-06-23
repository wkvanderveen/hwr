import numpy as np
import tensorflow as tf
import cv2
from core import utils
import os
from numpy import expand_dims
from random import choice

# tf.logging.set_verbosity(tf.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


class Tester(object):
    """docstring for Tester"""
    def __init__(self, img_dims, num_classes, size_threshold, checkpoint_dir,
                 letters_test_dir, max_boxes, filters, orig_letters):

        super(Tester, self).__init__()
        self.img_h = img_dims[0]
        self.img_w = img_dims[1]
        self.num_classes = num_classes
        self.filters = filters
        self.size_threshold = size_threshold
        self.checkpoint_dir = checkpoint_dir
        self.letters_test_dir = letters_test_dir
        self.max_boxes = max_boxes
        self.orig_letters = orig_letters

    def test(self, source_dir=None, image_path=None, show=False):
        if source_dir:
            image_name = choice(os.listdir(source_dir))
            image_path = os.path.join(source_dir, image_name)

        img = cv2.imread(image_path, 0)
        img = cv2.resize(src=img, dsize=(0, 0), fx=1/2, fy=1/2)
        if img.shape[0] < self.img_h and img.shape[1] < self.img_w:
            img = cv2.copyMakeBorder(src=img,
                                     top=(self.img_h-img.shape[0])//2,
                                     bottom=(self.img_h-img.shape[0])//2,
                                     left=(self.img_w-img.shape[1])//2,
                                     right=(self.img_w-img.shape[1])//2,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=[255, 255, 255])

        img = cv2.resize(img, (self.img_w, self.img_h))
        img = expand_dims(img, axis=2)

        classes = os.listdir(self.orig_letters)

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

            min_wh, max_wh = -10000, 10000
            min_ratio = 1/4  # 0 -- 1

            mask = np.logical_and(boxes[:, 0] >= min_wh, boxes[:, 0] <= max_wh)
            mask = np.logical_and(mask, boxes[:, 1] >= min_wh)
            mask = np.logical_and(mask, boxes[:, 2] >= min_wh)
            mask = np.logical_and(mask, boxes[:, 3] >= min_wh)
            mask = np.logical_and(mask, boxes[:, 1] <= max_wh)
            mask = np.logical_and(mask, boxes[:, 2] <= max_wh)
            mask = np.logical_and(mask, boxes[:, 3] <= max_wh)
            mask = np.logical_and(mask, boxes[:, 0] < boxes[:, 2])
            mask = np.logical_and(mask, boxes[:, 1] < boxes[:, 3])

            boxes = boxes[mask]
            scores = scores[mask]

            h = abs(boxes[:, 2]-boxes[:, 0])
            w = abs(boxes[:, 3]-boxes[:, 1])

            mask = np.logical_and(w/h > min_ratio, h/w > min_ratio)

            boxes = boxes[mask]
            scores = scores[mask]

            if self.filters:
                # Harder filters
                print(f"Test: Boxes before filtering:\t{boxes.shape[0]}")

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

                print(f"Test: Boxes after filtering:\t{boxes.shape[0]}")

                if boxes.shape[0] == 0:
                    print(f"Try changing the filters/thresholds in the parameters.")

            boxes, scores, labels = utils.cpu_nms(
                boxes=boxes,
                scores=scores,
                num_classes=self.num_classes,
                max_boxes=self.max_boxes)

            (image, results) = utils.draw_boxes(
                img,
                boxes,
                scores,
                labels,
                classes,
                [self.img_h, self.img_w],
                show=show,
                size_threshold=self.size_threshold)

        return results
