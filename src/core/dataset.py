import cv2
import numpy as np
from core import utils
import tensorflow as tf


class Parser(object):
    def __init__(self, image_h, image_w, anchors, cell_size, num_classes,
                 debug=False):

        self.anchors = anchors
        self.num_classes = num_classes
        self.image_h = image_h
        self.image_w = image_w
        self.debug = debug
        self.cell_size = cell_size

    def flip_left_right(self, image, gt_boxes):

        w = tf.cast(tf.shape(image)[1], tf.float32)
        image = tf.image.flip_left_right(image)

        xmin, ymin, xmax, ymax, label = tf.unstack(gt_boxes, axis=1)
        xmin, ymin, xmax, ymax = w-xmax, ymin, w-xmin, ymax
        gt_boxes = tf.stack([xmin, ymin, xmax, ymax, label], axis=1)

        return image, gt_boxes

    def random_distort_color(self, image, gt_boxes):

        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        return image, gt_boxes

    def random_blur(self, image, gt_boxes):

        gaussian_blur = lambda image: cv2.GaussianBlur(image, (5, 5), 0)
        h, w = image.shape.as_list()[:2]
        image = tf.py_func(gaussian_blur, [image], tf.uint8)
        image.set_shape([h, w, 1])

        return image, gt_boxes

    def random_crop(self, image, gt_boxes, min_object_covered=0.8,
                    aspect_ratio_range=[0.8, 1.2], area_range=[0.5, 1.0]):

        h = tf.cast(tf.shape(image)[0], tf.float32)
        w = tf.cast(tf.shape(image)[1], tf.float32)

        xmin, ymin, xmax, ymax, label = tf.unstack(gt_boxes, axis=1)

        bboxes = tf.stack([ymin/h, xmin/w, ymax/h, xmax/w], axis=1)

        bboxes = tf.clip_by_value(bboxes, 0, 1)

        begin, size, dist_boxes = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(bboxes, axis=0),
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range)

        # NOTE dist_boxes with shape: [ymin, xmin, ymax, xmax]
        # and in values in range(0, 1)
        # Employ the bounding box to distort the image.
        cropped_box = [dist_boxes[0, 0, 1] * w,
                       dist_boxes[0, 0, 0] * h,
                       dist_boxes[0, 0, 3] * w,
                       dist_boxes[0, 0, 2] * h]

        cropped_xmin = tf.clip_by_value(xmin, cropped_box[0], cropped_box[2]) \
            - cropped_box[0]
        cropped_ymin = tf.clip_by_value(ymin, cropped_box[1], cropped_box[3]) \
            - cropped_box[1]
        cropped_xmax = tf.clip_by_value(xmax, cropped_box[0], cropped_box[2]) \
            - cropped_box[0]
        cropped_ymax = tf.clip_by_value(ymax, cropped_box[1], cropped_box[3]) \
            - cropped_box[1]

        image = tf.slice(image, begin, size)
        gt_boxes = tf.stack([cropped_xmin,
                             cropped_ymin,
                             cropped_xmax,
                             cropped_ymax,
                             label],
                            axis=1)

        return image, gt_boxes

    def preprocess(self, image, gt_boxes):

        image, gt_boxes = utils.resize_image_correct_bbox(image,
                                                          gt_boxes,
                                                          self.image_h,
                                                          self.image_w)

        if self.debug:
            return image, gt_boxes

        y_true = tf.py_func(self.preprocess_true_boxes,
                            inp=[gt_boxes],
                            Tout=[tf.float32])

        return image, y_true

    def preprocess_true_boxes(self, gt_boxes):

        anchor_mask = list(range(0, len(self.anchors)))
        grid_sizes = [self.image_h // self.cell_size,
                      self.image_w // self.cell_size]

        box_centers = (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) / 2  # center
        box_sizes = gt_boxes[:, 2:4] - gt_boxes[:, 0:2]  # h & w

        gt_boxes[:, 0:2] = box_centers
        gt_boxes[:, 2:4] = box_sizes

        y_true = np.zeros(shape=[grid_sizes[0],
                                 grid_sizes[1],
                                 len(self.anchors),
                                 5 + self.num_classes],
                          dtype=np.float32)

        anchors_max = self.anchors / 2.
        anchors_min = -anchors_max
        valid_mask = box_sizes[:, 0] > 0

        # Discard zero rows.
        wh = box_sizes[valid_mask]
        # set the center of all boxes as the origin of their coordinates
        # and correct their coordinates
        wh = np.expand_dims(wh, -2)
        boxes_max = wh / 2.
        boxes_min = -boxes_max

        intersect_mins = np.maximum(boxes_min, anchors_min)
        intersect_maxs = np.minimum(boxes_max, anchors_max)
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]

        anchor_area = self.anchors[:, 0] * self.anchors[:, 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            if n not in anchor_mask:
                continue

            i = np.floor(gt_boxes[t, 0]
                         / self.image_w*grid_sizes[1]).astype('int32')
            j = np.floor(gt_boxes[t, 1]
                         / self.image_h*grid_sizes[0]).astype('int32')

            k = anchor_mask.index(n)
            c = gt_boxes[t, 4].astype('int32')

            y_true[j, i, k, 0:4] = gt_boxes[t, 0:4]
            y_true[j, i, k,   4] = 1.
            y_true[j, i, k, 5+c] = 1.

        return y_true

    def parser_example(self, serialized_example):

        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], dtype=tf.string),
                'boxes': tf.FixedLenFeature([], dtype=tf.string),
            }
        )

        image = tf.image.decode_jpeg(features['image'], channels=1)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        gt_boxes = tf.decode_raw(features['boxes'], tf.float32)
        gt_boxes = tf.reshape(gt_boxes, shape=[-1, 5])

        return self.preprocess(image, gt_boxes)


class dataset(object):
    def __init__(self, parser, tfrecords_path, batch_size, shuffle=None,
                 repeat=True):
        self.parser = parser
        self.filenames = tf.gfile.Glob(tfrecords_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        self._buildup()

    def _buildup(self):
        try:
            self._TFRecordDataset = tf.data.TFRecordDataset(self.filenames)
        except NotImplementedError:
            raise NotImplementedError("No tfrecords found!")

        self._TFRecordDataset = self._TFRecordDataset.map(
            map_func=self.parser.parser_example,
            num_parallel_calls=10)

        self._TFRecordDataset = self._TFRecordDataset.repeat() \
            if self.repeat else self._TFRecordDataset

        if self.shuffle is not None:
            self._TFRecordDataset = self._TFRecordDataset.shuffle(self.shuffle)

        self._TFRecordDataset = self._TFRecordDataset.batch(self.batch_size)\
            .prefetch(self.batch_size)
        self._iterator = self._TFRecordDataset.make_one_shot_iterator()

    def get_next(self):
        return self._iterator.get_next()
