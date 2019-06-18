import tensorflow as tf
from core import common
slim = tf.contrib.slim


class darknet53(object):
    """network for performing feature extraction"""

    def __init__(self, inputs, n_filters, n_strides, n_ksizes):
        self.outputs = self.forward(inputs, n_filters, n_strides, n_ksizes)

    def forward(self, inputs, n_filters, n_strides, n_ksizes):

        for i in range(min(len(n_filters), len(n_strides)), len(n_ksizes)):
            inputs = common._conv2d_fixed_padding(inputs,
                                                  filters=n_filters[i],
                                                  kernel_size=n_ksizes[i],
                                                  strides=n_strides[i])

        return inputs


class yolov3(object):

    def __init__(self, num_classes, anchors,
                 batch_norm_decay=0.9, leaky_relu=0.1):

        self._ANCHORS = anchors
        self._BATCH_NORM_DECAY = batch_norm_decay
        self._LEAKY_RELU = leaky_relu
        self._NUM_CLASSES = num_classes
        self.feature_maps = []

    def _yolo_block(self, inputs, filters, ksizes, strides):

        for i in range(min(len(filters), len(ksizes))):
            inputs = common._conv2d_fixed_padding(inputs,
                                                  filters=filters[i],
                                                  kernel_size=ksizes[i],
                                                  strides=strides[i])

        return inputs

    def _detection_layer(self, inputs, anchors):
        num_anchors = len(anchors)

        feature_map = slim.conv2d(inputs,
                                  num_anchors * (5 + self._NUM_CLASSES),
                                  kernel_size=1,
                                  stride=1,
                                  normalizer_fn=None,
                                  activation_fn=None,
                                  biases_initializer=tf.zeros_initializer())

        return feature_map

    def _reorg_layer(self, feature_map, anchors):

        num_anchors = len(anchors)
        grid_size = feature_map.shape.as_list()[1:3]
        # the downscale image in height and weight
        # [h,w] -> [y,x]
        stride = tf.cast(self.img_size // grid_size, tf.float32)
        feature_map = tf.reshape(feature_map,
                                 [-1,
                                  grid_size[0],
                                  grid_size[1],
                                  num_anchors,
                                  5 + self._NUM_CLASSES])

        box_centers, box_sizes, conf_logits, prob_logits = tf.split(
            feature_map, [2, 2, 1, self._NUM_CLASSES], axis=-1)

        box_centers = tf.nn.sigmoid(box_centers)

        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)

        a, b = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])
        x_y_offset = tf.cast(x_y_offset, tf.float32)

        box_centers = box_centers + x_y_offset
        box_centers = box_centers * stride[::-1]

        box_sizes = tf.exp(box_sizes) * anchors  # anchors -> [w, h]
        boxes = tf.concat([box_centers, box_sizes], axis=-1)
        return x_y_offset, boxes, conf_logits, prob_logits

    def forward(self, inputs, n_filters_dn, n_strides_dn, n_ksizes_dn,
                n_filt_yolo, ksizes_yolo, n_strides_yolo, is_training=False,
                reuse=False):

        self.img_size = tf.shape(inputs)[1:3]

        batch_norm_params = {
            'decay': self._BATCH_NORM_DECAY,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        # Set activation_fn and parameters for conv2d, batch_norm.
        with slim.arg_scope([slim.conv2d,
                             slim.batch_norm,
                             common._fixed_padding],
                            reuse=reuse):

            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(
                                    x, alpha=self._LEAKY_RELU)):

                with tf.variable_scope('darknet-53'):

                    # Convolve in darknet
                    inputs = darknet53(inputs,
                                       n_filters=n_filters_dn,
                                       n_strides=n_strides_dn,
                                       n_ksizes=n_ksizes_dn).outputs

                with tf.variable_scope('yolo-v3'):

                    # Convolve in yolo
                    inputs = self._yolo_block(inputs,
                                              filters=n_filt_yolo,
                                              ksizes=ksizes_yolo,
                                              strides=n_strides_yolo)

                    feature_map = self._detection_layer(inputs, self._ANCHORS)
                    feature_map = tf.identity(feature_map, name='feature_map')

            return feature_map

    def _reshape(self, x_y_offset, boxes, confs, probs):

        grid_size = x_y_offset.shape.as_list()[:2]

        boxes = tf.reshape(boxes,
                           [-1,
                            grid_size[0] * grid_size[1] * len(self._ANCHORS),
                            4])

        confs = tf.reshape(confs,
                           [-1,
                            grid_size[0] * grid_size[1] * len(self._ANCHORS),
                            1])

        probs = tf.reshape(probs,
                           [-1,
                            grid_size[0] * grid_size[1] * len(self._ANCHORS),
                            self._NUM_CLASSES])

        return boxes, confs, probs

    def predict(self, feature_map):
        """
        Note: given by feature_map, compute the receptive field
              and get boxes, confs and class_probs
        """
        feature_map_anchors = [(feature_map, self._ANCHORS)]

        results = [self._reorg_layer(feature_map, anchors)
                   for (feature_map, anchors) in feature_map_anchors]

        boxes_list, confs_list, probs_list = [], [], []

        for result in results:
            boxes, conf_logits, prob_logits = self._reshape(*result)

            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)

            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        boxes = tf.concat(boxes_list, axis=1)
        confs = tf.concat(confs_list, axis=1)
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes,
                                                     [1, 1, 1, 1],
                                                     axis=-1)
        x0 = center_x - width / 2.
        y0 = center_y - height / 2.
        x1 = center_x + width / 2.
        y1 = center_y + height / 2.

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        return boxes, confs, probs

    def compute_loss(self, pred_feature_map, y_true, iou_threshold):

        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        total_loss = 0.

        result = self.loss_layer(
            pred_feature_map, y_true, self._ANCHORS, iou_threshold)

        loss_xy += result[0]
        loss_wh += result[1]
        loss_conf += result[2]
        loss_class += result[3]

        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

    def loss_layer(self, feature_map, y_true, anchors, iou_threshold):
        # size in [h, w] format! don't get messed up!

        grid_size = tf.shape(feature_map)[1:3]
        grid_size_ = feature_map.shape.as_list()[1:3]

        y_true = tf.reshape(y_true, [-1,
                                     grid_size_[0],
                                     grid_size_[1],
                                     len(self._ANCHORS),
                                     5+self._NUM_CLASSES])

        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)

        # N: batch_size
        N = tf.cast(tf.shape(feature_map)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = \
            self._reorg_layer(feature_map, anchors)

        object_mask = y_true[..., 4:5]

        valid_true_boxes = tf.boolean_mask(y_true[..., 0:4],
                                           tf.cast(object_mask[..., 0],
                                                   'bool'))

        valid_true_box_xy = valid_true_boxes[:, 0:2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]

        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # calc iou

        iou = self._broadcast_iou(valid_true_box_xy,
                                  valid_true_box_wh,
                                  pred_box_xy,
                                  pred_box_wh)

        best_iou = tf.reduce_max(iou, axis=-1)

        # get_ignore_mask
        ignore_mask = tf.cast(best_iou < iou_threshold, tf.float32)

        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # get_tw_th, numerical range: 0 ~ 1

        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors

        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)

        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment:
        # box with smaller area has bigger weight.

        box_loss_scale = 2. - ((y_true[..., 2:3]
                                / tf.cast(self.img_size[1], tf.float32))
                               * (y_true[..., 3:4]
                                  / tf.cast(self.img_size[0], tf.float32)))

        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy)
                                * object_mask * box_loss_scale) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th)
                                * object_mask * box_loss_scale) / N

        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask

        conf_loss_pos = conf_pos_mask \
            * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                      logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask \
            * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                      logits=pred_conf_logits)

        conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / N

        class_loss = object_mask \
            * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:],
                                                      logits=pred_prob_logits)

        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    def _broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy,
                       pred_box_wh):
        '''
        maintain an efficient way to calculate the ios matrix between
        ground truth true boxes and the predicted boxes
        note: here we only care about the size match
        '''

        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)

        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]

        iou = intersect_area / (pred_box_area + true_box_area - intersect_area)

        return iou
