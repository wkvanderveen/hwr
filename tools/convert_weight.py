import os
import sys
import wget
import time
import argparse
import tensorflow as tf
from core import yolov3, utils

class WeightConverter(object):
    """docstring for WeightConverter"""
    def __init__(self, freeze, convert, num_classes, img_dims, checkpoint_dir, weights_dir, anchors_path, score_threshold, iou_threshold, checkpoint_step=None):
        super(WeightConverter, self).__init__()
        self.freeze = freeze
        self.convert = convert
        self.num_classes = num_classes
        self.anchors_path = anchors_path
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.img_h = img_dims[0]
        self.img_w = img_dims[1]
        self.checkpoint_step = checkpoint_step

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_dir = os.path.join(checkpoint_dir, "yolov3.ckpt")

        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        self.weights_dir = os.path.join(weights_dir, "yolov3.weights")

    def convert_weights(self):
        # flags = parser(description="freeze yolov3 graph from checkpoint file").parse_args()
        print(f"=> the input image size is [{self.img_h}, {self.img_w}]")
        anchors = utils.get_anchors(self.anchors_path, self.img_h, self.img_w)
        model = yolov3.yolov3(self.num_classes, anchors)

        with tf.Graph().as_default() as graph:
            sess = tf.Session(graph=graph)
            inputs = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, 3]) # placeholder for detector inputs
            print("=>", inputs)

            with tf.variable_scope('yolov3'):
                feature_map = model.forward(inputs, is_training=False)

            boxes, confs, probs = model.predict(feature_map)
            scores = confs * probs
            print("=>", boxes.name[:-2], scores.name[:-2])
            cpu_out_node_names = [boxes.name[:-2], scores.name[:-2]]
            boxes, scores, labels = utils.gpu_nms(boxes, scores, self.num_classes,
                                                  score_thresh=self.score_threshold,
                                                  iou_thresh=self.iou_threshold)
            print("=>", boxes.name[:-2], scores.name[:-2], labels.name[:-2])
            gpu_out_node_names = [boxes.name[:-2], scores.name[:-2], labels.name[:-2]]
            feature_map_1, feature_map_2, feature_map_3 = feature_map
            saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

            if self.convert:
                if not os.path.exists(self.weights_dir):
                    url = 'https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3.weights'
                    print(f"=> {self.weights_dir} does not exist!")
                    print(f"=> It will take a while to download it from {url}")
                    print('=> Downloading yolov3 weights ... ')
                    wget.download(url, self.weights_dir)

                load_ops = utils.load_weights(tf.global_variables(scope='yolov3'), self.weights_dir)
                sess.run(load_ops)
                save_path = saver.save(sess, save_path=self.checkpoint_dir)
                print(f'=> model saved in path: {save_path}')

            if self.freeze:
                ckpt_idx = self.checkpoint_dir + '-' + str(self.checkpoint_step)
                saver.restore(sess, ckpt_idx)
                print('=> checkpoint file restored from ', ckpt_idx)
                utils.freeze_graph(sess, '../../data/checkpoint/yolov3_cpu_nms.pb', cpu_out_node_names)
                utils.freeze_graph(sess, '../../data/checkpoint/yolov3_gpu_nms.pb', gpu_out_node_names)

if __name__ == "__main__":
    weightconverter = WeightConverter(freeze=True,
                                      num_classes=27,
                                      dimensions_path="../../data/max_wh.txt",
                                      checkpoint_dir="../../data/checkpoint/",
                                      weights_dir="../../data/weights")
