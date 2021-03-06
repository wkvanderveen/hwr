import os
import tensorflow as tf
from core import yolov3, utils

class WeightConverter(object):
    """docstring for WeightConverter"""
    def __init__(self, freeze, convert, num_classes, n_filters_dn,
                 n_strides_dn, n_ksizes_dn, img_dims, checkpoint_dir,
                 weights_dir, anchors_path, checkpoint_step=None):
        super(WeightConverter, self).__init__()
        self.freeze = freeze
        self.convert = convert
        self.num_classes = num_classes
        self.anchors_path = anchors_path
        self.img_h = img_dims[0]
        self.img_w = img_dims[1]
        self.n_filters_dn = n_filters_dn
        self.n_strides_dn = n_strides_dn
        self.n_ksizes_dn = n_ksizes_dn
        self.checkpoint_step = checkpoint_step

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_dir = os.path.join(checkpoint_dir, "yolov3.ckpt")

        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        self.weights_dir = os.path.join(weights_dir, "yolov3.weights")

    def convert_weights(self):
        print(f"=> the input image size is [{self.img_h}, {self.img_w}]")
        anchors = utils.get_anchors(self.anchors_path, self.img_h, self.img_w)
        model = yolov3.yolov3(self.num_classes, anchors)

        with tf.Graph().as_default() as graph:
            sess = tf.Session(graph=graph)
            inputs = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, 1]) # placeholder for detector inputs
            print("=>", inputs)

            with tf.variable_scope('yolov3'):
                feature_map = model.forward(inputs,
                                            n_filters_dn=self.n_filters_dn,
                                            n_strides_dn=self.n_strides_dn,
                                            n_ksizes_dn=self.n_ksizes_dn,
                                            is_training=False)

            boxes, confs, probs = model.predict(feature_map)
            scores = confs * probs
            print("=>", boxes.name[:-2], scores.name[:-2])
            cpu_out_node_names = [boxes.name[:-2], scores.name[:-2]]
            boxes, scores, labels = utils.gpu_nms(boxes, scores, self.num_classes)
            print("=>", boxes.name[:-2], scores.name[:-2], labels.name[:-2])
            gpu_out_node_names = [boxes.name[:-2], scores.name[:-2], labels.name[:-2]]

            saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

            if self.convert:
                load_ops = utils.load_weights(tf.global_variables(scope='yolov3'), self.weights_dir)
                sess.run(load_ops)
                save_path = saver.save(sess, save_path=self.checkpoint_dir)
                print(f'=> model saved in path: {save_path}')

            if self.freeze:
                ckpt_idx = self.checkpoint_dir + '-' + str(self.checkpoint_step)
                try:
                    saver.restore(sess, ckpt_idx)
                except:
                    print(f"Error: you tried to restore a checkpoint ({self.checkpoint_dir}) that doesn't exist.")
                    print("Please clear the network and retrain, or load a different checkpoint by changing the steps parameter.")
                print('=> checkpoint file restored from ', ckpt_idx)
                utils.freeze_graph(sess, '../../data/checkpoint/yolov3_cpu_nms.pb', cpu_out_node_names)
                utils.freeze_graph(sess, '../../data/checkpoint/yolov3_gpu_nms.pb', gpu_out_node_names)


if __name__ == "__main__":
    weightconverter = WeightConverter(freeze=True,
                                      num_classes=27,
                                      dimensions_path="../../data/max_wh.txt",
                                      checkpoint_dir="../../data/checkpoint/",
                                      weights_dir="../../data/weights")
