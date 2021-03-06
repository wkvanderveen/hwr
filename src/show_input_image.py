import cv2
import os
import numpy as np
import tensorflow as tf
from core import utils
from PIL import Image
from core.dataset import Parser, dataset

class ExampleDisplayer(object):
    """docstring for ExampleDisplayer"""
    def __init__(self, source_dir, img_dims, anchor_dir, num_classes, cell_size):
        super(ExampleDisplayer, self).__init__()
        self.source_dir = source_dir
        self.anchor_dir = anchor_dir
        self.num_classes = num_classes
        self.img_h = img_dims[0]
        self.img_w = img_dims[1]
        self.cell_size = cell_size

    def show_example(self):
        sess = tf.Session()
        classes = os.listdir(self.source_dir[:-len(".tfrecords") or None])

        train_tfrecord = self.source_dir
        anchors        = utils.get_anchors(self.anchor_dir, self.img_h, self.img_w)

        parser   = Parser(image_h=self.img_h,
                          image_w=self.img_w,
                          anchors=anchors,
                          num_classes=self.num_classes,
                          cell_size=self.cell_size,
                          debug=True)
        trainset = dataset(parser, train_tfrecord, 1, shuffle=1)

        is_training = tf.placeholder(tf.bool)
        example = trainset.get_next()

        image, boxes = sess.run(example)
        image, boxes = image[0], boxes[0]

        n_box = len(boxes)
        print(boxes)
        image = np.repeat(image, 3, axis=2)
        for i in range(n_box):
            image = cv2.rectangle(image,(int(float(boxes[i][0])),
                                         int(float(boxes[i][1]))),
                                        (int(float(boxes[i][2])),
                                         int(float(boxes[i][3]))), (255,0,0), 1)
            label = str(int(float(boxes[i][4])))
            image = cv2.putText(image, label, (int(float(boxes[i][0])),int(float(boxes[i][1]))),
                                cv2.FONT_HERSHEY_SIMPLEX,  .6, (0, 255, 0), 1, 2)


        image = Image.fromarray(np.uint8(image*255))
        image.show()


if __name__ == "__main__":

    with open("../../data/dimensions.txt", "r") as max_dimensions:
        img_h, img_w = [int(x) for x in max_dimensions.read().split()]
    img_dims = (img_h, img_w)

    num_classes = len(os.listdir("../../data/letters-train/"))

    displayer = ExampleDisplayer(source_dir="../../data/lines-train.tfrecords",
                                 anchor_dir="../../data/anchors.txt",
                                 num_classes=num_classes,
                                 img_dims=img_dims)
    displayer.show_example()
