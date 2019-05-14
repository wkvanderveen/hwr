import numpy as np
import tensorflow as tf
import os

class TfRecordMaker(object):
    """docstring for TfRecordMaker"""
    def __init__(self, label_path, imgs_dir):
        super(TfRecordMaker, self).__init__()
        self.label_path = label_path
        self.imgs_dir = imgs_dir

    def make_records(self):

        dataset = {}
        with open(self.label_path,'r') as f:
            for line in f.readlines():
                example = line.split(' ')
                image_path = example[0]
                boxes_num = len(example[1:]) // 5
                boxes = np.zeros([boxes_num, 5], dtype=np.float32)
                for i in range(boxes_num):
                    boxes[i] = example[1+i*5:6+i*5]
                dataset[image_path] = boxes

        image_paths = list(dataset.keys())
        images_num = len(image_paths)
        print(f">> Processing {images_num} images")

        tfrecord_file = os.path.normpath(self.imgs_dir)+".tfrecords"
        print(tfrecord_file)
        with tf.python_io.TFRecordWriter(tfrecord_file) as record_writer:
            for i in range(images_num):
                image = tf.gfile.FastGFile(image_paths[i], 'rb').read()
                boxes = dataset[image_paths[i]]
                boxes = boxes.tostring()

                example = tf.train.Example(features = tf.train.Features(
                    feature={
                        'image' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                        'boxes' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [boxes])),
                    }
                ))

                record_writer.write(example.SerializeToString())
            print(f">> Saving {images_num} images in {tfrecord_file}")


if __name__ == "__main__":
    recordmaker = TfRecordMaker(imgs_dir="../../data/lines-train/", label_path="../../data/labels-train.txt")
    recordmaker.make_records()
    recordmaker = TfRecordMaker(imgs_dir="../../data/lines-test/", label_path="../../data/labels-test.txt")
    recordmaker.make_records()
