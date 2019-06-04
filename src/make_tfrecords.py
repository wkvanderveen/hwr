import cv2
import numpy as np
import tensorflow as tf
import os, sys
np.set_printoptions(threshold=sys.maxsize)

class TfRecordMaker(object):
    """docstring for TfRecordMaker"""
    def __init__(self, label_path, imgs_dir, colab=False):
        super(TfRecordMaker, self).__init__()
        self.label_path = label_path
        self.imgs_dir = imgs_dir
        self.colab = colab

    def make_records(self):

        dataset = {}
        with open(self.label_path,'r') as f:
            if self.colab == True:
                increase=1
            else:
                increase=0

            for line in f.readlines():
                example = line.split(' ')
                if increase == 0:
                    image_path = example[0]
                else:
                    image_path = str(example[0]+" "+example[increase])
                boxes_num = len(example[(increase+1):]) // 5
                boxes = np.zeros([boxes_num, 5], dtype=np.float32)
                for i in range(boxes_num):
                    boxes[i] = example[increase+(1+i*5):increase+(6+i*5)]
                dataset[image_path] = boxes

        image_paths = list(dataset.keys())
        images_num = len(image_paths)
        print(f">> Processing {images_num} images")

        tfrecord_file = os.path.normpath(self.imgs_dir)+".tfrecords"

        with tf.python_io.TFRecordWriter(tfrecord_file) as record_writer:
            for i in range(images_num):
                # image = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE) << 1 CHANNEL
                image_orig = cv2.imread(image_paths[i])
                #image_orig = image_orig / 255.
                normalizedImg = cv2.normalize(image_orig, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                image = cv2.imencode('.jpg', normalizedImg)[1].tostring()

                # image = tf.gfile.FastGFile(image_paths[i], 'rb').read()
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
