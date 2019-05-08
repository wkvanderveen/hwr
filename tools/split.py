#python3 tools/split.py -d ../data/letters -o imagesTest -i imagesTrain -p 20
import argparse
import os
from shutil import copyfile

class Splitter(object):
    """docstring for Splitter"""
    def __init__(self, source_dir, train_dir, test_dir, percentage):
        super(Splitter, self).__init__()
        self.source_dir = source_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.percentage = percentage

    def split(self):
        print("Splitting %s%% of the data found in %s\n" % (self.percentage, self.source_dir))

        split = round(100 / self.percentage)
        i = 0

        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        for root, dirs, files in os.walk(self.source_dir):
            for name in files:
                if i % split == 0:
                    image_path = os.path.join(root, name)
                    test_dir = os.path.join(self.test_dir, os.path.sep.join(image_path.split(os.path.sep)[4:-1]))
                    path_out = os.path.join(self.test_dir, image_path.split(os.path.sep)[-1])

                    copyfile(image_path, path_out)
                else:
                    image_path = os.path.join(root, name)
                    train_dir = os.path.join(self.train_dir, os.path.sep.join(image_path.split(os.path.sep)[4:-1]))
                    path_out = os.path.join(self.train_dir, image_path.split(os.path.sep)[-1])

                    copyfile(image_path, path_out)
                i += 1

if __name__ == "__main__":
    splitter = Splitter(source_dir="../../data/letters",
                        train_dir="../../data/letters-train",
                        test_dir="../../data/letters-test",
                        percentage=20)
    splitter.split()
