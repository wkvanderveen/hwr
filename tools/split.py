import argparse
import os
from shutil import copyfile

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-tr", "--train", required=True,
                help="path for training dataset")
ap.add_argument("-te", "--test", required=True,
                help="path for test dataset")
ap.add_argument("-p", "--percentage", required=True,
                help="percentage of data to split")
args = vars(ap.parse_args())

print("Splitting %s%% of the data found in %s\n" % (args["percentage"], args["dataset"]))

split = round(100 / (int(args["percentage"])))
i = 0
moved = 0

for root, dirs, files in os.walk(args["dataset"]):
    for name in files:
        if i % split == 0:
            imagePath = os.path.join(root, name)
            testdir = os.path.join(args["test"], os.path.sep.join(imagePath.split(os.path.sep)[4:-1]))
            outputImagePath = os.path.join(testdir, imagePath.split(os.path.sep)[-1])
            print("copying %s -> %s" % (imagePath, outputImagePath))

            if not os.path.exists(testdir):
                os.makedirs(testdir)
            copyfile(imagePath, outputImagePath)
            moved += 1
        else:
            imagePath = os.path.join(root, name)
            traindir = os.path.join(args["train"], os.path.sep.join(imagePath.split(os.path.sep)[4:-1]))
            outputImagePath = os.path.join(traindir, imagePath.split(os.path.sep)[-1])

            if not os.path.exists(traindir):
                os.makedirs(traindir)
            copyfile(imagePath, outputImagePath)
        i += 1


print("\ncopied %d out of %d files to testset" % (moved, i))
print("intended to move %s%% -> actually moved %d%%" % (args["percentage"], ((moved * 100.0) / i)))
