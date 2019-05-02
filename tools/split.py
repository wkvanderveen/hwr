#python3 tools/split.py -d ../data/letters -o imagesTest -i imagesTrain -p 20
import argparse
import os
from shutil import copyfile

dataset = "../../data/letters"
train = "../../data/train"
test = "../../data/test"
percentage = 20

print("Splitting %s%% of the data found in %s\n" % (percentage, dataset))

split = round(100 / percentage)
i = 0
moved = 0

for root, dirs, files in os.walk(dataset):
    for name in files:
        if i % split == 0:
            imagePath = os.path.join(root, name)
            testdir = os.path.join(test, os.path.sep.join(imagePath.split(os.path.sep)[4:-1]))
            outputImagePath = os.path.join(testdir, imagePath.split(os.path.sep)[-1])

            if not os.path.exists(testdir):
                os.makedirs(testdir)
            copyfile(imagePath, outputImagePath)
            moved += 1
        else:
            imagePath = os.path.join(root, name)
            traindir = os.path.join(train, os.path.sep.join(imagePath.split(os.path.sep)[4:-1]))
            outputImagePath = os.path.join(traindir, imagePath.split(os.path.sep)[-1])

            if not os.path.exists(traindir):
                os.makedirs(traindir)
            copyfile(imagePath, outputImagePath)
        i += 1


print("\ncopied %d out of %d files to testset" % (moved, i))
print("intended to move %s%% -> actually moved %d%%" % (percentage, ((moved * 100.0) / i)))
