import os
import numpy
import cv2
import datetime
from imgaug import augmenters as iaa


print(datetime.datetime.now().time())


def data_augmenter():
    dataset = "../../data/train/"

    imagePaths = []
    for root, dirs, files in os.walk(dataset):
        for name in files:
            imagePaths.append(os.path.join(root, name))

    # ----------------------------------
    # State indices
    DAMAGE = 0
    SHEAR = 1

    MIN_STATE = [0, -10]
    INTERVAL = [0.05, 5]
    MAX_STATE = [0.20, 10]

    STATE = list(MIN_STATE)
    ORIGINAL_IMAGE = [0] * len(MIN_STATE)

    def increment_state():
        i = 0
        while i < len(STATE):
            if round(STATE[i], 2) == MAX_STATE[i]:
                STATE[i] = MIN_STATE[i]
            else:
                STATE[i] = STATE[i] + INTERVAL[i]
                break
            i = i + 1

    # loop over the input images
    j = 0

    max_h, max_w = 0, 0

    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        if max_h < image.shape[0]: max_h = image.shape[0]
        if max_w < image.shape[1]: max_w = image.shape[1]


        while (True):
            if STATE == ORIGINAL_IMAGE:
                cv2.imwrite(imagePath, image)
                increment_state()
                continue

            seq = iaa.Sequential([
                iaa.Affine(shear=(STATE[SHEAR]), cval=(255)),
                iaa.Invert(1),
                iaa.CoarseDropout(STATE[DAMAGE], size_percent=(0.02, 0.50)),
                iaa.Invert(1),
            ])

            images_aug = seq.augment_image(image)

            cv2.imwrite(os.path.split(imagePath)[0] + "/Augmented" + str(j) + "-S" + str(STATE[SHEAR]) +
                        "-D" + str(STATE[DAMAGE]) + ".jpeg", images_aug)

            state = numpy.around(STATE, decimals=2)
            STATE = [state[0], state[1]]
            if (STATE == MAX_STATE):
                STATE = list(MIN_STATE)
                break
            else:
                increment_state()

        j = j + 1

    return max_w, max_h


def generateTextLabels():
    datasets = ["../../data/test", "../../data/train"]

    def write_to_dataset(dataset_type, path, x1, y1, x2, y2, c):
        with open("../../data/dataset_{}.txt".format(dataset_type), "a") as filename:
            print(f"{path} {x1} {y1} {x2} {y2} {c - 1}", file=filename)
    idx = 0
    for dataset in datasets:
        print("Processing {}".format(datasets))
        for root, dirs, files in os.walk(dataset):
            dataset_type = os.path.split(dataset)[1]

            if dirs:
                num_letters = len(dirs)
                continue
            idx += 1

            _, letter = os.path.split(root)
            print("Processing letter {}-{} ({}/{})...".format(os.path.split(dataset)[1], letter, idx, num_letters*2))

            for name in files:
                image = cv2.imread(os.path.join(root, name))
                height, width, channels = image.shape
                write_to_dataset(dataset_type, os.path.join(root, name), 0, 0, width, height, idx)


max_w, max_h = data_augmenter()
generateTextLabels()
print(datetime.datetime.now().time())

with open("../../data/max_dimensions.txt", "w+") as filename:
    print(f"{max_w} {max_h}", file=filename)
