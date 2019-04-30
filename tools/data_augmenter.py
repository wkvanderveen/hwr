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
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)

        while (True):
            if STATE == ORIGINAL_IMAGE:
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


def generateTextLabels():
    sets = ["../../data/test", "../../data/train"]

    f = open("../../data/dataset.txt", "w+")
    f = open("../../data/anchors.txt", "w+")

    img_h, img_w = 32, 32

    def write_to_dataset(path, x1, x2, y1, y2, c):
        with open("../../data/dataset.txt", "a") as filename:
            print(f"{path} {x1 - 1} {x2 - 1} {y1} {y2} {c - 1}", file=filename)

    for set in sets:
        for root, dirs, files in os.walk(set):
            if dirs:
                idx = 0
                num_letters = len(dirs)
                continue
            idx += 1

            _, letter = os.path.split(root)
            print("Processing letter {} ({}/{})...".format(letter, idx, num_letters))

            for name in files:
                write_to_dataset(os.path.join(root, name), 0, 0, img_w, img_h, idx)


data_augmenter()
generateTextLabels()
print(datetime.datetime.now().time())