from math import ceil
import os
import numpy
import cv2
import sys
import datetime
from tqdm import tqdm
from imgaug import augmenters as iaa


class Augmenter(object):
    """docstring for Augmenter"""
    def __init__(self, source_dir, shear=0, coarse_dropout=(0, 0)):
        super(Augmenter, self).__init__()
        self.source_dir = source_dir
        self.shear = shear
        self.coarse_dropout = coarse_dropout


        self.DAMAGE = 0
        self.SHEAR = 1

        self.MIN_STATE = [0, -10]
        self.INTERVAL = [0.05, 5]
        self.MAX_STATE = [0.20, 10]

        self.STATE = list(self.MIN_STATE)
        self.ORIGINAL_IMAGE = [0] * len(self.MIN_STATE)


    def increment_state(self):
        i = 0
        while i < len(self.STATE):
            if round(self.STATE[i], 2) == self.MAX_STATE[i]:
                self.STATE[i] = self.MIN_STATE[i]
            else:
                self.STATE[i] = self.STATE[i] + self.INTERVAL[i]
                break
            i = i + 1

    def augment(self):
        image_paths = []
        for root, dirs, files in os.walk(self.source_dir):
            for name in files:
                image_paths.append(os.path.join(root, name))

        # loop over the input images
        j = 0

        for image_path in tqdm(image_paths):
            # load the image, pre-process it, and store it in the data list
            image = cv2.imread(image_path)

            while (True):
                if self.STATE == self.ORIGINAL_IMAGE:
                    cv2.imwrite(image_path, image)
                    self.increment_state()
                    continue

                # aug_seq = []
                # if self.shear:
                #     aug_seq.append(iaa.Affine(STATE[SHEAR], cval=(255)))

                # if not self.coarse_dropout == (0, 0):
                #     aug_seq.append(iaa.Invert(1))
                #     aug_seq.append(iaa.CoarseDropout(STATE[DAMAGE], size_percent=self.coarse_dropout))
                #     aug_seq.append(iaa.Invert(1))

                seq = iaa.Sequential([
                    iaa.Affine(shear=self.STATE[self.SHEAR], cval=(255)),
                    iaa.Invert(1),
                    iaa.CoarseDropout(self.STATE[self.DAMAGE], size_percent=self.coarse_dropout),
                    iaa.Invert(1)
                ])

                images_aug = seq.augment_image(image)

                cv2.imwrite(os.path.split(image_path)[0] + "/Augmented" + str(j) + "-S" + str(self.STATE[self.SHEAR]) +
                            "-D" + str(self.STATE[self.DAMAGE]) + ".jpeg", images_aug)

                state = numpy.around(self.STATE, decimals=2)
                self.STATE = [state[0], state[1]]
                if (self.STATE == self.MAX_STATE):
                    self.STATE = list(self.MIN_STATE)
                    break
                else:
                    self.increment_state()

            j = j + 1


if __name__ == "__main__":
    augmenter = Augmenter(source_dir="../data/letters-train", shear=True, coarse_dropout=(0.02, 0.5))
    augmenter.augment()
