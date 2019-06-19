import numpy as np
from PIL import Image
from skimage import io
import os
import sys
import random
from skimage.transform import warp, AffineTransform
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
TRAIN_X_PATH = '../../data/letters-train'
TEST_X_PATH = '../../data/letters-test'
AUGMENTATION_FACTOR = 1
SHOW_AUGMENT = False
AUGMENT = True


# Reads train and test data and converts data to numpy array
class DataReader:
    def __init__(self):
        self.data = []
        self.max_width = self.max_height = 0
        self.path_train = TRAIN_X_PATH  # '../../data/letters/' #define path of example letter images
        self.path_test = TEST_X_PATH
        self.save_path = '../../data/'
        self.save_file_train = self.save_path + "train_letters"
        self.save_file_train_labels = self.save_path + "train_labels"
        self.save_file_train = self.save_path + "train_letters_aug"
        self.save_file_train_labels = self.save_path + "train_labels_aug"
        self.save_file_test = self.save_path + "test_letters"
        self.save_file_test_labes = self.save_path + "test_labels"
        self.save_file_dimensions = self.save_path + "max_dimensions"
        self.threshold = 200
        self.show_augment = SHOW_AUGMENT
        self.augment = AUGMENT

    def randRange(self, a, b):
        # a utility functio to generate random float values in desired range
        return random.random() * (b - a) + a

    def randomAffine(self, im):
        tform = AffineTransform(scale=(self.randRange(0.75, 1.3), self.randRange(0.75, 1.3)), 
            translation=(self.randRange(-im.shape[0]//10, im.shape[0]//10), self.randRange(-im.shape[1]//10, im.shape[1]//10)))
        return warp(im, tform.inverse, mode='reflect')

    def extra_augmentation(self, image):
        augmented_list = []
        for idx in range(0, AUGMENTATION_FACTOR):
            augmented_image = self.randomAffine(image)
            augmented_list.append(augmented_image)
            if self.show_augment == True:
                print("BEFORE")
                io.imshow(image)
                plt.show()
                print("AFTER")
                io.imshow(augmented_image)
                plt.show()
        return augmented_list

    def read_letters(self, mode):
        letters_list = []
        classes_list = []
        if mode == 'train':
            print("Reading train letters from dataset")
            num_classes = len(os.listdir(self.path_train))
            for class_num, filepath in enumerate(sorted(os.listdir(self.path_train))):
                sub_path = self.path_train+'/'+filepath
                print('current letter: ', filepath)
                for sub_file in os.listdir(sub_path):
                    img_path = sub_path + '/' + sub_file
                    with Image.open(img_path) as img:
                        width, height = img.size
                        if width > self.max_width:
                            self.max_width = width
                        if height > self.max_height:
                            self.max_height = height
                    onehot = np.zeros(num_classes)
                    onehot[class_num] = 1
                    image = io.imread(img_path, as_gray=True)
                    letters_list.append(image)
                    classes_list.append(onehot)
                    
                    
            letters = np.array(letters_list)
            classes = np.array(classes_list)
            print("Number of letters found in dataset:", letters.shape[0])
            np.save(self.save_file_train, letters)
            print("Letters saved as npy file in: ", self.save_file_train)
            np.save(self.save_file_train_labels, classes)
            print("Classes saved as npy file in: ", self.save_file_train_labels)
            if self.augment:
                letters = [] #Empty for memory space
                classes = [] #Empty for memory space
                for idx, letter in enumerate(letters_list):
                    augmented = self.extra_augmentation(image)
                    for image_aug in augmented:
                        letters.append(image_aug)
                        classes.append(classes_list[idx])
                letters = np.array(letters)
                classes = np.array(classes)
                print("Number of letters found in dataset:", letters.shape[0])
                np.save(self.save_file_train, letters)
                print("Letters saved as npy file in: ", self.save_file_train)
                np.save(self.save_file_train_labels, classes)
                print("Classes saved as npy file in: ", self.save_file_train_labels)

        if mode == 'test':
            print("Reading test letters from dataset")
            num_classes = len(os.listdir(self.path_train))
            for class_num, filepath in enumerate(sorted(os.listdir(self.path_test))):
                sub_path = self.path_test+'/'+filepath
                print('current letter: ', filepath)
                for sub_file in os.listdir(sub_path):
                    img_path = sub_path + '/' + sub_file
                    # print(img_path)
                    onehot = np.zeros(num_classes)
                    onehot[class_num] = 1
                    letters_list.append(io.imread(img_path, as_gray=True))
                    classes_list.append(onehot)
            letters = np.array(letters_list)
            classes = np.array(classes_list)
            print("Number of letters found in dataset:", letters.shape[0])
            np.save(self.save_file_test, letters)
            print("Letters saved as npy file in: ", self.save_file_test)
            np.save(self.save_file_test_labes, classes)
            print("Classes saved as npy file in: ", self.save_file_test_labes)


if __name__ == '__main__':
    reader = DataReader()
    # Select mode to read train or test data
    reader.read_letters('train')
    reader.read_letters('test')
