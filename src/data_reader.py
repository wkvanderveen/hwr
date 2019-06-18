import numpy as np
from skimage import io
import os
import sys
np.set_printoptions(threshold=sys.maxsize)
TRAIN_X_PATH = '../../data/letters-train'
TEST_X_PATH = '../../data/letters-test'

# Reads train and test data and converts data to numpy array
class DataReader:
    def __init__(self):
        self.data = []
        self.path_train = TRAIN_X_PATH #'../../data/letters/' #define path of example letter images
        self.path_test = TEST_X_PATH
        self.save_path = '../../data/'
        self.save_file_train = self.save_path + "train_letters"
        self.save_file_train_labes = self.save_path + "train_labels"
        self.save_file_test = self.save_path + "test_letters"
        self.save_file_test_labes = self.save_path + "test_labels"
        self.threshold = 200

    def read_letters(self, mode):
        letters = []
        classes = []
        if mode == 'train':
            print("Reading train letters from dataset")
            num_classes = len(os.listdir(self.path_train))
            for class_num, filepath in enumerate(os.listdir(self.path_train)):
                sub_path = self.path_train+'/'+filepath
                print('current letter: ', filepath)
                for sub_file in os.listdir(sub_path):
                    img_path = sub_path+'/'+sub_file
                    #print(img_path)
                    onehot = np.zeros(num_classes)
                    onehot[class_num] = 1
                    letters.append(io.imread(img_path, as_gray=True))
                    classes.append(onehot)
            letters = np.array(letters)
            classes = np.array(classes)
            print("Number of letters found in dataset:", letters.shape[0])
            np.save(self.save_file_train, letters)
            print("Letters saved as npy file in: ", self.save_file_train)
            np.save(self.save_file_train_labes, classes)
            print("Classes saved as npy file in: ", self.save_file_train_labes)
        if mode == 'test':
            print("Reading test letters from dataset")
            num_classes = len(os.listdir(self.path_train))
            for class_num, filepath in enumerate(os.listdir(self.path_test)):
                sub_path = self.path_test+'/'+filepath
                print('current letter: ', filepath)
                for sub_file in os.listdir(sub_path):
                    img_path = sub_path+'/'+sub_file
                    #print(img_path)
                    onehot = np.zeros(num_classes)
                    onehot[class_num] = 1
                    letters.append(io.imread(img_path, as_gray=True))
                    classes.append(onehot)
            letters = np.array(letters)
            classes = np.array(classes)
            print("Number of letters found in dataset:", letters.shape[0])
            np.save(self.save_file_test, letters)
            print("Letters saved as npy file in: ", self.save_file_test)
            np.save(self.save_file_test_labes, classes)
            print("Classes saved as npy file in: ", self.save_file_test_labes)

if __name__=='__main__':
    reader = DataReader()
    #Select mode to read train or test data
    reader.read_letters('train')
    reader.read_letters('test')
