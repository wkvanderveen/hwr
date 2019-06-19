import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
np.set_printoptions(threshold=np.inf)


class SlidingWindow:
    def __init__(self):
        self.characters = ["Kaf-final", "Gimel", "Samekh", "Tet", "Lamed", "Dalet", "Alef", "Yod", "Resh", "Shin", "Taw", "Bet",
                      "Pe-final", "Mem-medial", "Het", "He", "Waw", "Mem", "Qof", "Nun-final", "Tsadi-final", "Kaf",
                      "Nun-medial", "Pe", "Tsadi-medial", "Ayin", "Zayin"] # The order is decided by data_reader.py
        self.model = load_model("../data/models/backup_model.model")
        self.image_file = "../data/backup_val_lines/line2.png"
        self.save_kernel_path = "../data/"
        self.txtfile = open("../../data/softmax.txt", "w")
        self.final_yaxis = False
        self.final_xaxis = False
        self.stop = False
        self.i = 0

        self.image = cv2.imread(self.image_file, cv2.IMREAD_GRAYSCALE)  # your image path
        self.aspect = self.image.shape[1] / self.image.shape[0]
        self.reshape_height = 60
        self.reshape_width = int(60 * self.aspect)
        self.image = cv2.resize(self.image, (self.reshape_width, self.reshape_height))
        self.stepSize = 4
        (self.w_width, self.w_height) = (19, 49)  # window size
        self.classificationMatrix = np.zeros(shape=(len(self.characters), self.reshape_width))

        self.PEAK_CONCAT_DIST = self.image.shape[0]*0.2
        self.CONFIDENCE_THRESHOLD = 0.8


    def find_mean(self, x):
        length = len(x)
        total = 0
        for i in range(length):
            total += x[i]
        return total/length

    def merge_peaks(self, x):
        width, height = x.shape # shape: [27, n]
        last_saved_idx = 0
        for idx in range(width):
            for idx2 in range(height):
                if x[idx, idx2] > 0:
                    #print(x[idx, idx2])
                    if (idx2 - last_saved_idx) < self.PEAK_CONCAT_DIST:
                        x[idx, last_saved_idx:idx2] = x[idx, last_saved_idx]
                    last_saved_idx = idx2
        return x

    def find_peaks(self, x):
        peaks = []
        upper_arr = []
        max = np.max(x)
        length = len(x)
        width, height = x.shape # shape: [27, n]
        
        for idx2 in range(height): #Loop over the histogram heights
            mean1 = self.find_mean(np.unique(x)) #unique since there are a lot of 0s; count the mean hist count of all letters
            last_saved_idx = 0 #used to concat same character peaks with small distance
            if mean1 != 0:
                for idx in range(width):
                    item = x[idx, idx2]
                    if item <= mean1:
                        x[idx, idx2] = 0.0
                        
                    if item > mean1:
                        upper_arr.append(item)
                # mean2 = self.find_mean(np.unique(np.array(upper_arr))) #unique since there are a lot of 0s
                # for idx in range(width):
                #     item = x[idx, idx2]
                #     if item <= mean2:
                #         x[idx, idx2] = 0.0
                # print(mean1, mean2)
        #print(x)
        return x

    def probs_to_one_hot(self, arr):
        arr_len = arr.shape[1]
        new = np.zeros(arr_len, dtype = int)
        maxim = np.max(arr)
        if maxim > self.CONFIDENCE_THRESHOLD: 
            new[np.argmax(arr)] = 1
        return new

    def get_letters(self):
        for x in range(0, self.image.shape[1], self.stepSize):
            self.final_yaxis = False

            if (x + self.w_width) >= self.image.shape[1]:
                x = self.image.shape[1] - self.w_width
                self.final_xaxis = True

            for y in range(0, self.image.shape[0], self.stepSize):
                self.i = self.i + 1
                filename = ""

                if (y + self.w_height) >= self.image.shape[0]:
                    y = self.image.shape[0] - self.w_height
                    self.final_yaxis = True

                if self.stop is False:
                    window = self.image[y:y + self.w_height, x:x + self.w_width]
                    temp = window.reshape((1, self.w_width, self.w_height, 1))
                    temp = np.interp(temp, (temp.min(), temp.max()), (0, 1)) #Normalize image between 0 and 1
                    predict = self.model.predict(temp)
                    onehot = self.probs_to_one_hot(predict)
                    idxes = [np.where(onehot != 0.0)[0]][0]
                    #softmaxes = [np.where(predict[0] != 0.0)[0]][0]
                    self.txtfile.write(str(self.i) + " - ")
                    for idx in idxes: #loop, in case 2 or more characters have same probability
                        self.classificationMatrix[idx][x] += 1
                        filename += self.characters[idx]
                        self.txtfile.write(self.characters[idx] + " " + str(predict) + " ")
                    self.txtfile.write('\n')
                    filename = self.save_kernel_path + str(self.i) + "-" + filename + ".png"
                    cv2.imwrite(filename, window)

                if self.final_yaxis and self.final_xaxis:
                    self.stop = True

                if self.final_yaxis:
                    break
        self.txtfile.close()
        ret = self.find_peaks(self.classificationMatrix).tolist()
        ret = self.merge_peaks(np.array(ret)).tolist()
        plt.subplot(2, 1, 1)
        for idx in range(0, 27):
            if any(item > 0 for item in ret[idx]) is True:
                plt.plot(ret[idx], label=self.characters[idx])
                plt.legend()
        plt.subplot(2, 1, 2)
        img=mpimg.imread(self.image_file)
        plt.imshow(img)
        plt.show()

        #TODO: implement program to go from histogram to Hebrew character output

if __name__ == '__main__':
    sw = SlidingWindow()
    sw.get_letters()
