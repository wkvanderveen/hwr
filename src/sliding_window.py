import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

np.set_printoptions(threshold=np.inf)


class SlidingWindow:
    def __init__(self):
        self.characters = ["Alef", "Ayin", "Bet", "Dalet", "Gimel", "He", "Het", "Kaf", "Kaf-final", "Lamed", "Mem",
                           "Mem-medial", "Nun-final",
                           "Nun-medial", "Pe", "Pe-final", "Qof", "Resh", "Samekh", "Shin", "Taw", "Tet", "Tsadi-final",
                           "Tsadi-medial", "Waw", "Yod", "Zayin"]
        self.model = load_model("../data/models/backup_model.model")
        self.save_kernel_path = "../data/"
        self.txtfile = open("../data/softmax.txt", "w")
        self.final_yaxis = False
        self.final_xaxis = False
        self.stop = False
        self.i = 0
        self.CONFIDENCE_THRESHOLD = 0.6
        self.SHOW_PLOT = False
        self.WRITE_WINDOWS = False

    def load_image(self, image):
        self.image = image
        self.aspect = self.image.shape[1] / self.image.shape[0]
        self.reshape_height = 60
        self.reshape_width = int(60 * self.aspect)
        if self.reshape_width < 39:
            print('reshaping')
            self.reshape_width = 39
            self.reshape_height = int(39 / self.aspect)
        self.image = cv2.resize(self.image, (self.reshape_width, self.reshape_height))
        self.stepSize = 1
        (self.w_width, self.w_height) = (39, 39)  # window size
        self.classificationMatrix = np.zeros(shape=(len(self.characters), self.reshape_width))
        self.PEAK_CONCAT_DIST = self.image.shape[0] * 0.2

    def probs_to_one_hot(self, arr):
        arr_len = arr.shape[1]
        new = np.zeros(arr_len, dtype=int)
        maxim = np.max(arr)
        if maxim > self.CONFIDENCE_THRESHOLD:
            new[np.argmax(arr)] = 1
        return new

    def get_letters(self):
        prediction_list = []
        for x in range(0, self.image.shape[1], self.stepSize):
            temp_prediction_list = []
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
                    temp = window.reshape((1, self.w_height, self.w_width, 1))
                    temp = np.interp(temp, (temp.min(), temp.max()), (0, 1))  # Normalize image between 0 and 1

                    whites_top = 0
                    for i in range(self.w_height):
                        if np.count_nonzero(temp[0][i] == 1) / 39. == 1:
                            whites_top += 1
                        else:
                            break
                    whites_bottom = 0
                    for i in range(self.w_height):
                        if np.count_nonzero(temp[0][i] == 1) / 39. == 1:
                            whites_bottom += 1
                        else:
                            whites_bottom = 0
                            continue

                    # IF WHITE LINES TOP AND BOTTOM IS EQUAL:
                    if abs(whites_bottom - whites_top) <= 1:
                        predict = self.model.predict(temp)
                        onehot = self.probs_to_one_hot(predict)
                        idxes = [np.where(onehot != 0.0)[0]][0]
                        # softmaxes = [np.where(predict[0] != 0.0)[0]][0]
                        self.txtfile.write(str(self.i) + " - ")
                        for idx in idxes:  # loop, in case 2 or more characters have same probability
                            self.classificationMatrix[idx][x] += 1
                            filename += self.characters[idx]
                            self.txtfile.write(self.characters[idx] + " " + str(predict) + " ")
                        self.txtfile.write('\n')
                        filename = self.save_kernel_path + str(self.i) + "-" + filename + ".png"
                        if self.WRITE_WINDOWS:
                            cv2.imwrite(filename, window)
                        predict = predict[0]  # collapse dimensions of double list 'predict'
                        temp_prediction_list.append(predict.tolist())

                if self.final_yaxis and self.final_xaxis:
                    self.stop = True
                if self.final_yaxis:
                    break

            mean_of_column = [float(sum(col)) / len(col) for col in zip(*temp_prediction_list)]
            if not mean_of_column == []:
                prediction_list.append(mean_of_column)

        self.txtfile.close()
        return prediction_list


if __name__ == '__main__':
    sw = SlidingWindow()
    image_file = "../data/backup_val_lines/line1.png"
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)  # your image path
    sw.load_image(image)
    prediction_list = sw.get_letters()
