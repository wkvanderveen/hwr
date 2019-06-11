import cv2
import numpy as np
from keras.models import load_model


class SlidingWindow:
    characters = ["Kaf-final", "Gimel", "Samekh", "Tet", "Lamed", "Dalet", "Alef", "Yod", "Resh", "Shin", "Taw", "Bet",
                  "Pe-final", "Mem-medial", "Het", "He", "Waw", "Mem", "Qof", "Nun-final", "Tsadi-final", "Kaf",
                  "Nun-medial", "Pe", "Tsadi-medial", "Ayin", "Zayin"] # The order is decided by data_reader.py
    model = load_model("../../data/backup_model.model")
    final_yaxis = False
    final_xaxis = False
    stop = False
    i = 0

    image = cv2.imread("../../data/TESTLINE.jpg", cv2.IMREAD_GRAYSCALE)  # your image path
    txtfile = open("../../data/softmax.txt", "w")
    aspect = image.shape[1] / image.shape[0]
    image = cv2.resize(image, (int(60 * aspect), 60))
    stepSize = 2
    (w_width, w_height) = (38, 38)  # window size

    for x in range(0, image.shape[1], stepSize):
        final_yaxis = False

        if (x + w_width) >= image.shape[1]:
            x = image.shape[1] - w_width
            final_xaxis = True

        for y in range(0, image.shape[0], stepSize):
            i = i + 1
            filename = ""

            if (y + w_height) >= image.shape[0]:
                y = image.shape[0] - w_height
                final_yaxis = True

            if stop is False:
                window = image[y:y + w_height, x:x + w_width]
                temp = window.reshape((1, 38, 38, 1))
                predict = model.predict(temp)
                softmaxes = [np.where(predict[0] != 0.0)[0]][0]
                txtfile.write(str(i) + " - ")
                for softmax in softmaxes:
                    filename += characters[softmax]
                    txtfile.write(characters[softmax] + " " + str(predict[0][softmax]) + " ")
                txtfile.write('\n')
                filename = "../../data/" + str(i) + "-" + filename + ".png"
                cv2.imwrite(filename, window)

            if final_yaxis and final_xaxis:
                stop = True

            if final_yaxis:
                break
    txtfile.close()
