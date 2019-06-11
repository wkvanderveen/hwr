import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)


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
    reshape_height = 60
    reshape_width = int(60 * aspect)
    image = cv2.resize(image, (reshape_width, reshape_height))
    stepSize = 4
    (w_width, w_height) = (38, 38)  # window size
    classificationMatrix = np.zeros(shape=(len(characters), reshape_width))

    def find_peaks(x):
        max = np.max(x)
        length = len(x)
        ret = []
        for i in range(length):
            ispeak = True
            if i - 1 > 0:
                ispeak &= (x[i] > 1.8 * x[i - 1])
            if i + 1 < length:
                ispeak &= (x[i] > 1.8 * x[i + 1])

            ispeak &= (x[i] > 0.05 * max)
            if ispeak:
                ret.append(i)
        return ret

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
                # txtfile.write(str(i) + " - ")
                for softmax in softmaxes:
                    classificationMatrix[softmax][x] += 1
                    filename += characters[softmax]
                    # txtfile.write(characters[softmax] + " " + str(predict[0][softmax]) + " ")
                # txtfile.write('\n')
                # filename = "../../data/" + str(i) + "-" + filename + ".png"
                # cv2.imwrite(filename, window)

            if final_yaxis and final_xaxis:
                stop = True

            if final_yaxis:
                break
    txtfile.close()
    ret = []
    for i in range(0, 27):
        ret.append(find_peaks(classificationMatrix[i]))
        # classificationMatrix[i] = classificationMatrix[i] - 15
        # if(classificationMatrix[i])
        plt.plot(classificationMatrix[i], label=characters[i])

    print(ret)
    plt.legend()
    plt.show()
