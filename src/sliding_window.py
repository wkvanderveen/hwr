import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import load_model

characters = ["Alef","Ayin","Bet","Dalet","Gimel","He","Het","Kaf","Kaf-final","Lamed","Mem","Mem-medial","Nun-final",
              "Nun-medial","Pe","Pe-final","Qof","Resh","Samekh","Shin","Taw","Tet","Tsadi-final","Tsadi-medial","Waw",
              "Yod","Zayin"]


class SlidingWindow:
    predictions = []
    final_yaxis = False
    final_xaxis = False

    image = cv2.imread("../../data/TESTLINE.png", cv2.IMREAD_GRAYSCALE)  # your image path
    aspect = image.shape[1]/image.shape[0]
    image = cv2.resize(image,(int(45*aspect),45))
    tmp = image  # for drawing a rectangle
    stepSize = 20
    (w_width, w_height) = (38, 38)  # window size

    for x in range(0, image.shape[1], stepSize):
        final_yaxis = False
        if (x + w_width) >= image.shape[1]:
            x = image.shape[1] - w_width
            final_xaxis = True
        for y in range(0, image.shape[0], stepSize):
            if (y + w_height) >= image.shape[0]:
                y = image.shape[0] - w_height
                final_yaxis = True

            window = np.ndarray(shape=(38,38))
            window = image[y:y + w_height,x:x + w_width]
            # print(f"X({x}:{x+w_width}), Y({y}:{y+w_height})")

            temp = window.reshape((1,38,38,1))
            model = load_model("../../data/backup_model.model")
            predict = model.predict(temp)
            print(predict[0])
            softmaxes = [np.where(s != 0) for s in predict[0]]
            print(softmaxes)
            # for softmax in softmaxes:
                # print(characters[softmax],predict[0][softmax])
            # test = np.array(temp[0][:,:,0]).astype('uint8')
            # print(test.shape)
            # plt.imshow(test)

            cv2.rectangle(tmp, (x, y), (x + w_width,y + w_height), (0), 2)  # draw rectangle on image
            # plt.imshow(np.array(tmp).astype('uint8'))
            predictions.append(predict)

            if final_xaxis:
                break
            if final_yaxis:
                break
    # show all windows
    # plt.show()

    print(predictions)