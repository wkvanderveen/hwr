import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import load_model


class SlidingWindow:
    predictions = []
    image = cv2.imread("../../data/TESTLINE.png", cv2.IMREAD_GRAYSCALE)  # your image path
    aspect = image.shape[1]/image.shape[0]
    image = cv2.resize(image,(int(45*aspect),45))
    tmp = image  # for drawing a rectangle
    stepSize = 10
    (w_width, w_height) = (38, 38)  # window size
    for x in range(0, image.shape[1], stepSize):
        if (x + w_width) > image.shape[1]:
            x = image.shape[1] - w_width
        for y in range(0, image.shape[0], stepSize):
            if (y + w_height) > image.shape[0]:
                y = image.shape[0] - w_height

            window = np.ndarray(shape=(1,38,38))
            window[0] = image[y:y + w_height,x:x + w_width]
            temp = window[0].reshape((1,38,38,1))
            model = load_model("../../data/backup_model.model")
            predict = model.predict(temp)
            print(temp[0][1:])
            plt.imshow(np.array(temp[0][1:]).astype('uint8'))
            plt.show()

            cv2.rectangle(tmp, (x, y), (x + w_width,y + w_height), (0), 2)  # draw rectangle on image
            # plt.imshow(np.array(tmp).astype('uint8'))
            predictions.append(predict)
    # show all windows
    # plt.show()

    print(predictions)