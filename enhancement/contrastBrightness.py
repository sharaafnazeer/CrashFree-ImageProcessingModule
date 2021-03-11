import cv2 as cv
import numpy as np


def gammaCorrection(image):
    gamma = 1.0
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    return cv.LUT(image, lookUpTable)


def adjustImage(image):
    alpha = 1.5  # Simple contrast control
    beta = 10  # Simple brightness control
    return gammaCorrection(cv.convertScaleAbs(image, alpha=alpha, beta=beta))
