import cv2


def convertToLAB(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return cv2.split(lab)
