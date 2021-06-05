# Importing all necessary libraries
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np

from os import makedirs
from os.path import exists

# project
from enhancement.contrastBrightness import adjustImage

faceCascadeDetector = cv2.CascadeClassifier('models/cascades/haarcascade_frontalface_default.xml')

# video source from webcam
cam = cv2.VideoCapture(0)
directory = "data"
try:
    # creating a folder named data
    if not exists(directory):
        makedirs(directory)

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0
frameOffset = 24
frameLimit = 24

gamma = 0.6
lambda_ = 0.15
lime = True
sigma = 3
bc = 1
bs = 1
be = 1
eps = 1e-3

while cam.isOpened():

    # read the current frame
    (grabbed, frame) = cam.read()

    # check frame is fetched or not
    if not grabbed:
        break

    # resixe the frame
    frame = cv2.resize(frame, (640, 480), fx=0, fy=0,
                       interpolation=cv2.INTER_CUBIC)

    adjustedImage = adjustImage(frame)
    cv2.imwrite('data/current.jpg', adjustedImage)

    savedImage = cv2.imread('data/current.jpg', 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    savedImage = clahe.apply(savedImage)

    # gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    # detect faces using haar cascade approach
    faces = faceCascadeDetector. \
        detectMultiScale(savedImage,
                         scaleFactor=1.1,
                         minNeighbors=5, minSize=(80, 80),
                         flags=cv2.CASCADE_SCALE_IMAGE)

    # draw faces
    for (x, y, w, h) in faces:
        cv2.rectangle(adjustedImage, (x, y),
                      (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", adjustedImage)
    key = cv2.waitKey(1)
cam.release()
cv2.destroyAllWindows()
