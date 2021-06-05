import pickle

import cv2
import dlib
import imutils
import pandas as pd
from imutils import face_utils
from featureExtraction.feature import eye_aspect_ratio, mouth_aspect_ratio, circularity, mouth_over_eye, lip_distance

p = "models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def testing():
    image = cv2.imread('Image/Sub 1/Alert/2021_03_17_11_37_IMG_3534.JPG')
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 0)

# loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        # shape = face_utils.shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box

        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mstart:mend]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        mar = mouth_aspect_ratio(mouth)
        print(ear)
        print(mar)

        lCircularity = circularity(leftEye)
        rCircularity = circularity(rightEye)
        cir = (lCircularity + rCircularity) / 2
        print(cir)

        moe = mouth_over_eye(mar, ear)
        print(moe)

        lipDis = lip_distance(shape)
        print(lipDis)

        # Result_String, features = model(shape, image, loadModel())

        # print(features)
        # print(Result_String)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        cv2.putText(gray, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(gray, (x, y), 1, (0, 0, 255), -1)
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", gray)
    cv2.waitKey(0)

# data, value = test()
# print(value)
testing()