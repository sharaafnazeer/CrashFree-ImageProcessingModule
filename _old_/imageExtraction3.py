import os
import cv2
import dlib
import numpy as np
from imutils import face_utils

from featureExtraction.feature import eye_aspect_ratio, mouth_aspect_ratio, circularity, mouth_over_eye, lip_distance

p = "models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


def resolveFiles(subjectFolders):
    drowsyFiles = []
    alertFiles = []

    for subjectFolder in subjectFolders:
        contentFolders = next(os.walk('Image/' + subjectFolder))[1]
        for contentFolder in contentFolders:

            if contentFolder == 'Drowsy':
                drowsyFilesTemp = next(os.walk('Image/' + subjectFolder + '/' + contentFolder))[2]
                for drowsy in drowsyFilesTemp:
                    fileName = 'Image/' + subjectFolder + '/' + contentFolder + '/' + drowsy
                    drowsyFiles.append(fileName)
            else:
                alertFilesTemp = next(os.walk('Image/' + subjectFolder + '/' + contentFolder))[2]
                for alert in alertFilesTemp:
                    fileName = 'Image/' + subjectFolder + '/' + contentFolder + '/' + alert
                    alertFiles.append(fileName)

    return drowsyFiles, alertFiles


def extractFeatures(d):
    leftEye = d[lStart:lEnd]
    rightEye = d[rStart:rEnd]
    mouth = d[mstart:mend]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0

    mar = mouth_aspect_ratio(mouth)
    lCircularity = circularity(leftEye)
    rCircularity = circularity(rightEye)
    cir = (lCircularity + rCircularity) / 2.0
    moe = mouth_over_eye(mar, ear)
    lipDis = lip_distance(d)

    return ear, mar, cir, moe, lipDis


drowsyImages, alertImages = resolveFiles(next(os.walk('Image'))[1])

features = []

drowsyImages.sort()
alertImages.sort()

# print(alertImages)
#
# # Build alert dataset
print("Building alert images")
for alertImage in alertImages:
    image = cv2.imread(alertImage)
    print(image.shape)
    image = cv2.resize(image, (480, 640))
    print(image.shape)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    print(rects)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        print(shape)

        if shape is not None:
            if sum(sum(shape)) != 0:
                ear, mar, cir, moe, lipDis = extractFeatures(shape)
                print(alertImage)
                # print(image.shape)
                print(ear, mar, cir, moe, lipDis)
                features.append([ear, mar, cir, moe, lipDis, 0.0])

print("Alert images successfully added")
features = np.array(features)
print('Saving features document')
np.savetxt("data/dataSetNew/featuresDataAlert.csv", features, delimiter=",")

features = []

# Build drowsy dataset
print("Building drowsy images")
for drowsyImage in drowsyImages:
    image = cv2.imread(drowsyImage)
    image = cv2.resize(image, (480, 640))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        if shape is not None:
            if sum(sum(shape)) != 0:
                ear, mar, cir, moe, lipDis = extractFeatures(shape)
                print(drowsyImage)
                # print(image.shape)
                print(ear, mar, cir, moe, lipDis)
                features.append([ear, mar, cir, moe, lipDis, 1.0])

print("Drowsy images successfully added")

features = np.array(features)
print('Saving features document')
np.savetxt("data/dataSetNew/featuresDataDrowsy.csv", features, delimiter=",")
