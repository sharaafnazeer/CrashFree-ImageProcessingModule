import os
import cv2
import numpy as np
from imutils import face_utils
from mlxtend.image import extract_face_landmarks

from featureExtraction.feature import eye_aspect_ratio, mouth_aspect_ratio, circularity, mouth_over_eye, lip_distance

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
    image = cv2.resize(image, (480, 640), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(alertImage)
    print(image.shape)
    try:
        landmarks = extract_face_landmarks(gray)
        if landmarks is not None:
            if sum(sum(landmarks)) != 0:
                # data.append(landmarks)
                # labels.append(0.0)
                ear, mar, cir, moe, lipDis = extractFeatures(landmarks)
                print(ear, mar, cir, moe, lipDis)
                features.append([ear, mar, cir, moe, lipDis, 0.0])
    except:
        print("An exception occurred" + alertImage)

print("Alert images successfully added")
features = np.array(features)
print('Saving features document')
np.savetxt("data/dataSetNew/featuresDataAlert.csv", features, delimiter=",")

features = []

# Build drowsy dataset
print("Building drowsy images")
for drowsyImage in drowsyImages:
    image = cv2.imread(drowsyImage)
    image = cv2.resize(image, (480, 640), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(drowsyImage)
    print(image.shape)
    try:
        landmarks = extract_face_landmarks(gray)
        if landmarks is not None:
            if sum(sum(landmarks)) != 0:
                # data.append(landmarks)
                # labels.append(1.0)
                ear, mar, cir, moe, lipDis = extractFeatures(landmarks)
                print(ear, mar, cir, moe, lipDis)
                features.append([ear, mar, cir, moe, lipDis, 1.0])
    except:
        print("An exception occurred" + drowsyImage)

print("Drowsy images successfully added")

features = np.array(features)
print('Saving features document')
np.savetxt("data/dataSetNew/featuresDataDrowsy.csv", features, delimiter=",")

# print(data)
# print(labels)
#
# data = np.array(data)
# labels = np.array(labels)
