# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:41:56 2021

@author: Sharaaf.Nazeer
"""

import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
from PIL import Image
from helpers.faceCropper import FaceCropper
from featureExtraction.feature import eye_aspect_ratio, mouth_aspect_ratio, circularity, mouth_over_eye, lip_distance

p = "models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# the width and height in pixels of the saved image
crop_width = 108
# whether this is just a face crop (true) or whether we're trying to include other elements in the image. 
# Based on the shortest distance between the detected face square and the edge of the image
simple_crop = True

filename_inc = 1

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



drowsyImages, alertImages = resolveFiles(next(os.walk('Image'))[1])

features = []

drowsyImages.sort()
alertImages.sort()

# # Build cropped images
print("Building alert images")
for alertImage in alertImages:
    image = cv2.imread(alertImage)
    cropper = FaceCropper()
    arrayCount, face = cropper.generate(image, name="neededData/Images/" + alertImage)


