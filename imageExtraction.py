import cv2
import numpy as np
from mlxtend.image import extract_face_landmarks

from featureExtraction.feature import eye_aspect_ratio, mouth_aspect_ratio, circularity, mouth_over_eye
from helpers.faceCropper import FaceCropper


def getFrame(sec):
    start = 180000
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start + sec * 1000)
    hasFrames, image = vidcap.read()
    # image = cv2.resize(image, (128, 128))
    return hasFrames, image


for j in [28]:

    print("Printing video series > %d" % j)
    for i in [0, 5, 10]:
        data = []
        labels = []
        images = []
        print("Printing video sub series > %d" % i)
        vidcap = cv2.VideoCapture('video/Fold3_part1/' + str(j) + '/' + str(i) + '.MOV')

        sec = 0
        frameRate = 1
        success, image = getFrame(sec)
        count = 0
        while success and count < 350:
            landmarks = extract_face_landmarks(image)
            # print(landmarks)
            if landmarks is not None:
                if sum(sum(landmarks)) != 0:
                    count += 1
                    cropper = FaceCropper()
                    arrayCount, face = cropper.generate(image, name="data/images/image_%d_%d_%d.jpg" % (j, i, count))

                    if arrayCount > 0:
                        data.append(landmarks)
                        labels.append([i])
                        images.append(face[0])

                    sec = sec + frameRate
                    sec = round(sec, 2)
                    success, image = getFrame(sec)
                    print(count)
                else:
                    sec = sec + frameRate
                    sec = round(sec, 2)
                    success, image = getFrame(sec)
                    print("face not detected****")
            else:
                sec = sec + frameRate
                sec = round(sec, 2)
                success, image = getFrame(sec)
                print("face not detected")

        data = np.array(data)
        labels = np.array(labels)
        images = np.array(images)
        # print(images.shape)
        reshapedImages = images.reshape(images.shape[0], -1)
        print(reshapedImages.shape)

        # load_original_arr = reshapedImages.reshape(
        #     reshapedImages.shape[0], reshapedImages.shape[1] // images.shape[3] // images.shape[2],
        #     reshapedImages.shape[1] // images.shape[3] // images.shape[2],
        #     images.shape[3])
        # print(load_original_arr.shape)
        #
        #
        # if (load_original_arr == images).all():
        #     print("Yes, both the arrays are same")
        # else:
        #     print("No, both the arrays are not same")

        features = []
        index = 0
        for d in data:
            eye = d[36:68]
            ear = eye_aspect_ratio(eye)
            mar = mouth_aspect_ratio(eye)
            cir = circularity(eye)
            mouth_eye = mouth_over_eye(eye)
            features.append([ear, mar, cir, mouth_eye, reshapedImages[index][0], labels[index][0]])
            index = index + 1

        features = np.array(features)
        print(features.shape)
        np.savetxt("data/Fold3_part1_features_labels_%d_%d.csv" % (j, i), features, delimiter=",")
        # np.savetxt("data/Fold3_part1_images.csv", reshapedImages, delimiter=",")
print("Extracted Successfully")

################ REFERENCE ##################

# arr = gfg.random.rand(5, 4, 3)
#
# # reshaping the array from 3D
# # matrice to 2D matrice.
# arr_reshaped = arr.reshape(arr.shape[0], -1)
#
# # saving reshaped array to file.
# gfg.savetxt("geekfile.txt", arr_reshaped)
#
# # retrieving data from file.
# loaded_arr = gfg.loadtxt("geekfile.txt")
#
# # This loadedArr is a 2D array, therefore
# # we need to convert it to the original
# # array shape.reshaping to get original
# # matrice with original shape.
# load_original_arr = loaded_arr.reshape(
#     loaded_arr.shape[0], loaded_arr.shape[1] // arr.shape[2], arr.shape[2])
#
# # check the shapes:
# print("shape of arr: ", arr.shape)
# print("shape of load_original_arr: ", load_original_arr.shape)
#
# # check if both arrays are same or not:
# if (load_original_arr == arr).all():
#     print("Yes, both the arrays are same")
# else:
#     print("No, both the arrays are not same")
