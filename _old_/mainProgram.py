import pickle

import cv2
import dlib
import pandas as pd
from imutils import face_utils
from featureExtraction.feature import eye_aspect_ratio, circularity, mouth_aspect_ratio, mouth_over_eye, lip_distance

p = "models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


def model(landmarks, image, predictionModel):
    features = pd.DataFrame(columns=["EAR", "MAR", "Circularity", "MOE", "LIP_DIS"])

    leftEye = landmarks[lStart:lEnd]
    rightEye = landmarks[rStart:rEnd]
    mouth = landmarks[mstart:mend]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0

    mar = mouth_aspect_ratio(mouth)
    lCircularity = circularity(leftEye)
    rCircularity = circularity(rightEye)
    cir = (lCircularity + rCircularity) / 2.0
    moe = mouth_over_eye(mar, ear)
    lipDis = lip_distance(landmarks)

    print(ear, mar, cir, moe, lipDis)

    df = features.append({"EAR": ear, "MAR": mar, "Circularity": cir, "MOE": moe, "LIP_DIS": lipDis},
                         ignore_index=True)
    # print(df)

    Result = predictionModel.predict(df)
    print(Result[0])
    if Result[0] == 1:
        Result_String = "Drowsy"
    else:
        Result_String = "Alert"

    return Result_String, df.values


def loadModel():
    filename = 'models/drowsyPrediction4.dat'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def live():
    cam = cv2.VideoCapture(0)
    data = []
    result = []
    while cam.isOpened():

        # read the current frame
        (grabbed, image) = cam.read()
        # image = cv2.resize(image, (480, 640))
        print(image.shape)
        print(grabbed)

        # check frame is fetched or not
        if grabbed:
            # Converting the image to gray scale
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Get faces into webcam's image
            rects = detector(image, 0)
            print(rects)

            # For each detected face, find the landmark.
            for (i, rect) in enumerate(rects):
                # Make the prediction and transfom it to numpy array
                shape = predictor(image, rect)
                print(shape)
                shape = face_utils.shape_to_np(shape)
                Result_String, features = model(shape, image, loadModel())
                cv2.putText(image, Result_String, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                data.append(features)
                result.append(Result_String)

                # Draw on our image, all the finded cordinate points (x,y)
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # Show the image
            cv2.imshow("Output", image)

            k = cv2.waitKey(300) & 0xFF
            if k == 27:
                break

    cv2.destroyAllWindows()
    cam.release()

    return data, result


def testing():

    image = cv2.imread('Image/Sub 1/Alert/2021_03_19_15_27_IMG_4216.JPG')
    image = cv2.resize(image, (480, 640))
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces into webcam's image
    rects = detector(gray, 0)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        # print(shape)
        shape = face_utils.shape_to_np(shape)
        Result_String, features = model(shape, image, loadModel())
        cv2.putText(image, Result_String, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        # data.append(features)
        # result.append(Result_String)

        print(features, Result_String)
        # Draw on our image, all the finded cordinate points (x,y)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # Show the image

    cv2.imshow("Output", image)

    k = cv2.waitKey(0)

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 400)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

data, value = live()
print(value)

# testing()