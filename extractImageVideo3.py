# import the necessary packages
from threading import Thread
from imutils import face_utils
import time
import dlib
import cv2

from os import makedirs
from os.path import exists

# project
from playsound import playsound

from enhancement.contrastBrightness import adjustImage
from utils.EAR import eye_aspect_ratio, lip_distance


def playSound():
    playsound("sounds/alarm1.mp3")


directory = "data"
try:
    # creating a folder named data
    if not exists(directory):
        makedirs(directory)

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# start the video stream thread
print("[INFO] starting video stream thread...")
cam = cv2.VideoCapture(0)
fileStream = False
time.sleep(1.0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/dlib/shape_predictor_68_face_landmarks.dat')
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

fps = cam.get(cv2.CAP_PROP_FPS)  # Frame rate of the camera

DROWSY_ALERT = False
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = fps * 40 / 100  # Initial Frame drowsy
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

YAWN_THRESH = 30

while cam.isOpened():
    # read the current frame
    ret, frame = cam.read()

    if ret:
        frame = cv2.resize(frame, (860, 540), fx=0, fy=0,
                           interpolation=cv2.INTER_CUBIC)

        adjustedImage = adjustImage(frame)
        cv2.imwrite('data/current.jpg', adjustedImage)

        savedImage = cv2.imread('data/current.jpg', 1)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        # savedImage = clahe.apply(savedImage)

        gray = cv2.cvtColor(savedImage, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # coordinates to compute the mouth aspect ratio for both eyes
            mouth = shape[mStart:mEnd]
            lipDistance = lip_distance(shape)

            # print(lipDistance)

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # if the alarm is not on, turn it on
                    if not DROWSY_ALERT:
                        DROWSY_ALERT = True
                        # T = Thread(target=playSound())  # create thread
                        # T.start()  # Launch created thread
                        # draw an alarm on the frame
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            elif lipDistance > YAWN_THRESH:
                print("ccc")

                if not DROWSY_ALERT:
                    DROWSY_ALERT = True
                    # draw an alarm on the frame
                    # T = Thread(target=playSound())  # create thread
                    # T.start()  # Launch created thread
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                DROWSY_ALERT = False
                COUNTER = 0

            # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "COUNTER: {:.2f}".format(COUNTER), (600, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
# do a bit of cleanup
cv2.destroyAllWindows()
cam.release()
