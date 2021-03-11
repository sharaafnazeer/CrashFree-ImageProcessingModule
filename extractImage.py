# Importing all necessary libraries
import cv2
import dlib
from os import makedirs
from os.path import exists
# project
from enhancement.contrastBrightness import adjustImage, gammaCorrection
from enhancement.labColor import convertToLAB

# Read the video from specified path
cam = cv2.VideoCapture("video/cut.mp4")
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

    # reading from frame
    ret, frame = cam.read()
    fps = cam.get(cv2.CAP_PROP_FPS)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    if ret:
        frame = cv2.resize(frame, (640, 480), fx=0, fy=0,
                           interpolation=cv2.INTER_CUBIC)
        # if video is still left continue creating images
        orgName = 'data/org_frame' + str(currentframe) + '.jpg'
        cbName = 'data/cb_frame' + str(currentframe) + '.jpg'
        gName = 'data/g_frame' + str(currentframe) + '.jpg'
        fName = 'data/final_frame' + str(currentframe) + '.jpg'
        fFaceName = 'data/final_frame_face' + str(currentframe) + '.jpg'

        if frameOffset == currentframe:
            print('Creating...' + orgName)
            # writing the extracted images
            cv2.imwrite(orgName, frame)

            imgOrg = cv2.imread(orgName, 1)

            # cbImage = adjustImage(imgOrg)
            # cv2.imwrite(cbName, cbImage)
            #
            # gImage = gammaCorrection(cbImage)
            # cv2.imwrite(gName, gImage)
            #
            # gImage = cv2.imread(orgName, 1)
            # l, a, b = convertToLAB(gImage)
            # clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
            # cl = clahe.apply(l)
            #
            # limg = cv2.merge((cl, a, b))
            # final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            #
            # cv2.imwrite(fName, final)

            # step1: Loads face detection model
            cnn_face_detector = dlib.cnn_face_detection_model_v1("models/dlib/mmod_human_face_detector.dat")

            # step2: loads the image
            image = cv2.imread(orgName)

            # step3: converts to gray image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # step4: detect faces using CNN model
            faces = cnn_face_detector(gray, 1)
            for faceRect in faces:
                rect = faceRect.rect
                x = rect.left()
                y = rect.top()
                w = rect.right() - x
                h = rect.bottom() - y

                # step5: draw rectangle around each face
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imwrite(fFaceName, image)


            frameOffset = frameOffset + frameLimit

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
