import cv2
import sys
import os

import dlib


class FaceCropper(object):
    def __init__(self):
        self.cnn_face_detector = dlib.get_frontal_face_detector()

    def generate(self, img, name):
        imageArray = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.cnn_face_detector(gray, 1)

        if faces is None:
            print('Failed to detect face')
            return 0

        for rect in faces:
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y

            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = gray[ny:ny + nr, nx:nx + nr]
            print(faceimg.shape)
            if faceimg.shape[0] > 0 and faceimg.shape[1] > 0:
                print("cc")
                lastimg = cv2.resize(faceimg, (32, 32))
                imageArray.append(lastimg)
                cv2.imwrite(name, lastimg)

        return len(imageArray), imageArray
