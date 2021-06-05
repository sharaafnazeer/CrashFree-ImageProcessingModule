import cv2
import dlib


class FaceCropper():
    def __init__(self, predictorPath):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictorPath)
        
    def runDetector(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)      
            if (len(rects) > 0):
              shape = self.predictor(gray,rects[0])
              return shape, gray, rects[0]
        except:
            pass
            
      
        return None, None, None

    def extractFace(self, faceRects, gray, size = 256): 
      print(faceRects)
      x = faceRects.left()
      y = faceRects.top()
      w = faceRects.right() - x
      h = faceRects.bottom() - y
    
      r = max(w, h) / 2
      centerx = x + w / 2
      centery = y + h / 2
      nx = int(centerx - r)
      ny = int(centery - r)
      nr = int(r * 2.1)
    
      face = gray[ny:ny + nr, nx:nx + nr]
      if face.shape[0] > 0 and face.shape[1] > 0:
        faceImage = cv2.resize(face, (size, size))
        return faceImage
      return None


    def extractEye(self, shape, gray, size = 256):
      x1Eye=shape.part(36).x
      x2Eye=shape.part(39).x
      y1Eye=shape.part(42).y
      y2Eye=shape.part(45).y    
      eye=gray[y2Eye-250:y1Eye+100,x1Eye-100:x2Eye+100]
      eyeImage = cv2.resize(eye, (size, size))   
      return eyeImage
  
    def extractMouth(self, shape, gray, size = 256):
      xmouthpoints = [shape.part(x).x for x in range(48,67)]
      ymouthpoints = [shape.part(x).y for x in range(48,67)]
      maxx = max(xmouthpoints)
      minx = min(xmouthpoints)
      maxy = max(ymouthpoints)
      miny = min(ymouthpoints) 
    
      mouth = gray[miny-15:maxy+15,minx-15:maxx+15]
      mouthImage = cv2.resize(mouth, (size, size))
      return mouthImage