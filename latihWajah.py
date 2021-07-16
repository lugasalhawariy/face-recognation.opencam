import cv2
import os
import numpy as np
from PIL import Image

facedata = 'facedata'
datasetXML = 'datasetXML'


def getImageLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    faceIDs = []
    for imagePath in imagePaths:
        PILImg = Image.open(imagePath).convert('L')  # Convert ke dalam gray
        imgNum = np.array(PILImg, 'uint8')
        faceID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(imgNum)
        for (x, y, w, h) in faces:
            faceSamples.append(imgNum[y:y+h, x:x+w])
            faceIDs.append(faceID)
        return faceSamples, faceIDs


faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Mesin sedang melakukan training data wajah. Tunggu sejenak...")
faces, IDs = getImageLabel(facedata)
faceRecognizer.train(faces, np.array(IDs))

# lalu simpan dalam bentuk XML
faceRecognizer.write(datasetXML+'/datasetWajah.xml')
print("Sebanyak {0} data wajah telah di latih ke mesin.",
      format(len(np.unique(IDs))))
