import cv2
import os
import numpy as np

facedata = 'facedata'
datasetXML = 'datasetXML'

cam = cv2.VideoCapture(0)
cam.set(3, 720)  # lebar
cam.set(4, 480)  # tinggi

faceDetector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read(datasetXML+'/datasetWajah.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ["Tidak diketahui", "Lugas", "Lugas 2"]

minWidth = 0.1 * cam.get(3)
minHeight = 0.1 * cam.get(4)

# proses pengulangan
while True:
    retV, frame = cam.read()  # read camera lalu dimasukan ke variabel frame
    frame = cv2.flip(frame, 1)
    gantikeGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detectMultiScale(framenya, scale faktor, min neighbors)
    faces = faceDetector.detectMultiScale(
        gantikeGray, 1.2, 5, minSize=(round(minWidth), round(minHeight)))

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        id, confidence = faceRecognizer.predict(
            gantikeGray[y:y+h, x:x+w])  # confidence = 0 (cocok bgt)

        if confidence <= 50:
            nameID = names[1]
            confidenceTxt = " {0}%".format(round(100-confidence))
        else:
            nameID = names[0]
            confidenceTxt = " {0}%".format(round(100-confidence))

        cv2.putText(frame, str(nameID), (x+5, y-5),
                    font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidenceTxt), (x+5, y+h-5),
                    font, 1, (0, 0, 255), 1)

    cv2.imshow('YOUR FACE', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break

print("EXIT")
cam.release()
cv2.destroyAllWindows()
